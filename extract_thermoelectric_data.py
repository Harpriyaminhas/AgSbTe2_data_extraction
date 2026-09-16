"""
AgSbTe2-family thermoelectric doping/property extraction pipeline.

Reads every PDF in ./PDFs, extracts body text + tables (PyMuPDF/pdfplumber) and
rendered page images, and uses Google Gemini (text + vision, structured JSON
output) to pull out:
  - base composition, sample/dopant composition, dopant(s)/co-dopant(s),
    doped site (Ag/Sb/Te/interstitial), doping percentage
  - Seebeck coefficient, electrical conductivity, thermal conductivity,
    lattice thermal conductivity, power factor, ZT/zT -- each tagged with
    the temperature it was reported at, and whether it came from text,
    a table, or a figure (vision pass)

Resumable: per-PDF raw JSON is cached in output/raw_json/, so re-running
skips already-processed papers. Results are appended incrementally to the
long-format CSV as each paper finishes, so partial output is always usable.

Usage (from VS Code integrated terminal, in this folder):
    python3 extract_thermoelectric_data.py
    python3 extract_thermoelectric_data.py --limit 5        # smoke test
    python3 extract_thermoelectric_data.py --no-figures     # text/tables only
    python3 extract_thermoelectric_data.py --redo "some_pdf_stem"
"""
from __future__ import annotations

import argparse
import base64
import collections
import csv
import json
import math
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pymupdf
import pdfplumber
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()
import os
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "PDFs"
OUT_DIR = BASE_DIR / "output"
RAW_JSON_DIR = OUT_DIR / "raw_json"
IMG_DIR = OUT_DIR / "page_images"
LOG_DIR = OUT_DIR / "logs"
for d in (OUT_DIR, RAW_JSON_DIR, IMG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_TEXT = "gemini-3.1-flash-lite"
MODEL_VISION = "gemini-3.1-flash-lite"
RENDER_DPI = 130
MAX_PAGES_FOR_VISION = 14
VISION_BATCH_SIZE = 5
SLEEP_BETWEEN_CALLS = 0.0  # pacing is now handled by the shared rate limiter below
DEFAULT_WORKERS = 8  # PDFs processed concurrently (each PDF's own calls stay sequential)

# The API key is on the free tier: generate_content_free_tier_requests is capped
# at 20/minute SHARED across every model. Uncoordinated concurrent workers blow
# past that instantly and burn most of the wall-clock time on 429 backoff. This
# limiter paces every call (across all threads) to just under the quota so we
# get as close to the theoretical-minimum runtime as the quota allows.
RATE_LIMIT_CALLS_PER_MIN = 10

CSV_PATH = OUT_DIR / "AgSbTe2_Thermoelectric_Extracted_Data_long.csv"
XLSX_PATH = OUT_DIR / "AgSbTe2_Thermoelectric_Extracted_Data.xlsx"
RUN_LOG_PATH = LOG_DIR / "run_log.csv"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not found. Put it in a .env file in this folder.")

client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=90_000))

PROPERTY_CHOICES = [
    "Seebeck coefficient",
    "electrical conductivity",
    "thermal conductivity",
    "lattice thermal conductivity",
    "power factor",
    "ZT",
]
SITE_CHOICES = ["Ag", "Sb", "Te", "interstitial", "multiple sites", "unclear"]
SOURCE_CHOICES = ["text", "table", "figure"]


# --------------------------------------------------------------------------
# Structured-output schema (Gemini fills this in directly -> guaranteed JSON)
# --------------------------------------------------------------------------
class Dopant(BaseModel):
    element: str
    ratio_or_percentage: str
    doped_site: str
    site_evidence: str = ""


class TEProperty(BaseModel):
    property: str
    value: str
    unit: str
    temperature_value: str = ""
    temperature_unit: str = ""
    source_type: str
    source_detail: str = ""
    notes: str = ""


class Material(BaseModel):
    sample_id: str
    full_formula: str
    dopant_or_codopant: List[Dopant] = []
    thermoelectric_properties: List[TEProperty] = []


class ExtractionResult(BaseModel):
    base_composition: str = "AgSbTe2"
    materials: List[Material] = []
    extraction_notes: str = ""


EMPTY_RESULT = {"base_composition": "AgSbTe2", "materials": [], "extraction_notes": ""}

SCHEMA_DESC = """
Extract doping and thermoelectric-property data for AgSbTe2-family compounds
(parent compound AgSbTe2, possibly written as Ag1-xSbTe2, AgSb1-xMxTe2, etc.,
including solid solutions/alloys with SnTe, GeTe, PbTe, etc.).

For EACH distinct sample/composition reported in the given content:
1. Identify the dopant(s) or co-dopant(s): element/ion symbol, the ratio or
   at.%/wt.%/mol% EXACTLY as written, and which site it replaces (Ag, Sb, Te,
   interstitial, or 'multiple sites' if it substitutes on more than one
   sublattice, or 'unclear' if the paper does not specify). If TWO OR MORE
   distinct dopant elements are present in the same sample (co-doping /
   dual-doping / multi-doping), list ALL of them as separate entries in
   dopant_or_codopant for that sample.
2. Identify EVERY reported value of: Seebeck coefficient, electrical
   conductivity, thermal conductivity, lattice thermal conductivity, power
   factor, and ZT (zT) -- at EVERY temperature given (room temperature and
   all elevated temperatures in a table or on a text-vs-T plot), tagging the
   source (text / table / figure) and where it came from (e.g. 'Table 2',
   'Fig. 3b', 'body text').

Rules:
- Only extract values EXPLICITLY stated or plotted. Do not infer or compute
  new numbers. If a plot only shows a curve with no readable value, skip it
  rather than guessing.
- If the paper (or these pages) contain no AgSbTe2-family doping/TE data,
  return an empty materials list and say why in extraction_notes.
- Keep values/units exactly as reported (do not convert units).
- doped_site must be one of: Ag, Sb, Te, interstitial, multiple sites, unclear.
- property must be one of: Seebeck coefficient, electrical conductivity,
  thermal conductivity, lattice thermal conductivity, power factor, ZT.
"""

TEXT_PROMPT_TMPL = """You are a materials scientist extracting structured data from a
thermoelectrics journal article (or its supplementary information) about
AgSbTe2-family compounds.

{schema}

--- ARTICLE TEXT ---
{text}

--- TABLES FOUND IN THE PDF (markdown) ---
{tables}
"""

FIGURE_PROMPT_TMPL = """You are a materials scientist. Look ONLY at plots/graphs in these
page images that show Seebeck coefficient, electrical conductivity, thermal
conductivity, lattice thermal conductivity, power factor, or ZT as a function
of temperature, for an AgSbTe2-family compound (possibly doped/co-doped).

For each such plot, read off approximate values at each visually
distinguishable temperature point/curve. This is a visual estimate, not an
exact number from text -- set source_type to "figure" and mention in "notes"
that the value was visually estimated from a figure, plus the figure number/
caption if visible on the page.

{schema}

If no relevant property-vs-temperature plots appear on these pages, return
an empty materials list.
"""

class RateLimiter:
    """Thread-safe sliding-window limiter: at most `max_calls` acquire() calls
    complete in any trailing `period` seconds, across all callers."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps = collections.deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] > self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_for = self.period - (now - self._timestamps[0]) + 0.05
            time.sleep(max(wait_for, 0.05))


rate_limiter = RateLimiter(RATE_LIMIT_CALLS_PER_MIN, period=60.0)

RETRYABLE = (
    genai_errors.ServerError,
    genai_errors.ClientError,
    TimeoutError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        # only retry rate-limit (429); other 4xx (bad request, 404) are not retryable
        return "429" in str(exc)
    if isinstance(exc, genai_errors.ServerError):
        return True
    # network-level timeouts from httpx/requests surface as generic exceptions
    name = type(exc).__name__
    return name in ("ReadTimeout", "ConnectTimeout", "ConnectionError", "RemoteProtocolError")


@retry(
    stop=stop_after_attempt(9),
    wait=wait_exponential(multiplier=3, min=4, max=120),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini(**kwargs):
    rate_limiter.acquire()
    try:
        return client.models.generate_content(**kwargs)
    except Exception as e:
        if _is_retryable(e):
            raise
        # not retryable -> re-raise immediately as a non-retried failure
        raise RuntimeError(f"non-retryable: {e}") from e


def call_text_pass(text: str, tables_md: str) -> ExtractionResult:
    prompt = TEXT_PROMPT_TMPL.format(schema=SCHEMA_DESC, text=text[:150_000], tables=tables_md[:40_000] or "(none found)")
    resp = _call_gemini(
        model=MODEL_TEXT,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            temperature=0,
        ),
    )
    return resp.parsed if resp.parsed is not None else ExtractionResult(**json.loads(resp.text))


def call_vision_pass(image_paths: List[Path]) -> ExtractionResult:
    parts = [types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png") for p in image_paths]
    prompt = FIGURE_PROMPT_TMPL.format(schema=SCHEMA_DESC)
    resp = _call_gemini(
        model=MODEL_VISION,
        contents=[prompt, *parts],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            temperature=0,
        ),
    )
    return resp.parsed if resp.parsed is not None else ExtractionResult(**json.loads(resp.text))


# --------------------------------------------------------------------------
# PDF content extraction
# --------------------------------------------------------------------------
def safe_name(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return s[:150]


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>()\[\]]+", re.IGNORECASE)


def extract_doi(pdf_path: Path) -> str:
    """Best-effort DOI lookup: regex over the first few pages' text, then PDF metadata."""
    try:
        doc = pymupdf.open(pdf_path)
        for i, page in enumerate(doc):
            if i >= 3:
                break
            for raw in DOI_RE.findall(page.get_text()):
                doi = raw.strip().rstrip(".,;:")
                if 8 < len(doi) < 100:
                    doc.close()
                    return doi
        meta_doi = (doc.metadata or {}).get("subject", "") or ""
        doc.close()
        m = DOI_RE.search(meta_doi)
        if m:
            return m.group(0).strip().rstrip(".,;:")
    except Exception:
        pass
    return ""


def extract_pdf_content(pdf_path: Path, name: str):
    text_parts = []
    doc = pymupdf.open(pdf_path)
    fig_dir = IMG_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    page_images: List[Path] = []
    for i, page in enumerate(doc):
        text_parts.append(f"\n\n--- PAGE {i + 1} ---\n" + page.get_text())
        img_path = fig_dir / f"page_{i + 1:02d}.png"
        if not img_path.exists():
            pix = page.get_pixmap(dpi=RENDER_DPI)
            pix.save(img_path)
        page_images.append(img_path)
    n_pages = len(doc)
    doc.close()
    full_text = "".join(text_parts)

    tables_md = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for t_idx, table in enumerate(tables or []):
                    if not table or len(table) < 2:
                        continue
                    rows_md = [
                        "| " + " | ".join((c or "").replace("\n", " ") for c in row) + " |"
                        for row in table
                    ]
                    tables_md.append(f"\n[Table on PDF page {i + 1}, table #{t_idx + 1}]\n" + "\n".join(rows_md))
    except Exception as e:
        tables_md.append(f"(pdfplumber table extraction failed: {e})")

    return {
        "text": full_text,
        "tables_md": "\n".join(tables_md),
        "page_images": page_images,
        "n_pages": n_pages,
    }


def merge_results(a: dict, b: ExtractionResult, tag: str) -> None:
    """Append b's materials into dict a in-place; a follows EMPTY_RESULT shape."""
    if b.base_composition and b.base_composition != "AgSbTe2":
        a["base_composition"] = b.base_composition
    for m in b.materials:
        a["materials"].append(m.model_dump())
    if b.extraction_notes:
        a["extraction_notes"] += f"[{tag}] {b.extraction_notes} "


# --------------------------------------------------------------------------
# Per-PDF pipeline
# --------------------------------------------------------------------------
def process_one_pdf(pdf_path: Path, run_figures: bool, force: bool = False) -> dict:
    name = safe_name(pdf_path.stem)
    raw_out_path = RAW_JSON_DIR / f"{name}.json"
    if raw_out_path.exists() and not force:
        cached = json.loads(raw_out_path.read_text(encoding="utf-8"))
        if not cached.get("doi"):
            cached["doi"] = extract_doi(pdf_path)
            raw_out_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
        return cached

    combined = {
        "source_pdf": pdf_path.name,
        "doi": extract_doi(pdf_path),
        "base_composition": "AgSbTe2",
        "materials": [],
        "extraction_notes": "",
        "status": "ok",
        "n_pages": None,
    }

    try:
        content = extract_pdf_content(pdf_path, name)
        combined["n_pages"] = content["n_pages"]

        # --- text + table pass ---
        try:
            result = call_text_pass(content["text"], content["tables_md"])
            merge_results(combined, result, "text-pass")
        except Exception as e:
            combined["extraction_notes"] += f"[text-pass ERROR: {e}] "
            combined["status"] = "partial-error"

        time.sleep(SLEEP_BETWEEN_CALLS)

        # --- figure/vision pass ---
        if run_figures and content["page_images"]:
            pages = content["page_images"][:MAX_PAGES_FOR_VISION]
            for i in range(0, len(pages), VISION_BATCH_SIZE):
                batch = pages[i:i + VISION_BATCH_SIZE]
                try:
                    result = call_vision_pass(batch)
                    merge_results(combined, result, f"figure-pass p{i + 1}-{i + len(batch)}")
                except Exception as e:
                    combined["extraction_notes"] += f"[figure-pass p{i + 1}-{i + len(batch)} ERROR: {e}] "
                    combined["status"] = "partial-error"
                time.sleep(SLEEP_BETWEEN_CALLS)

    except Exception as e:
        combined["status"] = "failed"
        combined["extraction_notes"] += f"[FATAL: {e}] "
        combined["_traceback"] = traceback.format_exc()

    raw_out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


# --------------------------------------------------------------------------
# Flatten to long-format rows
# --------------------------------------------------------------------------
LONG_COLUMNS = [
    "source_pdf", "doi", "base_composition", "sample_id", "full_formula",
    "num_dopants", "is_codoped",
    "primary_dopant_element", "primary_dopant_pct", "primary_dopant_site",
    "codopant_elements", "all_dopants_detail",
    "property", "value", "unit",
    "temperature_value", "temperature_unit",
    "source_type", "source_detail", "notes",
]


def flatten_one(record: dict) -> List[dict]:
    rows = []
    source_pdf = record.get("source_pdf", "")
    doi = record.get("doi", "")
    base_comp = record.get("base_composition", "AgSbTe2")
    for mat in record.get("materials", []):
        sample_id = mat.get("sample_id", "")
        full_formula = mat.get("full_formula", "")
        dopants = mat.get("dopant_or_codopant", []) or []
        n_dop = len(dopants)
        primary = dopants[0] if dopants else {}
        codopants = dopants[1:] if n_dop > 1 else []
        codopant_str = "; ".join(
            f"{d.get('element', '?')} ({d.get('ratio_or_percentage', '?')}) @ {d.get('doped_site', '?')}"
            for d in codopants
        )
        all_detail = "; ".join(
            f"{d.get('element', '?')} ({d.get('ratio_or_percentage', '?')}) @ {d.get('doped_site', '?')} site"
            for d in dopants
        )
        base_row = {
            "source_pdf": source_pdf,
            "doi": doi,
            "base_composition": base_comp,
            "sample_id": sample_id,
            "full_formula": full_formula,
            "num_dopants": n_dop,
            "is_codoped": "Yes" if n_dop > 1 else "No",
            "primary_dopant_element": primary.get("element", ""),
            "primary_dopant_pct": primary.get("ratio_or_percentage", ""),
            "primary_dopant_site": primary.get("doped_site", ""),
            "codopant_elements": codopant_str,
            "all_dopants_detail": all_detail,
        }
        props = mat.get("thermoelectric_properties", []) or []
        if not props:
            rows.append({**base_row, "property": "", "value": "", "unit": "",
                         "temperature_value": "", "temperature_unit": "",
                         "source_type": "", "source_detail": "", "notes": ""})
        for p in props:
            rows.append({
                **base_row,
                "property": p.get("property", ""),
                "value": p.get("value", ""),
                "unit": p.get("unit", ""),
                "temperature_value": p.get("temperature_value", ""),
                "temperature_unit": p.get("temperature_unit", ""),
                "source_type": p.get("source_type", ""),
                "source_detail": p.get("source_detail", ""),
                "notes": p.get("notes", ""),
            })
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N PDFs (smoke test)")
    ap.add_argument("--no-figures", action="store_true", help="Skip the vision/figure-reading pass")
    ap.add_argument("--redo", type=str, default=None, help="Re-process only PDFs whose filename contains this substring, ignoring cache")
    ap.add_argument("--redo-failed", action="store_true", help="Also force re-processing of any cached PDF whose status wasn't 'ok' (e.g. hit a 429/quota error last run)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of PDFs to process concurrently")
    args = ap.parse_args()

    run_figures = not args.no_figures

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.redo:
        pdfs = [p for p in pdfs if args.redo.lower() in p.name.lower()]
        force = True
    else:
        force = False
    if args.limit:
        pdfs = pdfs[: args.limit]

    def needs_forced_redo(pdf_path: Path) -> bool:
        if not args.redo_failed:
            return False
        raw_path = RAW_JSON_DIR / f"{safe_name(pdf_path.stem)}.json"
        if not raw_path.exists():
            return False
        try:
            return json.loads(raw_path.read_text(encoding="utf-8")).get("status") != "ok"
        except Exception:
            return True

    print(f"Processing {len(pdfs)} PDF(s) with {args.workers} concurrent worker(s). Figure/vision pass: {run_figures}. Model: {MODEL_TEXT}")

    is_new_csv = not CSV_PATH.exists()
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=LONG_COLUMNS)
    if is_new_csv:
        writer.writeheader()
    csv_lock = threading.Lock()

    run_log_rows = []
    all_rows_for_xlsx = []
    # if resuming, preload existing CSV rows for the final Excel export
    if not is_new_csv:
        try:
            all_rows_for_xlsx = pd.read_csv(CSV_PATH).to_dict("records")
        except Exception:
            all_rows_for_xlsx = []

    def handle_one(pdf_path: Path):
        try:
            force_this = force or needs_forced_redo(pdf_path)
            record = process_one_pdf(pdf_path, run_figures=run_figures, force=force_this)
        except Exception as e:
            record = {"source_pdf": pdf_path.name, "status": "failed",
                      "extraction_notes": f"[uncaught: {e}]", "materials": [],
                      "base_composition": "AgSbTe2", "n_pages": None}
            traceback.print_exc()
        rows = flatten_one(record)
        log_row = {
            "source_pdf": pdf_path.name,
            "status": record.get("status", "unknown"),
            "n_pages": record.get("n_pages"),
            "n_materials": len(record.get("materials", [])),
            "extraction_notes": (record.get("extraction_notes", "") or "")[:500],
        }
        return rows, log_row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(handle_one, p): p for p in pdfs}
        for fut in tqdm(as_completed(futures), total=len(pdfs), desc="Extracting"):
            rows, log_row = fut.result()
            with csv_lock:
                for r in rows:
                    writer.writerow(r)
                csv_file.flush()
                all_rows_for_xlsx.extend(rows)
                run_log_rows.append(log_row)

    csv_file.close()

    # de-dup + rewrite clean CSV/XLSX from everything accumulated so far
    df = pd.DataFrame(all_rows_for_xlsx, columns=LONG_COLUMNS)
    df = df.drop_duplicates()
    df.to_csv(CSV_PATH, index=False)

    run_log_df = pd.DataFrame(run_log_rows)
    run_log_df.to_csv(RUN_LOG_PATH, index=False)

    materials_summary = (
        df[df["sample_id"] != ""]
        .drop_duplicates(subset=["source_pdf", "sample_id", "full_formula"])
        [["source_pdf", "doi", "base_composition", "sample_id", "full_formula", "num_dopants",
          "is_codoped", "primary_dopant_element", "primary_dopant_pct", "primary_dopant_site",
          "codopant_elements", "all_dopants_detail"]]
    )

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer_x:
        df.to_excel(writer_x, sheet_name="Extracted_Data", index=False)
        materials_summary.to_excel(writer_x, sheet_name="Materials_Summary", index=False)
        run_log_df.to_excel(writer_x, sheet_name="Run_Log", index=False)

    print("\nDone.")
    print(f" - Long-format CSV : {CSV_PATH}  ({len(df)} rows)")
    print(f" - Excel workbook  : {XLSX_PATH}")
    print(f" - Run log         : {RUN_LOG_PATH}")
    print(f" - Raw per-PDF JSON: {RAW_JSON_DIR}")


if __name__ == "__main__":
    main()
