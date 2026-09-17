"""Formula parsing + composition-based descriptor computation for AgSbTe2-family samples."""
import math
import re

import numpy as np
from pymatgen.core import Composition, Element

R_GAS = 8.314462618  # J/mol.K


def clean_formula(raw: str) -> str:
    s = str(raw).strip()
    s = s.replace("−", "-").replace("‐", "-").replace("–", "-")
    s = s.replace("−", "-")
    s = re.sub(r"\s+", "", s)
    return s


def _substitute_x_placeholder(s: str) -> str:
    """'(...)0.6(AgSbTe2)0.4,x=0.10' -> substitute the literal x/X template
    variable with the value given after a trailing 'x=VALUE' (or 'x = VALUE'),
    then drop the trailing clause."""
    m = re.search(r"[,;]?x=(-?\d*\.?\d+)$", s, re.IGNORECASE)
    if not m:
        return s
    val = m.group(1)
    body = s[:m.start()]
    # no periodic-table element symbol contains a lowercase 'x', so every
    # lowercase 'x' here is safely the template variable, however it's
    # written (standalone, or immediately after an element like 'Pbx')
    body = body.replace("x", val)
    return body


def _fix_parenthesized_coefficients(s: str) -> str:
    """'Te(0.995)(AgSbTe2)0.005' -> 'Te0.995(AgSbTe2)0.005' -- a bare number in
    parentheses directly after an element symbol is a coefficient, not a
    sub-formula group."""
    return re.sub(r"([A-Z][a-z]?)\((\d*\.?\d+)\)", r"\1\2", s)


def _evaluate_inline_arithmetic(s: str) -> str:
    """'Sb1.1-0.9Pb0.1' -> evaluate the '1.1-0.9' coefficient arithmetic that
    sometimes survives extraction (a subtraction left unresolved in the
    source text) so pymatgen sees a plain number."""
    def repl(m):
        try:
            return f"{float(m.group(1)) - float(m.group(2)):.6f}"
        except ValueError:
            return m.group(0)
    return re.sub(r"(\d+\.?\d*)-(\d+\.?\d*)(?=[A-Z(]|$)", repl, s)


def formula_to_composition(raw: str):
    """Best-effort parse of a reported formula string into a pymatgen Composition
    with fractional element amounts. Returns None if unparseable."""
    s = clean_formula(raw)
    if not s:
        return None
    candidates = [s]
    s2 = _substitute_x_placeholder(s)
    s2 = _fix_parenthesized_coefficients(s2)
    candidates.append(s2)
    candidates.append(_evaluate_inline_arithmetic(s2))
    for cand in candidates:
        try:
            c = Composition(cand)
            if len(c) > 0 and c.num_atoms > 0:
                return c
        except Exception:
            continue
    return None


_SITE_BASE_AMOUNT = {"Ag": 1.0, "Sb": 1.0, "Te": 2.0}
_CODOPANT_RE = re.compile(r"([A-Z][a-z]?)\s*\(([^)]*)\)\s*@\s*([A-Za-z]+)")


def reconstruct_formula_from_fields(base_composition: str, primary_element: str, primary_pct,
                                     primary_site: str, codopant_str: str = None):
    """Fallback used only when the reported full_formula string itself can't
    be parsed: rebuild a composition directly from the already-extracted
    structured fields (dopant element, ratio, site) rather than dropping the
    row. Only applied when the site is an unambiguous Ag/Sb/Te substitution
    (never guessed for 'unclear'/'multiple sites'/interstitial), so no site
    assignment is invented that the extraction didn't already state."""
    if primary_site not in _SITE_BASE_AMOUNT:
        return None
    frac = parse_fraction(primary_pct)
    if not (frac == frac) or not (0 < frac < 1):  # NaN check + sanity range
        return None
    if not primary_element or str(primary_element).strip().lower() in ("", "nan"):
        return None

    amounts = dict(_SITE_BASE_AMOUNT)
    dopants = [(str(primary_element).strip(), primary_site, frac)]

    if codopant_str and str(codopant_str).strip().lower() not in ("", "nan"):
        for el, pct_str, site in _CODOPANT_RE.findall(str(codopant_str)):
            if site not in _SITE_BASE_AMOUNT:
                continue
            co_frac = parse_fraction(pct_str)
            if co_frac == co_frac and 0 < co_frac < 1:
                dopants.append((el, site, co_frac))

    for el, site, frac in dopants:
        amounts[site] = amounts[site] * (1 - frac)
        add_amt = _SITE_BASE_AMOUNT[site] * frac
        amounts[el] = amounts.get(el, 0.0) + add_amt

    formula = "".join(f"{el}{amt:.5f}" for el, amt in amounts.items() if amt > 1e-6)
    return formula_to_composition(formula)


def parse_temperature_to_kelvin(value, unit):
    try:
        v = float(str(value).strip())
    except (ValueError, TypeError):
        return np.nan
    u = str(unit).strip().lower()
    if u in ("c", "°c", "degc", "celsius"):
        return v + 273.15
    return v  # assume Kelvin (K, or already blank/unspecified defaults to as-given)


def parse_fraction(raw) -> float:
    """Parse a doping-ratio string ('2 at%', 'x=0.02', '5 mol%', '0.05') to a
    unitless fraction (0-1 scale). Returns NaN if no number can be recovered."""
    if raw is None:
        return np.nan
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    val = float(m.group(0))
    if "%" in s:
        return val / 100.0
    # bare numbers like 'x=0.02' or '0.05' are already fractional; values > 1
    # with no % sign are almost always at.% written without the sign (e.g. '2')
    if val > 1.0:
        return val / 100.0
    return val


def mixing_entropy(comp: Composition) -> float:
    fracs = [amt / comp.num_atoms for amt in comp.get_el_amt_dict().values() if amt > 0]
    return -R_GAS * sum(c * math.log(c) for c in fracs)


def ionic_character(comp: Composition) -> float:
    """Composition-weighted mean pairwise electronegativity difference,
    matching the descriptor doc's IC ~ |chi_A - chi_B| definition."""
    el_amt = comp.get_el_amt_dict()
    n = comp.num_atoms
    items = [(Element(el), amt / n) for el, amt in el_amt.items()]
    try:
        chis = [(el.X, frac) for el, frac in items if el.X is not None]
    except Exception:
        return np.nan
    if len(chis) < 2:
        return 0.0
    total, weight = 0.0, 0.0
    for i in range(len(chis)):
        for j in range(i + 1, len(chis)):
            xi, fi = chis[i]
            xj, fj = chis[j]
            w = fi * fj
            total += w * abs(xi - xj)
            weight += w
    return total / weight if weight > 0 else 0.0


_HOST_SITE_CHARGE = {"Ag": 1, "Sb": 3, "Te": -2}


def dopant_host_mismatch(dopant_symbol: str, site: str):
    """Charge and Shannon-ionic-radius mismatch between a dopant and the host
    ion (Ag+, Sb3+, or Te2-) it substitutes. Operationalizes the Ag+/Sb3+
    cation-disorder design axis central to Biswas-group AgSbTe2 work (e.g.
    Roychowdhury et al., Science 2021; Hg/Cd-doping studies): dopants whose
    charge and size are close to the host ion promote cation ordering and
    nanoscale coherent strain, a major lever on both carrier transport and
    lattice thermal conductivity. Returns (charge_mismatch, radius_mismatch),
    NaN where the site is interstitial/multiple/unclear or radii are unknown.
    """
    site = str(site or "").strip()
    if site not in _HOST_SITE_CHARGE or not dopant_symbol or str(dopant_symbol).strip() in ("", "nan"):
        return np.nan, np.nan
    host_charge = _HOST_SITE_CHARGE[site]
    try:
        host_radius = Element(site).data["Ionic radii"][str(host_charge)]
    except Exception:
        return np.nan, np.nan
    try:
        dop_el = Element(str(dopant_symbol).strip())
    except Exception:
        return np.nan, np.nan
    oxi_states = dop_el.common_oxidation_states or ()
    if not oxi_states:
        return np.nan, np.nan
    dop_charge = min(oxi_states, key=lambda x: abs(x - host_charge))
    charge_mismatch = abs(dop_charge - host_charge)

    radii = dop_el.data.get("Ionic radii", {})
    if not radii:
        return charge_mismatch, np.nan
    avail = {int(k): v for k, v in radii.items()}
    closest_state = min(avail, key=lambda k: abs(k - dop_charge))
    dop_radius = avail[closest_state]
    return charge_mismatch, abs(dop_radius - host_radius)


def dopant_site_fractions(comp: Composition) -> dict:
    """Ag / Sb / Te fraction of the composition (0 if absent) -- cheap,
    directly meaningful descriptors for a rock-salt AgSbTe2-family alloy."""
    el_amt = comp.get_el_amt_dict()
    n = comp.num_atoms
    return {
        "frac_Ag": el_amt.get("Ag", 0.0) / n,
        "frac_Sb": el_amt.get("Sb", 0.0) / n,
        "frac_Te": el_amt.get("Te", 0.0) / n,
    }
