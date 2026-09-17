"""Robust parsing of the free-text 'value' column from the extracted dataset
into a numeric float, handling ranges, scientific-notation variants, and
qualifier words -- while refusing to guess on genuinely non-numeric entries."""
import re

import numpy as np

_SCI_PATTERN = re.compile(
    r"([-+]?\d*\.?\d+)\s*[x×*]\s*10\s*\^?\s*([-+]?\d+)", re.IGNORECASE
)
_RANGE_PATTERN = re.compile(
    r"([-+]?\d*\.?\d+)\s*(?:-|to|–|−)\s*([-+]?\d*\.?\d+)", re.IGNORECASE
)
_PLAIN_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

NON_NUMERIC_MARKERS = {"not explicitly provided", "plotted", "maximum", "minimum", "variable", "unclear", ""}


def parse_numeric_value(raw) -> float:
    if raw is None:
        return np.nan
    s = str(raw).strip()
    if not s or s.lower() in NON_NUMERIC_MARKERS or s.lower() == "nan":
        return np.nan

    m = _SCI_PATTERN.search(s)
    if m:
        try:
            return float(m.group(1)) * (10 ** float(m.group(2)))
        except ValueError:
            pass

    m = _RANGE_PATTERN.search(s)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            return (a + b) / 2.0
        except ValueError:
            pass

    m = _PLAIN_FLOAT.search(s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return np.nan
    return np.nan
