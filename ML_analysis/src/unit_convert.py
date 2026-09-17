"""
Unit harmonization for the messy free-text 'unit' strings in the extracted
dataset. Values reported in different unit conventions differ by 10-10,000x
for the same physical quantity, which silently wrecks regression if left
unconverted. Canonical targets:
  seebeck_coefficient      -> uV/K
  electrical_conductivity  -> S/cm   (resistivity units are inverted; mobility
                                       units, e.g. cm2/V/s, are a different
                                       physical quantity and are dropped)
  thermal_conductivity     -> W/m.K
  power_factor             -> uW/cm.K2
  ZT                        -> dimensionless (no conversion needed)

Unrecognized unit strings return None (row dropped upstream) rather than
guessed, so a bad guess never silently enters the training data.
"""
import re

_PREFIX_MULT = {"u": 1e-6, "µ": 1e-6, "μ": 1e-6, "l": 1e-6,  # 'l' = OCR glitch for µ
                "m": 1e-3, "n": 1e-9, "k": 1e3}


def _normalize(raw: str) -> str:
    s = str(raw).strip().lower()
    s = s.replace("micro-", "u").replace("micro", "u")
    if s.startswith("x10"):
        s = s[1:]
    repl = {
        "µ": "u", "μ": "u", "·": "", "×": "", "−": "-",
        "⁻¹": "-1", "⁻²": "-2", "∙": "", "°": "",
        "Ω": "ohm", "Ω": "ohm", "ω": "ohm", "⋅": "", " ": "", "^": "",
        "/": "", "*": "", "(": "", ")": "",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    # a hyphen immediately followed by a letter is a separator (e.g. "ohm-cm");
    # a hyphen followed by a digit is a negative exponent (e.g. "cm-1") -- only strip the former
    s = re.sub(r"-(?=[a-z])", "", s)
    return s


def _sci_prefix(s: str):
    m = re.match(r"10(-?\d+)", s)
    if m:
        return 10 ** int(m.group(1)), s[m.end():]
    return 1.0, s


def _strip_prefix(s: str, base_chars: tuple):
    """If s starts with a known metric-prefix letter immediately followed by
    one of base_chars, return (multiplier, remainder-after-base-char)."""
    for p, mult in _PREFIX_MULT.items():
        if s.startswith(p) and len(s) > len(p) and s[len(p)] in base_chars:
            return mult, s[len(p):]
    if s and s[0] in base_chars:
        return 1.0, s
    return None, s


def convert_seebeck(value: float, raw_unit: str):
    s = _normalize(raw_unit)
    sci, s = _sci_prefix(s)
    prefix, s = _strip_prefix(s, ("v",))
    if prefix is None or not s.startswith("v"):
        return None
    volts = value * sci * prefix
    return volts * 1e6  # -> uV/K (per-K vs per-C treated as numerically equivalent)


def convert_electrical_conductivity(value: float, raw_unit: str):
    s = _normalize(raw_unit)
    if "v-1s-1" in s or ("cm2" in s and "v-1" in s):
        return None  # Hall mobility, not conductivity -- wrong physical quantity
    sci, s = _sci_prefix(s)
    has_ohm = "ohm" in s
    # 'ohm' (or Omega) immediately followed by an inverse exponent (ohm-1 /
    # ohm^-1, already stripped of '^') is Siemens-equivalent, NOT resistivity;
    # bare 'ohm'/'ohm*length' with no inverse exponent IS a resistivity unit.
    already_inverted = bool(re.search(r"ohm-?1", s))
    is_resistivity = has_ohm and not already_inverted
    base_chars = ("s",) if not has_ohm else ("o",)
    prefix, s = _strip_prefix(s, base_chars)
    if prefix is None:
        return None
    if is_resistivity:
        if not s.startswith("ohm"):
            return None
        rest = s[len("ohm"):]
        length_is_m = rest == "m"
        length_is_cm = rest == "cm"
        if not (length_is_m or length_is_cm):
            return None
        ohm_len = value * sci * prefix
        ohm_cm = ohm_len * (100.0 if length_is_m else 1.0)
        if ohm_cm == 0:
            return None
        return 1.0 / ohm_cm  # S/cm
    else:
        base_tag = "ohm" if has_ohm else "s"
        if not s.startswith(base_tag):
            return None
        rest = s[len(base_tag):]
        rest = rest.replace("-1", "", 1)  # drop the already-consumed inverse exponent, if present
        length_is_cm = rest == "cm"
        length_is_m = rest == "m"
        if not (length_is_cm or length_is_m):
            return None
        s_per_len = value * sci * prefix
        return s_per_len * (1.0 if length_is_cm else 0.01)  # S/m -> S/cm


def convert_thermal_conductivity(value: float, raw_unit: str):
    s = _normalize(raw_unit)
    sci, s = _sci_prefix(s)
    prefix, s = _strip_prefix(s, ("w",))
    if prefix is None or not s.startswith("w"):
        return None
    rest = s[1:]
    length_is_cm = "cm" in rest
    length_is_m = (not length_is_cm) and "m" in rest.replace("mk", "m").replace("degcm", "")
    if "deg-cm" in s or "degcm" in rest:  # 'W/deg-cm' style
        length_is_cm, length_is_m = True, False
    if not (length_is_cm or length_is_m):
        return None
    w_per_len_k = value * sci * prefix
    return w_per_len_k * (100.0 if length_is_cm else 1.0)  # -> W/m.K


def convert_power_factor(value: float, raw_unit: str):
    s = _normalize(raw_unit)
    sci, s = _sci_prefix(s)
    prefix, s = _strip_prefix(s, ("w",))
    if prefix is None or not s.startswith("w"):
        return None
    rest = s[1:]
    length_is_cm = "cm" in rest
    length_is_m = (not length_is_cm) and "m" in rest
    if not (length_is_cm or length_is_m):
        return None
    watts_per_len_k2 = value * sci * prefix
    w_per_cm_k2 = watts_per_len_k2 * (1.0 if length_is_cm else 0.01)
    return w_per_cm_k2 * 1e6  # W -> uW


CONVERTERS = {
    "seebeck_coefficient": convert_seebeck,
    "electrical_conductivity": convert_electrical_conductivity,
    "thermal_conductivity": convert_thermal_conductivity,
    "power_factor": convert_power_factor,
    "ZT": lambda value, raw_unit: value,  # already dimensionless
}

# Generous but physically-grounded plausibility bounds for AgSbTe2-family
# chalcogenide thermoelectrics. Values outside these are far more likely to
# be an extraction-time unit/decimal mislabeling (e.g. a value that's really
# ~1000 uV/K tagged as "mV/K", giving an impossible 1e6 uV/K after correct
# arithmetic conversion) than a genuine physical measurement -- dropped and
# logged rather than left in to corrupt model training.
PLAUSIBLE_RANGE = {
    "seebeck_coefficient": (-1000.0, 1000.0),       # uV/K
    "electrical_conductivity": (1e-3, 5e4),          # S/cm
    "thermal_conductivity": (0.05, 20.0),            # W/m.K
    "power_factor": (0.01, 200.0),                   # uW/cm.K2
    "ZT": (0.0, 5.0),
}

CANONICAL_UNIT = {
    "seebeck_coefficient": "uV/K",
    "electrical_conductivity": "S/cm",
    "thermal_conductivity": "W/m.K",
    "power_factor": "uW/cm.K2",
    "ZT": "dimensionless",
}
