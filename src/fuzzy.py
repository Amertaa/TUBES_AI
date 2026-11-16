"""Fuzzy Inference System tipe Mamdani (diimplementasi manual, tanpa library).

Input  : severity  (0-10)  = rata-rata tertimbang severity kelas dengan bobot probabilitas NN
         confidence (0-1)  = probabilitas tertinggi NN
Output : priority  (0-100) -> Accept / Monitor / Rework / Reject
Metode : fuzzifikasi -> AND = min -> implikasi = clip -> agregasi = max -> defuzzifikasi = centroid
"""
import numpy as np
from .config import CLASSES, SEVERITY


def trimf(x, a, b, c):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(c > b, (c - x) / (c - b + 1e-12), 1.0)
    return np.clip(np.minimum(left, right), 0, 1)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), 1.0)
    right = np.where(d > c, (d - x) / (d - c + 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0, 1)


# ---- Linguistic variables & membership functions ----
SEV_MF = {"Low": ("trap", (0, 0, 2, 4.5)), "Medium": ("tri", (3, 5, 7)), "High": ("trap", (5.5, 8, 10, 10))}
CONF_MF = {"Low": ("trap", (0, 0, 0.4, 0.6)), "Medium": ("tri", (0.45, 0.65, 0.85)), "High": ("trap", (0.7, 0.85, 1, 1))}
OUT_MF = {"Accept": ("trap", (0, 0, 15, 35)), "Monitor": ("tri", (20, 40, 60)),
          "Rework": ("tri", (45, 65, 85)), "Reject": ("trap", (65, 85, 100, 100))}
RANGES = {"severity": (0, 10), "confidence": (0, 1), "priority": (0, 100)}

# ---- Rule base: (severity, confidence) -> output ----
RULES = [
    ("Low", "High", "Accept"), ("Low", "Medium", "Monitor"), ("Low", "Low", "Monitor"),
    ("Medium", "High", "Rework"), ("Medium", "Medium", "Monitor"), ("Medium", "Low", "Monitor"),
    ("High", "High", "Reject"), ("High", "Medium", "Rework"), ("High", "Low", "Rework"),
]


def mf(spec, x):
    kind, p = spec
    return trimf(x, *p) if kind == "tri" else trapmf(x, *p)


def expected_severity(proba):
    sev = np.array([SEVERITY[c] for c in CLASSES], float)
    return float(np.dot(proba, sev))


def infer(severity, confidence, n=501):
    """Return dict: memberships input, rules fired, kurva agregasi, crisp priority, label."""
    u = np.linspace(*RANGES["priority"], n)
    ms = {k: float(mf(v, severity)) for k, v in SEV_MF.items()}
    mc = {k: float(mf(v, confidence)) for k, v in CONF_MF.items()}
    agg = np.zeros_like(u)
    fired = []
    for s, c, o in RULES:
        strength = min(ms[s], mc[c])                       # AND = min
        fired.append({"rule": f"IF severity={s} AND confidence={c} THEN priority={o}", "strength": strength, "out": o})
        agg = np.maximum(agg, np.minimum(strength, mf(OUT_MF[o], u)))   # clip + max
    crisp = float((u * agg).sum() / agg.sum()) if agg.sum() > 1e-9 else 50.0   # centroid
    out_deg = {k: float(mf(v, crisp)) for k, v in OUT_MF.items()}
    label = max(out_deg, key=out_deg.get)
    return {"severity": severity, "confidence": confidence, "sev_mu": ms, "conf_mu": mc,
            "fired": fired, "universe": u, "aggregated": agg, "priority": crisp, "label": label}


def decide(proba):
    """proba (7,) dari NN -> hasil fuzzy."""
    return infer(expected_severity(proba), float(np.max(proba)))
