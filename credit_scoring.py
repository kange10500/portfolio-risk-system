"""Scoring de crédit sur données SYNTHÉTIQUES (démonstration méthodologique).

Aucune donnée client réelle n'est utilisée. Le générateur est un modèle
logistique connu : les métriques mesurent donc la capacité à retrouver ce
modèle, pas la performance sur un portefeuille réel. L'AUC maximale atteignable
est bornée par le bruit du générateur (voir `oracle_auc`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

FEATURES = ["util", "age", "late_30", "late_90", "debt_ratio", "log_income",
            "dependents", "n_loans", "n_real_estate"]


def _logit_part(d: pd.DataFrame) -> np.ndarray:
    return (1.8 * d.util + 0.8 * d.late_30 + 1.5 * d.late_90 + 0.5 * d.debt_ratio
            - 0.02 * (d.age - 40) - 0.6 * (d.log_income - 8.5) + 0.1 * d.dependents)


def make_synthetic(n=15_000, target_rate=0.067, seed=42) -> pd.DataFrame:
    """Génère la population ; l'intercept est calé pour atteindre `target_rate`."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": rng.integers(22, 75, n),
        "log_income": rng.normal(8.5, 0.6, n),
        "debt_ratio": np.clip(rng.exponential(0.4, n), 0, 3),
        "util": np.clip(rng.beta(1.5, 4, n), 0, 1),
        "late_30": rng.poisson(0.2, n),
        "late_90": rng.poisson(0.05, n),
        "dependents": rng.poisson(0.7, n),
        "n_loans": rng.poisson(4, n),
        "n_real_estate": rng.poisson(0.8, n),
    })
    z = _logit_part(d).to_numpy()
    f = lambda b0: (1 / (1 + np.exp(-(b0 + z)))).mean() - target_rate
    b0 = brentq(f, -15, 5)
    d["pd_true"] = 1 / (1 + np.exp(-(b0 + z)))
    d["default"] = rng.binomial(1, d.pd_true)
    d.attrs["intercept"] = b0
    return d


def oracle_auc(d: pd.DataFrame) -> float:
    """AUC de la vraie PD : plafond théorique pour n'importe quel modèle."""
    return float(roc_auc_score(d["default"], d["pd_true"]))


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float((tpr - fpr).max())


def psi(expected, actual, buckets=10) -> float:
    bp = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    bp[0], bp[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=bp)[0] / len(expected)
    a = np.histogram(actual, bins=bp)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return dict(AUC=round(auc, 4), Gini=round(2 * auc - 1, 4),
                KS=round(ks_statistic(y, p), 4), Brier=round(brier_score_loss(y, p), 4),
                mean_pd=round(float(np.mean(p)), 4), observed_rate=round(float(np.mean(y)), 4))


# Grille interne : bornes SUPÉRIEURES de PD. « D » n'est pas une classe de PD :
# c'est l'état de défaut, donc absent de la grille de notation des clients sains.
GRID = [("AAA", 0.005), ("AA", 0.01), ("A", 0.02), ("BBB", 0.04),
        ("BB", 0.08), ("B", 0.15), ("CCC", 1.0)]


def rating(pd_value: float) -> str:
    for name, ub in GRID:
        if pd_value <= ub:
            return name
    return "CCC"


def expected_loss(pd_, lgd, ead) -> np.ndarray:
    return np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)
