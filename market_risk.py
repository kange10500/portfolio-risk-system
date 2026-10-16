"""Risque de marché : VaR, CVaR, GARCH(1,1) skew-t, backtesting.

Convention : les rendements sont des rendements logarithmiques journaliers ;
toutes les VaR / CVaR sont renvoyées comme des nombres positifs (pertes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

WEIGHTS = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
ASSETS = ["CAC40", "BNP_Paribas", "TotalEnergies", "Gold", "EURUSD"]


def portfolio_returns(returns: pd.DataFrame, weights=WEIGHTS) -> pd.Series:
    """Rendement du portefeuille à poids constants (rééquilibré chaque jour).

    Approximation usuelle : on pondère des rendements log, ce qui n'est exact
    qu'au premier ordre. Écart négligeable à l'échelle journalière.
    """
    w = np.asarray(weights, dtype=float)
    assert np.isclose(w.sum(), 1.0)
    return (returns[ASSETS] @ w).rename("Portfolio")


# ----------------------------------------------------------------- VaR / CVaR
def var_historical(r, confidence=0.99) -> float:
    return float(-np.quantile(np.asarray(r), 1 - confidence))


def var_normal(r, confidence=0.99) -> float:
    r = np.asarray(r)
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(1 - confidence)))


def var_student(r, confidence=0.99) -> float:
    """VaR Student-t ajustée par maximum de vraisemblance (loc, scale, nu).

    Le quantile est celui de la loi ajustée, sans re-normalisation par la
    variance : avec nu < 3 la variance théorique est infinie, on ne s'en sert pas.
    """
    nu, loc, scale = stats.t.fit(np.asarray(r))
    return float(-(loc + scale * stats.t.ppf(1 - confidence, nu)))


def var_monte_carlo_t(r, confidence=0.99, n=200_000, seed=42) -> float:
    """VaR Monte Carlo : simulation depuis une Student-t ajustée.

    Contrairement à une simulation gaussienne (qui redonne la VaR normale à
    l'erreur d'échantillonnage près), ce tirage reproduit les queues épaisses.
    """
    nu, loc, scale = stats.t.fit(np.asarray(r))
    sims = stats.t.rvs(nu, loc=loc, scale=scale, size=n,
                       random_state=np.random.default_rng(seed))
    return float(-np.quantile(sims, 1 - confidence))


def var_monte_carlo_normal(r, confidence=0.99, n=200_000, seed=42) -> float:
    r = np.asarray(r)
    sims = np.random.default_rng(seed).normal(r.mean(), r.std(ddof=1), n)
    return float(-np.quantile(sims, 1 - confidence))


def cvar_historical(r, confidence=0.99) -> float:
    r = np.asarray(r)
    v = var_historical(r, confidence)
    tail = r[r <= -v]
    return float(-tail.mean())


# ---------------------------------------------------------------------- GARCH
def fit_garch(r: pd.Series):
    """GARCH(1,1) avec innovations skew-t de Hansen, rendements en %."""
    return arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1,
                      dist="skewt").fit(disp="off")


def var_garch(res, confidence=0.99, innovations="skewt") -> pd.Series:
    """VaR conditionnelle (in-sample) : -(mu + sigma_t * q_alpha).

    `innovations` doit être cohérent avec le modèle estimé : le quantile q est
    celui de la loi skew-t standardisée estimée. Utiliser le quantile normal avec
    une volatilité estimée sous skew-t sous-estime la VaR à 99 %.
    """
    mu = res.params["mu"] / 100
    sig = res.conditional_volatility / 100
    if innovations == "skewt":
        q = res.model.distribution.ppf(1 - confidence,
                                       res.params[["eta", "lambda"]].values)
    else:
        q = stats.norm.ppf(1 - confidence)
    return -(mu + sig * q)


# ---------------------------------------------------------------- Backtesting
def violations(r, var) -> np.ndarray:
    return (np.asarray(r) < -np.asarray(var)).astype(int)


def kupiec_pof(viol, confidence=0.99) -> dict:
    viol = np.asarray(viol)
    T, V, p = len(viol), int(viol.sum()), 1 - confidence
    x = V / T
    ll0 = (T - V) * np.log(1 - p) + V * np.log(p)
    ll1 = (T - V) * np.log(1 - x) + V * np.log(x) if 0 < V < T else 0.0
    lr = -2 * (ll0 - ll1)
    pv = 1 - stats.chi2.cdf(lr, 1)
    return dict(T=T, violations=V, expected=round(T * p, 1), rate=round(x, 4),
                LR=round(float(lr), 3), p_value=round(float(pv), 4), reject_5pct=bool(pv < 0.05))


def christoffersen_ind(viol) -> dict:
    v = np.asarray(viol)
    a, b = v[:-1], v[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())

    def ll(k, n, p):  # k succès sur n, probabilité p ; 0*log(0) = 0
        out = 0.0
        if k: out += k * np.log(p)
        if n - k: out += (n - k) * np.log(1 - p)
        return out

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr = -2 * (ll(n01 + n11, n00 + n01 + n10 + n11, pi)
               - ll(n01, n00 + n01, pi01) - ll(n11, n10 + n11, pi11))
    pv = 1 - stats.chi2.cdf(lr, 1)
    return dict(n00=n00, n01=n01, n10=n10, n11=n11, LR=round(float(lr), 3),
                p_value=round(float(pv), 4), reject_5pct=bool(pv < 0.05))


def basel_zone(n_violations_250d: int) -> str:
    """Zone de Bâle (VaR 99 %, fenêtre de 250 jours) : vert 0-4, jaune 5-9, rouge >= 10.

    Cette grille n'est définie que sur 250 jours ; ne pas l'appliquer à un
    échantillon plus long (le comptage doit se faire sur la fenêtre glissante).
    """
    return "verte" if n_violations_250d <= 4 else ("jaune" if n_violations_250d <= 9 else "rouge")


def rolling_hist_var_violations(r, confidence=0.99, window=250) -> np.ndarray:
    """Violations hors échantillon : VaR historique estimée sur les `window` jours précédents."""
    r = np.asarray(r)
    out = [r[i] < -var_historical(r[i - window:i], confidence) for i in range(window, len(r))]
    return np.array(out, dtype=int)
