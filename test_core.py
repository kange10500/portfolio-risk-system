import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import market_risk as mr, credit_scoring as cs

R = pd.read_csv(pathlib.Path(__file__).resolve().parents[1] / "data" / "log_returns.csv",
                index_col=0, parse_dates=True)
P = mr.portfolio_returns(R)


def test_var_ordering_and_values():
    assert abs(mr.var_historical(P, .99) - 0.028769) < 1e-5
    assert mr.cvar_historical(P, .99) > mr.var_historical(P, .99)
    assert mr.var_historical(P, .99) > mr.var_historical(P, .95)


def test_mc_normal_matches_parametric_normal():
    assert abs(mr.var_monte_carlo_normal(P) - mr.var_normal(P)) < 5e-4


def test_kupiec_expected_rate_is_not_rejected():
    v = np.zeros(1000, int); v[:10] = 1
    assert not mr.kupiec_pof(v, .99)["reject_5pct"]
    v[:40] = 1
    assert mr.kupiec_pof(v, .99)["reject_5pct"]


def test_christoffersen_detects_clustering():
    v = np.zeros(1000, int); v[100:110] = 1
    assert mr.christoffersen_ind(v)["reject_5pct"]


def test_basel_zones():
    assert [mr.basel_zone(k) for k in (0, 4, 5, 9, 10)] == ["verte", "verte", "jaune", "jaune", "rouge"]


def test_garch_skewt_quantile_is_more_conservative_in_tail_than_normal_quantile_at_99():
    res = mr.fit_garch(P)
    a = mr.var_garch(res, .99, "skewt"); b = mr.var_garch(res, .99, "normal")
    assert (mr.violations(P, a).sum()) < (mr.violations(P, b).sum())


def test_synthetic_default_rate_matches_target():
    d = cs.make_synthetic()
    assert abs(d["default"].mean() - 0.067) < 0.006


def test_rating_monotone():
    assert cs.rating(0.001) == "AAA" and cs.rating(0.05) == "BB" and cs.rating(0.5) == "CCC"


def test_psi_zero_on_same_sample():
    x = np.random.default_rng(0).normal(size=5000)
    assert cs.psi(x, x) < 1e-6
