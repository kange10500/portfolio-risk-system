"""Exécute tout le pipeline et écrit reports/results.json, 3 figures et risk_report.xlsx.

    python run_pipeline.py
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src import market_risk as mr, credit_scoring as cs

ROOT = pathlib.Path(__file__).parent
REP = ROOT / "reports"; REP.mkdir(exist_ok=True)
out = {}

# ------------------------------------------------------------ risque de marché
R = pd.read_csv(ROOT / "data/log_returns.csv", index_col=0, parse_dates=True)
P = mr.portfolio_returns(R)
out["market"] = {"period": [str(P.index[0].date()), str(P.index[-1].date())], "n_obs": len(P),
                 "ann_vol": round(float(P.std() * np.sqrt(252)), 4),
                 "skew": round(float(P.skew()), 3), "excess_kurtosis": round(float(P.kurt()), 2)}
var = {}
for c in (0.95, 0.99):
    var[str(c)] = {"historique": mr.var_historical(P, c), "normale": mr.var_normal(P, c),
                   "student_t": mr.var_student(P, c), "monte_carlo_normal": mr.var_monte_carlo_normal(P, c),
                   "monte_carlo_student_t": mr.var_monte_carlo_t(P, c), "cvar_historique": mr.cvar_historical(P, c)}
out["var_1d"] = {c: {k: round(v * 100, 3) for k, v in d.items()} for c, d in var.items()}

res = mr.fit_garch(P)
out["garch"] = {"params": {k: round(float(v), 4) for k, v in res.params.items()},
                "persistence": round(float(res.params["alpha[1]"] + res.params["beta[1]"]), 4)}
bt = {}
for c in (0.95, 0.99):
    for inn in ("skewt", "normal"):
        v = mr.var_garch(res, c, inn); viol = mr.violations(P, v)
        bt[f"garch_{inn}_{int(c*100)}"] = {"kupiec": mr.kupiec_pof(viol, c),
                                          "christoffersen": mr.christoffersen_ind(viol)}
    h = np.full(len(P), mr.var_historical(P, c)); viol = mr.violations(P, h)
    bt[f"historique_constante_insample_{int(c*100)}"] = {"kupiec": mr.kupiec_pof(viol, c)}
roll = mr.rolling_hist_var_violations(P, 0.99, 250)
bt["historique_glissante_250j_hors_echantillon_99"] = {
    "kupiec": mr.kupiec_pof(roll, 0.99), "christoffersen": mr.christoffersen_ind(roll),
    "violations_250_derniers_jours": int(roll[-250:].sum()),
    "zone_bale_250_derniers_jours": mr.basel_zone(int(roll[-250:].sum()))}
out["backtesting"] = bt

v99 = mr.var_garch(res, 0.99, "skewt"); viol99 = mr.violations(P, v99).astype(bool)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(P.index, P.values * 100, lw=.6, color="steelblue", label="Rendement portefeuille")
ax[0].plot(P.index, -v99 * 100, "r", lw=1, label="-VaR 99 % GARCH skew-t")
ax[0].scatter(P.index[viol99], P.values[viol99] * 100, s=18, color="k", zorder=5, label=f"Violations ({viol99.sum()})")
ax[0].set_ylabel("%"); ax[0].legend(fontsize=8)
ax[1].plot(res.conditional_volatility.index, res.conditional_volatility * np.sqrt(252) / 100, color="navy")
ax[1].set_ylabel("Vol. conditionnelle annualisée"); ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout(); plt.savefig(REP / "var_garch_violations.png", dpi=130); plt.close()

# ------------------------------------------------------------------- crédit
d = cs.make_synthetic()
out["credit"] = {"data": "synthétique", "n": len(d), "default_rate": round(float(d["default"].mean()), 4),
                 "oracle_auc": round(cs.oracle_auc(d), 4)}
X, y = d[cs.FEATURES], d["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
models = {"Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
          "Gradient boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
                                                          subsample=.8, random_state=42)}
out["credit"]["models"] = {}
probs = {}
for n, m in models.items():
    m.fit(Xtr, ytr); probs[n] = m.predict_proba(Xte)[:, 1]
    out["credit"]["models"][n] = cs.metrics(yte, probs[n])
best = max(out["credit"]["models"], key=lambda k: out["credit"]["models"][k]["AUC"])
out["credit"]["best"] = best
out["credit"]["psi_train_vs_test"] = round(cs.psi(models[best].predict_proba(Xtr)[:, 1], probs[best]), 4)

rng = np.random.default_rng(7)
te = Xte.copy(); te["pd"] = probs[best]; te["rating"] = te.pd.map(cs.rating)
te["lgd"] = 0.45; te["ead"] = rng.lognormal(9.5, 0.7, len(te)).round(0)
te["el"] = cs.expected_loss(te.pd, te.lgd, te.ead)
g = te.groupby("rating").agg(n=("pd", "size"), pd_moy=("pd", "mean"), ead=("ead", "sum"), el=("el", "sum"))
g["taux_defaut_observe"] = yte.groupby(te["rating"]).mean()
order = [r for r, _ in cs.GRID if r in g.index]
g = g.loc[order]
out["credit"]["by_rating"] = json.loads(g.round(4).to_json(orient="index"))
out["credit"]["el_total"] = round(float(te.el.sum()), 0); out["credit"]["ead_total"] = round(float(te.ead.sum()), 0)
out["credit"]["el_rate"] = round(float(te.el.sum() / te.ead.sum()), 4)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
from sklearn.metrics import roc_curve
for n, p in probs.items():
    f, t, _ = roc_curve(yte, p); ax[0].plot(f, t, label=f"{n} (AUC {out['credit']['models'][n]['AUC']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
ax[0].set_title("ROC (données synthétiques)")
ax[1].bar(g.index, g.pd_moy * 100, color="steelblue", label="PD prédite")
ax[1].plot(g.index, g.taux_defaut_observe * 100, "o-", color="darkorange", label="Taux observé")
ax[1].set_ylabel("%"); ax[1].set_title("PD prédite vs taux observé par classe"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(REP / "credit_scoring.png", dpi=130); plt.close()

# ------------------------------------------------------------- rapport Excel
with pd.ExcelWriter(REP / "risk_report.xlsx") as xw:
    pd.DataFrame(out["var_1d"]).rename_axis("méthode").to_excel(xw, sheet_name="VaR 1 jour (%)")
    rows = []
    for k, v in bt.items():
        kp = v["kupiec"]; rows.append({"test": k, "T": kp["T"], "violations": kp["violations"],
                                       "attendu": kp["expected"], "LR Kupiec": kp["LR"], "p-value": kp["p_value"],
                                       "rejet 5 %": kp["reject_5pct"]})
    pd.DataFrame(rows).to_excel(xw, sheet_name="Backtesting", index=False)
    pd.DataFrame(out["credit"]["models"]).T.to_excel(xw, sheet_name="Scoring (synthétique)")
    g.reset_index().to_excel(xw, sheet_name="Notation et EL", index=False)

(REP / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
