# portfolio-risk-system

Mesure du risque de marché d'un portefeuille de cinq actifs (VaR, CVaR, GARCH, backtesting) et démonstration d'un pipeline de scoring de crédit sur données synthétiques. Projet personnel réalisé pendant le master Ingénierie économique et financière (Université de Rennes).

Tous les chiffres ci-dessous sont produits par `python run_pipeline.py` et sont stockés dans `reports/results.json`.

## Ce que fait le projet

**Risque de marché** (`src/market_risk.py`)
- Portefeuille : CAC 40 (25 %), BNP Paribas (20 %), TotalEnergies (20 %), or (20 %), EUR/USD (15 %), poids constants.
- Période : 3 janvier 2019 au 30 décembre 2024, 1 434 rendements journaliers (source des prix : Yahoo Finance).
- VaR historique, normale, Student-t, Monte Carlo (normal et Student-t), CVaR historique.
- GARCH(1,1) à innovations skew-t de Hansen, VaR conditionnelle.
- Backtesting : tests de Kupiec (fréquence) et de Christoffersen (indépendance), zones de Bâle.

**Scoring de crédit** (`src/credit_scoring.py`)
- Données **synthétiques** (15 000 emprunteurs, taux de défaut 6,6 %). Aucune donnée client réelle n'est utilisée.
- Régression logistique et gradient boosting, métriques AUC, Gini, KS, Brier, PSI.
- Grille de notation interne par bornes de PD et perte attendue EL = PD × LGD × EAD (LGD fixée à 45 %, EAD simulée).

## Résultats

### VaR journalière du portefeuille (en % de la valeur)

| Méthode | 95 % | 99 % |
|---|---|---|
| Historique | 1,450 | 2,877 |
| Normale | 1,648 | 2,346 |
| Student-t (ajustée par maximum de vraisemblance) | 1,349 | 2,697 |
| Monte Carlo, tirage normal | 1,654 | 2,347 |
| Monte Carlo, tirage Student-t | 1,354 | 2,690 |
| CVaR historique | 2,531 | 4,548 |

Les rendements du portefeuille ont une asymétrie de -1,08 et un excès de kurtosis de 16,1 : la loi normale sous-estime nettement la VaR à 99 % (2,35 % contre 2,88 % en historique). Le Monte Carlo à tirage normal redonne la VaR normale, ce qui est attendu et n'apporte rien ; seul le tirage Student-t reproduit les queues épaisses. Le degré de liberté Student-t estimé est d'environ 3, donc la variance théorique est à peine définie : ce résultat est à lire comme un ordre de grandeur.

### GARCH(1,1) skew-t

Persistance α + β = 0,980 (α = 0,092, β = 0,888, ν = 5,55, asymétrie λ = -0,18).

### Backtesting de la VaR 99 %

| VaR testée | Violations (attendues) | Kupiec p-value | Christoffersen p-value | Conclusion à 5 % |
|---|---|---|---|---|
| GARCH skew-t, quantile skew-t | 18 (14,3) | 0,350 | 0,499 | Non rejetée |
| GARCH skew-t, quantile normal | 34 (14,3) | < 0,001 | 0,248 | Rejetée (fréquence) |
| Historique constante (in-sample) | 15 (14,3) | 0,862 | non testé | Non rejetée, mais testée sur ses propres données |
| Historique glissante 250 j (hors échantillon) | 22 (11,8) | 0,008 | < 0,001 | Rejetée, violations groupées |

Trois enseignements :
1. Le quantile utilisé doit être cohérent avec la loi estimée. Avec le quantile normal appliqué à une volatilité estimée sous skew-t, la VaR 99 % est sous-estimée (34 violations au lieu de 14) ; avec le quantile skew-t, le modèle passe les tests.
2. Le test in-sample de la VaR historique constante est favorable par construction : le quantile est calculé sur les mêmes données. Le test honnête est la version glissante hors échantillon, qui est rejetée, surtout autour de mars 2020.
3. Sur les 250 derniers jours, la VaR historique glissante a 4 violations, soit la zone verte de Bâle. La grille de Bâle n'est définie que sur 250 jours et n'est pas appliquée à l'échantillon complet.

Limite : les paramètres GARCH sont estimés sur l'échantillon testé. Un backtest rigoureux ré-estimerait le modèle sur fenêtre glissante.

![VaR GARCH](reports/var_garch_violations.png)

### Scoring de crédit (données synthétiques)

| Modèle | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Régression logistique | 0,698 | 0,396 | 0,294 | 0,0578 |
| Gradient boosting | 0,684 | 0,368 | 0,304 | 0,0589 |

L'AUC de la vraie PD du générateur est de 0,710 : c'est le plafond atteignable par n'importe quel modèle sur ces données. La régression logistique est donc proche de l'optimum, et le gradient boosting n'apporte rien parce que le générateur est lui-même logistique. Ces chiffres ne disent rien de la performance sur un vrai portefeuille. PSI entre apprentissage et test : 0,0005.

Notation et perte attendue sur l'échantillon de test (4 500 emprunteurs) : EL totale 2,20 M€ pour 75,5 M€ d'exposition, soit 2,9 %. Les PD moyennes par classe suivent les taux de défaut observés (voir graphique).

![Scoring](reports/credit_scoring.png)

## Structure

```
src/market_risk.py       VaR, CVaR, GARCH, Kupiec, Christoffersen, zones de Bâle
src/credit_scoring.py    générateur synthétique, métriques, notation, EL
tests/test_core.py       9 tests
run_pipeline.py          exécute tout et écrit reports/
notebooks/               exploration commentée, exécutée
data/                    prix et rendements journaliers (Yahoo Finance)
reports/                 results.json, figures, risk_report.xlsx
```

## Exécution

```bash
pip install -r requirements.txt
python -m pytest tests
python run_pipeline.py
```

## Limites

- Cinq actifs, une seule période, pas de réestimation glissante du GARCH.
- Horizon d'un jour uniquement (pas de passage à 10 jours).
- Scoring sur données synthétiques : démonstration de méthode, pas de résultat empirique.
- LGD constante et EAD simulée.

Voir `CORRECTIONS.md` pour la liste des erreurs de la première version.
