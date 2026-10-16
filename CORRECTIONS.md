# Corrections apportées à la première version

| # | Problème | Correction |
|---|---|---|
| 1 | Le README annonçait VaR 99 % de 1,82 %, 1,76 %, 1,78 % et CVaR de 2,41 %. Les sorties réelles des notebooks étaient 2,88 %, 2,70 %, 2,35 % et 4,55 %. | README réécrit à partir de `reports/results.json`, produit par le code. |
| 2 | La VaR GARCH utilisait le quantile normal alors que le modèle est skew-t : 34 violations, test de Kupiec rejeté. Le README annonçait pourtant « accepté ». | Quantile skew-t estimé : 18 violations, Kupiec p = 0,35. Les deux versions sont publiées. |
| 3 | Grille de Bâle (0-4 / 5-9 / 10+) appliquée à 1 434 jours alors qu'elle est définie sur 250 jours. | Comptage sur les 250 derniers jours d'une VaR glissante hors échantillon. |
| 4 | Le « Monte Carlo » tirait des rendements normaux de même moyenne et variance : il redonnait la VaR normale. | Ajout d'un Monte Carlo Student-t ; le tirage normal est conservé et étiqueté comme tel. |
| 5 | VaR Student-t : seule la présentation était trompeuse (le quantile ajusté était bon). | Documentée ; mise en garde sur ν ≈ 3. |
| 6 | Le scoring annonçait AUC 0,83 avec XGBoost et « Give Me Some Credit ». Les données étaient synthétiques (taux de défaut réel 1,8 %, non 6,7 %), XGBoost donnait 0,61. | Données déclarées synthétiques, intercept calé sur 6,6 %, AUC du générateur calculée (0,710) comme plafond. |
| 7 | `class_weight='balanced'` faussait les probabilités : 80 % des clients en classe « D ». | Pas de repondération : les PD sont calibrées. « D » (défaut) retiré de la grille de PD. |
| 8 | Rapport Excel : « Conforme » écrit en dur sur chaque ligne, AUC 0,832 en dur. | Rapport généré uniquement à partir des résultats calculés, sans statut en dur. |
| 9 | README : emojis, `votre-username`, chemin local `/Users/angek/...` dans les sorties, mention de candidature, fichiers `.DS_Store` et `.Rhistory`. | Supprimés, `.gitignore` ajouté. |
| 10 | Base SQLite binaire versionnée. | Retirée ; les CSV suffisent et le pipeline régénère les résultats. |
