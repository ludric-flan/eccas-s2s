# Chaîne OSF CAPC-AC dans eccas-s2s — guide d'utilisation

Workflow de référence : `WORKFLOW_OSF_CAPC-AC_v0.3` (page web et `.md` dans `Previsions_S2S/`).
La chaîne tourne **chaque mois** avec l'initialisation du mois (1 cycle = 1 fichier `config/cycle_YYYYMM.yaml`).

## Environnement

```bash
conda env create -f environment.yml    # inclut R, r-verification et r-boot
conda activate eccas-s2s
pip install -e .                       # une fois
pytest -q                              # tous les tests doivent passer
```

**Scores de vérification en R.** Les scores de zone (RPS/RPSS, Brier et sa décomposition, ROC et sa p-value, CRPS, Heidke/Peirce/Gerrity, diagrammes de fiabilité) sont calculés avec le paquet R `verification`, comme dans la chaîne de référence du CAPC-AC. L'environnement conda l'installe ; si R est déjà présent sur la machine :

```bash
Rscript -e 'install.packages(c("verification","boot"), repos="https://cloud.r-project.org")'
python -c "from eccas_s2s.validate.r_bridge import check_packages; print(check_packages())"
```

Les cartes de scores par point de grille sont calculées en Python (`eccas_s2s.validate.scores`) ; un test compare les deux implémentations.

## Démarrer un nouveau cycle mensuel

1. Copier `config/cycle_YYYYMM.yaml` du mois précédent en `config/cycle_<nouveau>.yaml`.
2. Modifier `cycle.init_date` et vérifier les numéros de système C3S (`systems.c3s.models`).
3. Lancer les étapes dans l'ordre (les scripts s'ajoutent phase après phase) :

| Étape | Commande | Phase |
|---|---|---|
| E2 · téléchargement C3S | `python scripts/run_download_c3s.py --config config/cycle_202609.yaml --variable precip` | P0 |
| | *chaque modèle reçoit les flux déclarés par sa clé `precip_from` : journalier, mensuel, ou les deux. `--stream daily\|monthly` force un flux.* | |
| E2 · contrôle qualité C3S | `python scripts/run_qc_c3s.py --config config/cycle_202609.yaml --variable precip` | P0 |
| E1 · référence CHIRPS (archive + normales, seulement si CHIRPS a changé) | `python scripts/run_obs_chirps.py --config config/cycle_202609.yaml` | P1 |
| E2 · cumuls C3S par période | `python scripts/run_c3s_totals.py --config config/cycle_202609.yaml` | P1 |
| | *`--streams monthly` ne recalcule qu'un flux (sans réécrire l'autre fichier) ; `--periods` choisit les périodes (voir ci-dessous).* | |
| E2 · températures C3S par période | `python scripts/run_c3s_temperature.py --config config/cycle_202609.yaml` | P1 |
| E1 · CHIRPS 1981–1990 (une fois) | `python scripts/run_download_chirps.py --config ... --years 1981 1990` | P1 |
| E1 · température observée ERA5 (horaire → journalier) | `python scripts/run_download_era5_hourly.py --config ... --years 1981 2026 --workers 4` | P1 |
| E1 · référence ERA5 (archive + normales) | `python scripts/run_obs_era5.py --config config/cycle_202609.yaml` | P1 |
| E2 · NMME (désactivé depuis le 23/09/2026 — voir plus bas) | `run_download_nmme.py`, `run_nmme_totals.py` | — |
| E4 · skill brut, produit par produit (netCDF) | `python scripts/run_skill_raw.py --config config/cycle_202609.yaml --systems c3s --variables precip` | P2 |
| | *`--metrics deciding` (défaut) \| `reported` \| `all` \| liste explicite* | |
| E4 · cartes de skill (une carte par produit, métrique et période) | `python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml` | P2 |
| | *`--metrics all` trace aussi les compagnons ; `--products` restreint le catalogue* | |
| E4 · diagrammes de fiabilité et ROC (R) | `python scripts/run_skill_diagrams.py --config config/cycle_202609.yaml --scales month season` | P2 |
| E4 · reconstruire le tableau de synthèse depuis les netCDF | `python scripts/run_skill_raw.py --config config/cycle_202609.yaml --from-maps` | P2 |
| … | (ajoutés au fil des phases) | |

Chaque étape existe aussi en notebook opérationnel (voir plus bas). Scripts et notebooks appellent la même fonction `run(...)` de `eccas_s2s/operations/`.

Tester une requête sans télécharger : ajouter `--dry-run`.

## Où sont les fichiers

| Contenu | Emplacement (défini dans le YAML) |
|---|---|
| Données brutes immuables | `DATA_OSF/raw/<système>/<YYYYMM>/` |
| Référence CHIRPS dérivée (partagée par tous les cycles) | `DATA_OSF/derived/obs/chirps/` |
| Référence température ERA5 dérivée | `DATA_OSF/derived/obs/era5/` |
| Valeurs NMME par période | `DATA_OSF/derived/nmme/<YYYYMM>/` |
| Cumuls C3S par période | `DATA_OSF/derived/c3s/<YYYYMM>/c3s_<centre>_<variable>_<kind>_periods.nc` (flux journalier) et `…_periods_monthly.nc` (flux mensuel) |
| Journaux et manifestes d'exécution | `OUTPUTS_OSF/runs/<run_id>/{run.log, manifest.json}` |
| Scores de skill (netCDF, un fichier par métrique) | `OUTPUTS_OSF/skill/<YYYYMM>/raw/netcdf/<système>_<modèle>/<échelle>/<variable>/<produit>/<métrique>.nc` |
| Cartes de skill (une par période) | `…/raw/figures/<système>_<modèle>/<échelle>/<variable>/<produit>/<métrique>/<période>.png` |
| Diagrammes fiabilité et ROC (par famille) | `…/raw/diagrams/<système>_<modèle>/<échelle>/<variable>/<famille>/{reliability,roc}/<période>.png` |
| Synthèse et registres | `OUTPUTS_OSF/skill/<YYYYMM>/raw/skill_raw_summary.csv`, `OUTPUTS_OSF/registry/{products,models}_eligibility.csv` |
| Prévisions émises (archive) | `ARCHIVE_OSF/<YYYY>/<MM>/<run_id>/` |

## Modules OSF (phase P0)

| Module | Rôle |
|---|---|
| `eccas_s2s.settings` | lecture et validation de la configuration du cycle |
| `eccas_s2s.provenance` | `RunContext` (journal + manifeste), `archive_run` |
| `eccas_s2s.core.periods` | décades / mois / saisons relatives à l'initialisation, années bissextiles |
| `eccas_s2s.core.daily` | pluie journalière depuis le cumul C3S (**jour 0 = jour d'initialisation**), agrégation par période |
| `eccas_s2s.io.c3s_read` | lecture GRIB → structure `(year, number, lead_day, latitude, longitude)` |
| `eccas_s2s.agro.calendars` | paramètres Liebmann (v10) et faisabilité par initialisation |
| `eccas_s2s.io.c3s_qc` | contrôle qualité des fichiers bruts : membres, années, horizon reçu, échéances manquantes |
| `eccas_s2s.obs.chirps` | lecture CHIRPS, contrôle qualité et cumuls décadaires/mensuels en une passe mois par mois |
| `eccas_s2s.obs.climatology` | cumuls observés des périodes d'un cycle, normales 1991–2020 par maille et par période calendaire |
| `eccas_s2s.obs.regrid` | moyenne par blocs exacte 0,05° → 1° (grilles emboîtées) ; remaillage conservatif 0,25° → 1° (grilles non emboîtées) |
| `eccas_s2s.obs.era5` | lecture ERA5 journalier, contrôle qualité, moyennes décadaires et mensuelles |
| `eccas_s2s.io.nmme_cpc` | fichiers NMME du serveur NOAA/CPC (moyenne d'ensemble) |
| `eccas_s2s.core.monthly` | agrégation mois et saisons depuis des données mensuelles |
| `eccas_s2s.products.masks` | masque de saison sèche (méthodes `relative` et `absolute`) |
| `eccas_s2s.viz.maps` | panneaux de cartes CEEAC |
| `eccas_s2s.validate.cv` | validation croisée LOYO (moyenne et quantiles laissant l'année dehors) |
| `eccas_s2s.validate.pairs` | couples prévision/observation, catégories observées et probabilités des terciles |
| `eccas_s2s.validate.scores` | scores par maille (déterministes, RPS/RPSS, Brier, aire ROC) |
| `eccas_s2s.core.geo` | masque CEEAC (point de grille dans le shapefile), fraction de surface au-dessus d'un seuil |
| `eccas_s2s.validate.pooled` | couples de tous les points de grille d'une zone, pour les diagrammes |
| `eccas_s2s.validate.r_bridge` + `zone_diagrams.R` | diagrammes de fiabilité et ROC avec le paquet R `verification` (`zone_scores.R` ne sert plus qu'au contrôle croisé Python ↔ R des tests) |
| `eccas_s2s.viz.ceeac_maps` | cartes à la charte CAPC-AC (shapefile CEEAC, logo, barre de couleur commune) |
| `eccas_s2s.operations.*` | étapes opérationnelles (`run(...)` + `main(argv)`) : `download_c3s`, `qc_c3s`, `obs_chirps`, `c3s_totals`, `c3s_temperature`, `skill_raw`, `skill_diagrams`, `plot_skill_raw`, `calibrate_hindcast`, `housekeeping` |

Les modules historiques (`eccas_s2s.config`, `core.processing`, `pipeline`) sont conservés tels quels.

## Vérification par produit (phase P2)

La chaîne ne livre pas « une prévision » mais un **catalogue de produits**, et chacun est une
question opérationnelle différente : un cumul se juge sur son erreur, une carte de terciles sur
son score de probabilité ordonnée, une carte « plus de 200 mm » sur son score de Brier. La
vérification suit donc le catalogue (`eccas_s2s/products/catalogue.py`), 13 produits par échelle
pour la pluie.

| Nature | Produits | Décide | Compagnons calculés avec |
|---|---|---|---|
| valeur | `cumul`, `anomalie`, `spi` | MSESS, ACC (anomalie) | corrélation (discrimination) |
| 3 catégories | `terciles`, `spi_classes` | RPSS | RPSS débiaisé, GROC |
| événement | `categorie_BN/AN`, `quintile_bas/haut`, `depassement_median`, `depassement_<mm>` | BSS | BSS débiaisé, ROC − 0,5 |

- **Deux conventions de lecture.** Un produit défini par une **position dans la distribution**
  (terciles, P20, médiane, P80, classes SPI, SPI) est lu dans la climatologie **du modèle
  lui-même** en LOYO, comme `run_forecast_v2` ; un produit défini en **valeur absolue** (cumul,
  anomalie, seuil en mm) est lu contre le seuil **observé**. Le premier survit au biais, le
  second le subit — et c'est ce que la phase P2 doit établir. Côté observation, tous les seuils
  viennent de l'observation en LOYO. Le basculement est porté par l'attribut
  `uses_own_climatology` de la distribution : la loi calibrée de P3, qui vit déjà sur l'échelle
  de l'observation, utilise partout les seuils observés.
- **Scores débiaisés.** Une probabilité comptée sur *m* membres porte une erreur
  d'échantillonnage qui gonfle le RPS et le score de Brier ; nos hindcasts vont de 20 à 31
  membres, donc le score brut classe en partie par taille d'ensemble. `rpss_fair` et `bss_fair`
  (Ferro 2007) retirent cette pénalité et servent à comparer les modèles entre eux ; les
  versions classiques restent publiées, car ce sont elles que l'utilisateur subit.
- **Valeur sans skill par métrique.** Un score de skill vaut 0 sans skill, mais une **aire ROC
  et un GROC valent 0,5** : la fraction de surface utile est comptée au-dessus de
  `NO_SKILL_VALUE`, sans quoi le critère compterait tout le domaine.
- **Admission d'un modèle (§3.3).** Le registre répond à « ce modèle est-il exploitable ? », pas
  à « est-il bon brut ». Le critère est la **discrimination** sur ≥ 5 % du masque : un modèle qui
  ne discrimine nulle part est écarté (aucune calibration ne crée de l'information), un modèle
  qui discrimine mais reste biaisé est conservé — c'est le travail de P3. D'où trois états par
  produit : *exploitable — utilisable brut*, *exploitable — à calibrer*, *non exploitable*.
- **Diagrammes par famille.** Une fiabilité et un ROC par famille, toutes les classes sur le même
  repère : `terciles` (BN/NN/AN), `classes_spi`, `seuils_percentiles` (P20, médiane, P80),
  `seuils_cumul` (les seuils en mm de l'échelle). Pour les seuils en millimètres **seulement**,
  les mailles où l'événement ne se produit jamais ou toujours sont écartées du poolage : 300 mm
  en saison est banal à l'équateur et impossible au Sahel, et les mélanger ferait parler la
  courbe du gradient climatique. Cartes et diagrammes lisent les **mêmes** probabilités
  (`product_probabilities`).

## Calibration et comparaison (phase P3)

La calibration reprend **exactement** la vérification de P2 : mêmes produits, mêmes métriques
décisives, même contexte d'observation, mêmes périodes, même masque. Seule la distribution
change — l'ensemble brut en P2, la loi calibrée en P3 — donc l'écart mesuré ne peut venir que de
la calibration.

- **Une méthode n'est essayée que sur les produits auxquels elle peut répondre**
  (`eccas_s2s/calibrate/products.py`, table déclarative). La régression logistique par catégorie
  ne rend que les trois catégories des terciles : elle ne sert ni un cumul ni un seuil en mm.
  L'ELR décrit la distribution par ses seuils : elle sert les produits probabilistes, pas une
  valeur. Les corrections de biais, EQM, QDM et NGR/EMOS servent tout.
- **La référence du gain est la prévision interpolée sur la grille d'observation** (méthode
  `raw`), et non les cartes de P2 : celles-ci sont calculées à 1°, et les comparer à des scores
  calculés à 0,25° mélangerait l'effet de la calibration et celui du changement de grille.
- **Le gain est calculé maille par maille** (`eccas_s2s/validate/comparison.py`) et conservé en
  netCDF (`gain_<métrique>.nc`). Le résumé porte quatre nombres, pondérés par le cosinus de la
  latitude : `median_gain`, `fraction_improved`, `fraction_repaired` (le brut était sous sa
  valeur sans skill, le calibré est au-dessus) et `fraction_broken` (l'inverse). Les deux
  dernières existent parce qu'une médiane ment : une méthode qui répare l'équateur et casse le
  Sahel a le même gain médian qu'une méthode qui ne fait rien.
- **Méthode recommandée** : elle doit améliorer la maille typique **et** au moins la moitié du
  masque ; la retenue est celle du plus fort gain médian, départagée par la surface réparée. Si
  aucune ne qualifie, la recommandation est `raw` et le registre écrit pourquoi. Registre :
  `registry/calibration_methods.csv`.
- **Sorties**, en miroir de l'arbre brut, avec un niveau `<méthode>` de plus :
  `skill/<cycle>/calibrated/netcdf/<système>_<modèle>/<échelle>/<variable>/<produit>/<méthode>/<métrique>.nc`.
  Les cartes (`--kind calibrated`) et les diagrammes (`--kind calibrated`) ne tracent par défaut
  que la **méthode recommandée**, avec la mention **Calibrated** dans le titre. Pour les
  diagrammes, la famille prend la méthode recommandée pour la majorité de ses produits : les
  trois courbes d'un même repère doivent venir de la même distribution.

| Étape | Commande |
|---|---|
| E5 · calibration + scores + comparaison | `python scripts/run_calibrate_hindcast.py --config config/cycle_202609.yaml --variables precip --scales month season` |
| E5 · cartes calibrées | `python scripts/run_plot_skill_raw.py --config ... --kind calibrated` |
| E5 · diagrammes calibrés | `python scripts/run_skill_diagrams.py --config ... --kind calibrated` |
| E5 · cartes de gain (calibré − brut) | `python scripts/run_plot_skill_raw.py --config ... --kind gain --products terciles` |

## Format des netCDF (CF-1.8)

Tous les fichiers de la chaîne passent par `eccas_s2s/io/netcdf.py` (`save`, `open_cf`), pour
qu'ils s'ouvrent ailleurs que dans xarray — CDO, ncview, GrADS, QGIS.

| Règle | Pourquoi |
|---|---|
| Une période devient un **axe `time` daté** (1ᵉʳ jour) avec `time_bnds` (début, fin exclue) | une dimension de type chaîne fait échouer CDO (« Unsupported file structure ») et ncview (« unknown data type (12) ») |
| Tout autre axe textuel (catégories, clés calendaires) devient un **entier** avec `flag_values` / `flag_meanings` | convention CF §3.5, lisible par tous |
| **Aucune variable texte** : libellés, clés et noms de catégories en **attributs globaux**, listes séparées par `\|`, dans l'ordre de l'axe | un tableau de caractères porté par l'axe du temps est lu comme « coordonnée caractère » par CDO, qui échoue dessus |
| `units`, `standard_name`, `axis` sur latitude et longitude ; pas de `_FillValue` sur une coordonnée | sans cela CDO annonce une grille « generic » au lieu de `lonlat` |
| Écriture atomique (`.tmp.nc` puis renommage), flottants en simple précision compressés | un lecteur ne voit jamais un fichier à moitié écrit |

`open_cf()` restaure la vue interne : `sel(period="season_m1")` et `sel(category="AN")`
fonctionnent comme avant, la conversion étant invisible pour le reste du code.

**Les sources restent en GRIB, et c'est délibéré.** Une prévision d'ensemble porte cinq axes
(année d'initialisation, membre, échéance, latitude, longitude) : le GRIB les porte
nativement — `number`, `step`, `time` — et `cfgrib` les lit directement, là où la conversion en
netCDF les rend pénibles et où le modèle de données de CDO, qui s'arrête à quatre axes, ne
suffit plus. La chaîne télécharge donc les hindcasts et prévisions C3S en GRIB et les ouvre avec
`cfgrib` (`eccas_s2s/io/c3s_read.py`), comme le fait `run_forecast_v2`.

Les netCDF de la chaîne sont des **produits dérivés** : cumuls par période, scores, gains,
normales. Ceux qui gardent les cinq axes (les cumuls de hindcast, intermédiaires internes)
s'ouvrent dans xarray et ncview ; pour un usage CDO, on part de la source GRIB ou d'une tranche
(une année, un membre). Tous les autres — scores, cartes de gain, archives d'observation — sont
à trois ou quatre axes et s'ouvrent partout.

Les fichiers écrits avant cette convention se réécrivent sans recalcul :

```bash
python scripts/run_housekeeping.py --config config/cycle_202609.yaml --to-cf
```

## Conventions à retenir

- **Dates des cumuls C3S :** le champ de l'échéance +24 h (daté du 2 à 00 UTC) contient la pluie du **1er**. L'ancienne chaîne décalait toutes les périodes d'un jour ; la chaîne OSF les aligne sur le calendrier des observations. Voir le notebook 00, section 4.
- **Périodes complètes uniquement :** une période qui dépasse l'horizon du modèle n'est pas produite.
- **Indicateurs de saison :** produits seulement si toute leur fenêtre de référence est dans la prévision ; une fenêtre déjà commencée n'est pas complétée avec des observations.
- **Production :** lancer la chaîne depuis un code commité (`dirty: false` dans le manifeste).

- **Deux flux de pluie C3S (décision du 23/09/2026) :** l'archive journalière ne contient, pour les systèmes à démarrages décalés, que les membres du 1er du mois ; l'archive mensuelle les contient tous. D'où, pour l'init. de septembre 2026 : UKMO 7 membres en hindcast contre **28**, BoM 3 contre **27**, NCEP aucun hindcast journalier contre **24**. **UKMO, BoM et NCEP** prennent donc leurs **mois et saisons** du flux mensuel, et leurs **décades** du flux journalier (UKMO et BoM seulement : NCEP n'a pas de décades). Le routage est déclaré modèle par modèle dans `systems.c3s.models.<centre>.precip_from`, jamais codé en dur ; `hindcast_streams()` (`operations/skill_raw.py`) l'applique à la vérification et à la calibration. Les deux flux sont écrits dans **deux fichiers distincts** parce qu'ils ne portent pas le même ensemble.
- **Conversion des cumuls mensuels :** l'archive mensuelle fournit `tprate`, un **débit moyen** en m/s sur le mois cible ; la conversion en millimètres utilise la **vraie longueur du mois** (`mm = tprate × 1000 × 86400 × n_jours`). Multiplier par 30 jours, comme le font certaines chaînes, sous-compterait un mois de 31 jours de 3 % et surcompterait février de 7 à 10 %. Vérifié sur ECMWF, qui porte les **mêmes 25 membres** des deux côtés : écart absolu moyen **0,005 mm** (0,014 % d'un mois à 60 mm), maximum 0,31 mm. Tolérance inscrite en test permanent : maximum < 0,5 mm, moyenne < 0,05 mm.
- **Normales observées :** 1991–2020, par maille et par période calendaire (`dekad_MM_D`, `month_MM`, `season_MM`), percentiles estimés par la position de Weibull ; communes à tous les modèles (D12).
- **Grilles :** CHIRPS 0,05° et C3S 1° sont emboîtées (20 × 20) : passage à 1° par moyenne par blocs exacte. Coordonnées CHIRPS recalées (stockées en simple précision).
- **Température :** observation = T2m horaire ERA5 (0,25°) agrégée en moyenne, maximum et minimum journaliers UTC ; modèles = Tmax/Tmin quotidiens C3S et T2m des **statistiques mensuelles** C3S (mois et saisons seulement).
- **NMME (désactivé le 23/09/2026) :** le dépôt disponible ne livre que la **moyenne d'ensemble**, donc aucune probabilité brute. Le système est mis à `enabled: false` dans la configuration plutôt que supprimé du code : le réactiver sera une ligne le jour où une source avec membres sera trouvée. Ce qui suit reste vrai de ce système — moyenne d'ensemble seulement (pas de membres) et fichiers **mensuels** ; donc mois et saisons uniquement (jamais de décades), et **aucune probabilité brute** : pas de RPSS, de score de Brier ni d'aire ROC. Ces modèles sont notés sur les scores déterministes, et leur éligibilité (§3.3) ne retient que le critère déterministe ; leurs probabilités ne pourront venir que de la calibration (P3). Les échelles autorisées par système sont dans `SYSTEM_SCALES` (`eccas_s2s/operations/skill_raw.py`).
- **Score de Brier :** `verification::brier` est appelé deux fois — seuils fins pour le score et le BSS (le classement par défaut en dix classes déplace la valeur), classement par défaut pour la décomposition fiabilité / résolution / incertitude (avec des seuils fins chaque classe ne contient qu'une prévision : la fiabilité se confond avec le score et la résolution avec l'incertitude). Les deux cas sont couverts par un test.
- **Couples archivés :** `zones/<modèle>/<zone>/pairs.csv` conserve les couples ; `--from-pairs` rejoue les scores R en quelques minutes après une correction du script, sans relire les hindcasts.
- **Masque CEEAC :** tout ce qui est vérifié, cartographié et publié est restreint aux mailles dont le centre tombe dans le shapefile CEEAC (`eccas_s2s.core.geo`, même définition que `create_geographic_mask` de la chaîne de référence). Il n'y a plus de traitement par zones : les scores sont calculés point de grille par point de grille, ce qui est plus détaillé qu'une moyenne de zone.
- **Organisation des sorties :** trois arbres (`netcdf/`, `figures/`, `diagrams/`) partageant les mêmes branches `<système>_<modèle>/<échelle>/<variable>/<métrique>/`. Un netCDF par métrique, une carte par période (dates explicites : « Novembre 2026 », « OND 2026 », « 1ʳᵉ décade de Novembre 2026 »).
- **Cartes lissées.** Les cartes sont tracées avec un **ombrage de Gouraud** — la couleur est
  interpolée entre les centres de maille — puis **découpées sur le contour CEEAC**. Un score varie
  continûment dans l'espace, et une carte en damier se lit comme si chaque maille était une mesure
  indépendante. Le lissage a un effet de bord : une maille manquante efface les quatre quadrilatères
  qui l'entourent, ce qui rongerait la frontière du masque ; les trous sont donc comblés par le plus
  proche voisin **avant** le tracé, et la découpe remet le bord exactement où le masque le met — rien
  de ce qui a été comblé n'est jamais visible.
- **Cartes de gain** (`--kind gain`) : palette divergente centrée sur zéro, brun pour une
  dégradation, vert pour une amélioration, blanc pour |gain| < 0,02. Produites pour les
  **terciles**, produit phare de la prévision saisonnière ; les autres produits sont disponibles
  par `--products`.
- **Lecture des cartes de skill :** même grille de lecture pour toutes les métriques — **gris sous la valeur sans skill, vert au-dessus** — et une phrase sous chaque carte rappelant la condition de bon skill (AUC > 0,5, RPSS > 0, …).
- **Diagrammes de fiabilité et ROC :** tracés en R, à partir des couples de **tous les points de grille du masque** (24 années seules ne remplissent pas dix classes de probabilité). Les intervalles de confiance rééchantillonnent des **années entières** (les points de grille voyagent avec leur année), et une classe alimentée par trop peu d'années est marquée d'une croix grise hors de la courbe. Une figure de fiabilité et une figure ROC par période, les trois catégories sur le même repère.
- **Hindcasts C3S :** demandés avec un jour d'échéance de plus, pour couvrir le 29 février des années bissextiles.
- **Horizon :** `max_lead_days` = horizon **reçu** (DWD 181 j, Météo-France 212 j pour l'init. 09) ; le contrôle qualité signale tout écart avec la configuration.
- **Horizon commun 6 mois d'échéance** (`horizon.max_lead_months` dans le cycle) : le CDS refuse `leadtime_month 7` pour UKMO, BoM et NCEP, qui n'auraient donc au-delà ni mois ni saison. Une période est retenue si `month_offset + n_months <= 6` — pour une init. de septembre, jusqu'à **février 2027** et **DJF 2026-27**.
- **Choix des périodes (`--periods`)** : toute étape accepte le même sélecteur, et c'est lui qui décide dès qu'il est donné — une commande d'exploitation ne dépend donc jamais de ce que contient le bloc `development`.

  | Sélecteur | Effet |
  |---|---|
  | *(aucun)* | sous-ensemble de développement (`development.periods_per_scale`, 2 périodes par échelle) |
  | `--all-periods` ou `--periods all` | tout l'horizon |
  | `--periods all-months` / `all-seasons` / `all-decades` | une échelle entière |
  | `--periods season_m1` | une période par sa clé |
  | `--periods "OND 2026"` | une période par le libellé imprimé sur les cartes (français accepté : `"1ʳᵉ décade de Novembre 2026"`) |
  | `--periods all-seasons 2026-10` | plusieurs sélecteurs à la fois (la virgule marche aussi) |

  Rien n'est propre à une initialisation : les sélecteurs sont **relatifs au mois du cycle** (avec une init. de septembre `season_m0` vaut SON, avec une init. de mars il vaut MAM) et les libellés sont engendrés à partir de la date d'initialisation du cycle. Ils se tapent sans accent ni exposant (`1re decade de decembre 2026`) et sans casse.

  Un nom inconnu **arrête** l'étape (`SelectionError`) en listant les valeurs admises : une faute de frappe ne doit pas ressembler à un modèle sans données. Le sélecteur retenu est journalisé et écrit dans le manifeste ; sans lui, le sous-ensemble de développement est signalé en **avertissement**.
- **Sous-ensemble de développement** (`development.periods_per_scale`) : les deux premières décades, les deux premiers mois et les deux premières saisons, pour la mise au point seulement. En exploitation, on passe `--periods`, ou on retire le bloc de la configuration.
- **Un seul point de passage pour les périodes :** `cfg.periods_for(horizon_days, scales, selection=...)` (`eccas_s2s/settings.py`). Horizon, sélecteur et sous-ensemble y sont appliqués une fois pour toutes ; aucune opération ne rappelle `build_periods` directement.
- **Écriture partielle :** un run sur une sélection réécrit le fichier de cumuls en entier ; si le fichier existant contenait des périodes absentes de la sélection, l'étape l'**écrit en avertissement** avant de les remplacer.

## Notebooks

Deux familles :

- `notebooks/pedagogiques/` — expliquent la méthode, avec figures ; **commités avec leurs sorties**.
- `notebooks/operationnels/` — exécutent l'outil : une cellule *Paramètres*, puis « Exécuter tout » ; **commités sans sorties** (elles dépendent du cycle).

| Notebook | Type | Contenu |
|---|---|---|
| `pedagogiques/00_P0_fondations.ipynb` | pédagogique | configuration, provenance, périodes, dates des cumuls, faisabilité agro |
| `pedagogiques/01_observations_chirps.ipynb` | pédagogique | QC CHIRPS, cumuls calendaires, normales et percentiles par maille, passage à 1°, sensibilité à la période de référence |
| `pedagogiques/02_cumuls_c3s.ipynb` | pédagogique | cumuls C3S par période, biais de moyenne et de variabilité des modèles bruts |
| `operationnels/OP_01_telechargement_qc_c3s.ipynb` | opérationnel | téléchargement C3S + contrôle qualité + horizon commun |
| `operationnels/OP_02_reference_chirps.ipynb` | opérationnel | archive CHIRPS + normales (idempotent) |
| `pedagogiques/03_temperature_nmme.ipynb` | pédagogique | référence ERA5, percentiles extrêmes par maille, biais de température des modèles, NMME |
| `operationnels/OP_03_cumuls_c3s.ipynb` | opérationnel | cumuls C3S par période |
| `operationnels/OP_04_reference_temperature_era5.ipynb` | opérationnel | téléchargement ERA5 horaire → journalier, archive et normales |
| `operationnels/OP_05_temperatures_c3s.ipynb` | opérationnel | téléchargement et périodes de T2m, Tmax, Tmin |
| `operationnels/OP_06_nmme.ipynb` | opérationnel | téléchargement NMME et valeurs par mois et saison — **suspendu** (NMME désactivé) |
| `operationnels/OP_07_skill_brut.ipynb` | opérationnel | scores bruts, cartes et diagrammes (P2) |
| `pedagogiques/04_skill_brut.ipynb` | pédagogique | lecture des scores bruts, cartes et diagrammes |
| `pedagogiques/05_calibration.ipynb` | pédagogique | pourquoi une méthode par produit, lire un gain sans se tromper, pourquoi l'anomalie et le SPI restent bruts |
| `operationnels/OP_08_calibration.ipynb` | opérationnel | calibration, scores calibrés, comparaison, registre, cartes de gain et diagrammes |

## Git

Commits au fil des phases, poussés vers `github.com/ludric-flan/eccas-s2s` (auteur `ludric-flan <armandludricngongang@gmail.com>`). Une prévision émise doit être produite depuis un commit propre (`dirty: false` dans le manifeste).
