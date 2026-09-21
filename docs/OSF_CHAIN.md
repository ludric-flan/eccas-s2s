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
| E2 · contrôle qualité C3S | `python scripts/run_qc_c3s.py --config config/cycle_202609.yaml --variable precip` | P0 |
| E1 · référence CHIRPS (archive + normales, seulement si CHIRPS a changé) | `python scripts/run_obs_chirps.py --config config/cycle_202609.yaml` | P1 |
| E2 · cumuls C3S par période | `python scripts/run_c3s_totals.py --config config/cycle_202609.yaml` | P1 |
| E2 · températures C3S par période | `python scripts/run_c3s_temperature.py --config config/cycle_202609.yaml` | P1 |
| E1 · CHIRPS 1981–1990 (une fois) | `python scripts/run_download_chirps.py --config ... --years 1981 1990` | P1 |
| E1 · température observée ERA5 (horaire → journalier) | `python scripts/run_download_era5_hourly.py --config ... --years 1981 2026 --workers 4` | P1 |
| E1 · référence ERA5 (archive + normales) | `python scripts/run_obs_era5.py --config config/cycle_202609.yaml` | P1 |
| E2 · NMME (téléchargement puis périodes) | `python scripts/run_download_nmme.py --config ...` puis `run_nmme_totals.py` | P1 |
| E4 · skill brut (cartes + scores de zone R) | `python scripts/run_skill_raw.py --config config/cycle_202609.yaml --systems c3s nmme --variables precip t2m tmax tmin` | P2 |
| E4 · cartes de skill | `python scripts/run_plot_skill_raw.py --config config/cycle_202609.yaml --scores pearson rpss roc_area msess` | P2 |
| E4 · diagrammes de fiabilité et ROC (R) | `python scripts/run_skill_diagrams.py --config config/cycle_202609.yaml --scales month season --zones domain` | P2 |
| E4 · reconstruire le tableau de synthèse depuis les cartes | `python scripts/run_skill_raw.py --config config/cycle_202609.yaml --from-maps` | P2 |
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
| Cumuls C3S par période | `DATA_OSF/derived/c3s/<YYYYMM>/` |
| Journaux et manifestes d'exécution | `OUTPUTS_OSF/runs/<run_id>/{run.log, manifest.json}` |
| Cartes de skill brut | `OUTPUTS_OSF/skill/<YYYYMM>/raw/maps/<système>_<modèle>_<variable>_skill.nc` |
| Scores de zone (R) | `OUTPUTS_OSF/skill/<YYYYMM>/raw/zones/<système>_<modèle>_<variable>/<zone>/` |
| Figures de skill | `OUTPUTS_OSF/skill/<YYYYMM>/raw/figures/` et `.../diagrams/<...>/<zone>/` |
| Synthèse et éligibilité | `OUTPUTS_OSF/skill/<YYYYMM>/raw/skill_raw_summary.csv`, `OUTPUTS_OSF/registry/models_eligibility.csv` |
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
| `eccas_s2s.validate.zones` | masques du domaine et des trois zones pluviométriques, fraction de skill positif |
| `eccas_s2s.validate.pooled` | couples de tous les points de grille d'une zone, pour les diagrammes |
| `eccas_s2s.validate.r_bridge` + `zone_scores.R` + `zone_diagrams.R` | scores de zone et diagrammes avec le paquet R `verification` |
| `eccas_s2s.viz.ceeac_maps` | cartes à la charte CAPC-AC (shapefile CEEAC, logo, barre de couleur commune) |
| `eccas_s2s.operations.*` | étapes opérationnelles (`run(...)` + `main(argv)`) : `download_c3s`, `qc_c3s`, `obs_chirps`, `c3s_totals` |

Les modules historiques (`eccas_s2s.config`, `core.processing`, `pipeline`) sont conservés tels quels.

## Conventions à retenir

- **Dates des cumuls C3S :** le champ de l'échéance +24 h (daté du 2 à 00 UTC) contient la pluie du **1er**. L'ancienne chaîne décalait toutes les périodes d'un jour ; la chaîne OSF les aligne sur le calendrier des observations. Voir le notebook 00, section 4.
- **Périodes complètes uniquement :** une période qui dépasse l'horizon du modèle n'est pas produite.
- **Indicateurs de saison :** produits seulement si toute leur fenêtre de référence est dans la prévision ; une fenêtre déjà commencée n'est pas complétée avec des observations.
- **Production :** lancer la chaîne depuis un code commité (`dirty: false` dans le manifeste).

- **Ensembles UKMO et BoM :** fichiers bruts conservés tels que téléchargés, traités avec les membres disponibles au 1er du mois (2–11 membres) ; le contrôle qualité le signale.
- **Normales observées :** 1991–2020, par maille et par période calendaire (`dekad_MM_D`, `month_MM`, `season_MM`), percentiles estimés par la position de Weibull ; communes à tous les modèles (D12).
- **Grilles :** CHIRPS 0,05° et C3S 1° sont emboîtées (20 × 20) : passage à 1° par moyenne par blocs exacte. Coordonnées CHIRPS recalées (stockées en simple précision).
- **Température :** observation = T2m horaire ERA5 (0,25°) agrégée en moyenne, maximum et minimum journaliers UTC ; modèles = Tmax/Tmin quotidiens C3S et T2m des **statistiques mensuelles** C3S (mois et saisons seulement).
- **NMME :** moyenne d'ensemble seulement (pas de membres) et fichiers **mensuels** ; donc mois et saisons uniquement (jamais de décades), et **aucune probabilité brute** : pas de RPSS, de score de Brier ni d'aire ROC. Ces modèles sont notés sur les scores déterministes, et leur éligibilité (§3.3) ne retient que le critère déterministe ; leurs probabilités ne pourront venir que de la calibration (P3). Les échelles autorisées par système sont dans `SYSTEM_SCALES` (`eccas_s2s/operations/skill_raw.py`).
- **Diagrammes de fiabilité et ROC :** tracés en R, à partir des couples de **tous les points de grille** de la zone (24 années seules ne remplissent pas dix classes de probabilité). Les intervalles de confiance rééchantillonnent des **années entières** (les points de grille voyagent avec leur année), et une classe alimentée par trop peu d'années est marquée d'une croix grise hors de la courbe. Une figure de fiabilité et une figure ROC par période, les trois catégories sur le même repère.
- **Hindcasts C3S :** demandés avec un jour d'échéance de plus, pour couvrir le 29 février des années bissextiles.
- **Horizon :** `max_lead_days` = horizon **reçu** (DWD 181 j, Météo-France 212 j pour l'init. 09) ; le contrôle qualité signale tout écart avec la configuration.

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
| `operationnels/OP_06_nmme.ipynb` | opérationnel | téléchargement NMME et valeurs par mois et saison |

## Git

Commits au fil des phases, poussés vers `github.com/ludric-flan/eccas-s2s` (auteur `ludric-flan <armandludricngongang@gmail.com>`). Une prévision émise doit être produite depuis un commit propre (`dirty: false` dans le manifeste).
