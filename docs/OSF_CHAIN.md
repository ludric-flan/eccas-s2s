# Chaîne OSF CAPC-AC dans eccas-s2s — guide d'utilisation

Workflow de référence : `WORKFLOW_OSF_CAPC-AC_v0.3` (page web et `.md` dans `Previsions_S2S/`).
La chaîne tourne **chaque mois** avec l'initialisation du mois (1 cycle = 1 fichier `config/cycle_YYYYMM.yaml`).

## Environnement

```bash
conda activate eccas-s2s
pip install -e .          # une fois
pytest -q                 # tous les tests doivent passer
```

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
- **NMME :** moyenne d'ensemble seulement (pas de membres), mensuel, donc mois et saisons ; 6 modèles depuis 1982 ou 1991.
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
| `operationnels/OP_03_cumuls_c3s.ipynb` | opérationnel | cumuls C3S par période |

## Git

Commits au fil des phases, poussés vers `github.com/ludric-flan/eccas-s2s` (auteur `ludric-flan <armandludricngongang@gmail.com>`). Une prévision émise doit être produite depuis un commit propre (`dirty: false` dans le manifeste).
