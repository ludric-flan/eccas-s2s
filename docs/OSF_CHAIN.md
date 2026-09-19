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
| … | (ajoutés au fil des phases) | |

Chaque étape existe aussi en notebook opérationnel (voir plus bas). Scripts et notebooks appellent la même fonction `run(...)` de `eccas_s2s/operations/`.

Tester une requête sans télécharger : ajouter `--dry-run`.

## Où sont les fichiers

| Contenu | Emplacement (défini dans le YAML) |
|---|---|
| Données brutes immuables | `DATA_OSF/raw/<système>/<YYYYMM>/` |
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
| `eccas_s2s.operations.*` | étapes opérationnelles (`run(...)` + `main(argv)`) : `download_c3s`, `qc_c3s` |

Les modules historiques (`eccas_s2s.config`, `core.processing`, `pipeline`) sont conservés tels quels.

## Conventions à retenir

- **Dates des cumuls C3S :** le champ de l'échéance +24 h (daté du 2 à 00 UTC) contient la pluie du **1er**. L'ancienne chaîne décalait toutes les périodes d'un jour ; la chaîne OSF les aligne sur le calendrier des observations. Voir le notebook 00, section 4.
- **Périodes complètes uniquement :** une période qui dépasse l'horizon du modèle n'est pas produite.
- **Indicateurs de saison :** produits seulement si toute leur fenêtre de référence est dans la prévision ; une fenêtre déjà commencée n'est pas complétée avec des observations.
- **Production :** lancer la chaîne depuis un code commité (`dirty: false` dans le manifeste).

- **Ensembles UKMO et BoM :** fichiers bruts conservés tels que téléchargés, traités avec les membres disponibles au 1er du mois (2–11 membres) ; le contrôle qualité le signale.
- **Horizon :** `max_lead_days` = horizon **reçu** (DWD 181 j, Météo-France 212 j pour l'init. 09) ; le contrôle qualité signale tout écart avec la configuration.

## Notebooks

Deux familles :

- `notebooks/pedagogiques/` — expliquent la méthode, avec figures ; **commités avec leurs sorties**.
- `notebooks/operationnels/` — exécutent l'outil : une cellule *Paramètres*, puis « Exécuter tout » ; **commités sans sorties** (elles dépendent du cycle).

| Notebook | Type | Contenu |
|---|---|---|
| `pedagogiques/00_P0_fondations.ipynb` | pédagogique | configuration, provenance, périodes, dates des cumuls, faisabilité agro |
| `operationnels/OP_01_telechargement_qc_c3s.ipynb` | opérationnel | téléchargement C3S + contrôle qualité + horizon commun |

## Git

Commits au fil des phases, poussés vers `github.com/ludric-flan/eccas-s2s` (auteur `ludric-flan <armandludricngongang@gmail.com>`). Une prévision émise doit être produite depuis un commit propre (`dirty: false` dans le manifeste).
