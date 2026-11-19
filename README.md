# Data Engine

A repeatable, human-in-the-loop workflow that turns a multi-camera recording into labels:
camera **time alignment**, **2D pose detections and tracklets**, and **3D person tracks**
with persistent identities. It runs every machine step it can, reports exactly what it
could not decide, asks a person only those questions, feeds the answers back as constraints
and repeats until nothing is unknown.

```
provide data ─► ingest ─► 2D detection ─► calibration check ─► time alignment ─► 2D tracklets ─► Autofit 3D tracking
                                                                                     │
        ┌────────────────────────────── every round ───────────────────────────────┐ │
        │  3D association + identity + swap check ─► coverage recovery ─► human     │◄┘
        │            ▲                              tasks ─► answers ─► constraints │
        │            └──────────────────────────────────────────────────────────┘   │
        └────────────────────────────────────────────────────────────────────────────┘
                                       │ no open unknowns
                                       ▼
                                    publish  (data_management label sets)
```

It builds on what the golden-set work in `cvat_labeling` and `data_management` established
(see [docs/DESIGN.md](docs/DESIGN.md) for the mapping from learnings to design):

| Piece | Reused from |
|---|---|
| Dataset catalog, verified chunk layouts, exact frame access | `data_management` catalog / frame index (file joins it cannot verify are checked by pixel comparison) |
| Detector (fine-tuned YOLOv8x-pose-p6 + ViTPose, TensorRT) | `data_management` live path, `cvat_labeling` curated v6 weights |
| Detection post-filter and 10 fps BoT-SORT settings | `cvat_labeling/prelabel_policy.json` (read at run time; must equal the engine recipe) |
| Time alignment | `data_management.alignment_generator` driven over the whole recording, then verified by geometry |
| Detection-level 3D tracking with a human prior | `Autofit` (`HumanBotSort`), run as an evidence stage on CPU |
| Mount-shift check | `data_management.mount_check` |
| Camera model, triangulation gates, physical veto | `cvat_labeling` `pseudo3d.geometry` / `identity_triangulation` contract (tested for equality) |
| Label publication | `data_management` label store |

## Quick start

```powershell
cd C:\Users\iantw\git\data_engine
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m data_engine.cli init felix_20251223173052_20251223183103
.venv\Scripts\python.exe -m data_engine.cli run  felix_20251223173052_20251223183103
.venv\Scripts\python.exe -m data_engine.cli serve --port 8600
```

Then open <http://127.0.0.1:8600>: the dashboard shows the state of every dataset, the
pipeline, the open unknowns and the human tasks; **Review tasks** opens the task runner.
After answering, press **Run next round** (or `data-engine run <dataset>` again).

A pilot on a slice of a recording uses a tagged workspace:

```powershell
.venv\Scripts\python.exe -m data_engine.cli init "felix_20251223173052_20251223183103@pilot40s" --limit-frames 400
```

| Command | What it does |
|---|---|
| `data-engine list` | Workspaces with phase, round, open unknowns and tasks |
| `data-engine init DATASET[@tag] [--limit-frames N]` | Create a workspace (idempotent) |
| `data-engine run DATASET [--until STAGE] [--force STAGE …]` | One round: apply answers, rerun what changed, create tasks |
| `data-engine status DATASET [--json]` | Phase, stages, unknowns, tasks, next actions |
| `data-engine tasks DATASET` | List human tasks |
| `data-engine serve` | Web dashboard and task runner |
| `data-engine publish DATASET` | Build and validate label-set drafts at 10 Hz (every frame); publish only when enabled |

## How it decides when it is done

Every stage reports the **complete set of things it could not decide** (the *unknown
ledger*). A dataset is complete when no unknown is open: each one is resolved by a machine
rule, resolved by a human answer, or explicitly marked *undeterminable* (occluded, out of
view, too little evidence to measure). The engine never interpolates, extrapolates or
assumes a value to close an unknown. Each round's open set is recorded; a round that does
not shrink it is flagged on the dashboard, because the recursion is only justified while
it converges.

## Layout

```
src/data_engine/
  engine.py        state machine: stages, freshness, rounds, status
  db.py            per-dataset SQLite: rounds, stage runs, unknowns, tasks, constraints, events
  taxonomy.py      unknown kinds and human task types
  config.py        site paths and the frozen machine recipe (hashed into every manifest)
  stages/          ingest, detect, calibration, align, tracklets, autofit, solve, recover, tasks, publish
  workers/         steps that run in another repository's interpreter (TensorRT, catalog, label store)
  solve/           evidence, clustering, 3D tracks, refinement, identity, swap check, coverage recovery
  tasks/apply.py   answers -> constraints
  web/             dashboard + task runner (FastAPI, static JS)
tests/             pytest (no GPU or recordings needed, except the calibration file for geometry tests)
docs/DESIGN.md     design, taxonomy, algorithms, decision log
```

Workspaces live under `C:\data_engine_data\<dataset>[@tag]\` (state database, immutable
detection shards, per-round artifacts, one run manifest per stage run). Site paths and
interpreters are in `config.py` and can be overridden with `engine.local.json`.

## Evidence for the solver's rules

Every rule that decides something without a human was scored against the golden identities
(`experiments/bench/`); what was adopted and what was rejected, with numbers, is in
[experiments/RESULTS.md](experiments/RESULTS.md). To re-run the bench after a change:

```powershell
.venv\Scripts\python.exe experiments\bench\bench.py adopted
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```
