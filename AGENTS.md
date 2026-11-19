# Data Engine agent guidance

Read [README.md](README.md) and [docs/DESIGN.md](docs/DESIGN.md) first.

## Rules

- Platform agnostic (Windows and Unix). Run Python from the repository root with the
  repository `.venv`; workers under `src/data_engine/workers/` run in another repository's
  interpreter (see `config.SITE_DEFAULTS["interpreters"]`) and must only import
  `data_engine.dataset`, `data_engine.video` helpers and that repository.
- Never invent labels: no interpolation, extrapolation or default identity. What cannot be
  measured is an unknown or NaN. A stage reports the complete open set of every unknown
  kind it owns.
- Human evidence is append-only and keyed by detection keys (`camera/frame/index`). Never
  key anything durable on tracklet, component or identity numbers. Never weaken or bypass
  a human `different`; supersede only with a later explicit human answer.
- The raw detection store is immutable. Filtering, rejection and suppression are
  assignments (`kept`, `tracklet = -1`, `not_person`), never deletions.
- Every threshold that can change a label lives in `config.RECIPE`. Changing one is a
  recipe change: record it in the decision log of `docs/DESIGN.md`. The post-filter and
  tracker settings must stay equal to `cvat_labeling/prelabel_policy.json`.
- Do not modify `data_management`, `cvat_labeling` or `Autofit` from this repository's
  work unless the user asks. Alignment logic belongs to
  `data_management.alignment_generator`; fine-tuning belongs to `cvat_labeling/finetuning`.
- Labels are published only through the data_management label store
  (`stages/publish.py`), never by writing files into `data/labels`. Publication to the real
  store is an explicit operator action; machine provenance needs a commit of this repo.
- Do not start GPU jobs or kill processes you did not start without asking; the detector
  stage is resumable and can be re-run.
- Tests: `.venv\Scripts\python.exe -m pytest tests -q` (no GPU or recordings needed).
