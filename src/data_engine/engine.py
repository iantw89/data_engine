"""The Data Engine state machine.

    provide data -> [ingest -> detect -> calibration -> align -> tracklets]      (once)
                 -> [solve -> gapfill -> tasks]                                  (every round)
                 -> human answers -> constraints -> next round ... -> publish

A *round* is one pass over the stages whose inputs changed. Stages are pure functions of
(recipe section, upstream outputs, active constraints); each records an input hash, so a
stage reruns exactly when something it depends on changed - new human answers, a retrained
model, a recipe change - and never otherwise.

Progress is measured by the unknown ledger: every stage reports the complete set of things
it could not decide. A round that does not shrink the open set is flagged, because the
recursion is only justified while it drives toward a complete solution.
"""

from __future__ import annotations

import importlib
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import RECIPE, recipe_hash
from .taxonomy import KINDS, TASK_TYPES
from .workspace import Workspace, file_sha256, stable_hash


@dataclass
class StageResult:
    status: str = "done"  # 'done' | 'blocked' | 'failed' | 'skipped'
    metrics: dict = field(default_factory=dict)
    message: str = ""
    outputs: dict[str, str] = field(default_factory=dict)  # name -> path relative to workspace
    inputs: dict[str, Any] = field(default_factory=dict)  # hashes/identities of what was consumed
    # Complete open set per unknown kind owned by this stage (kinds absent are left untouched).
    unknowns: dict[str, list[dict]] = field(default_factory=dict)
    # Identity of what downstream stages consume. Default: every run counts as new output.
    # A stage that can tell its output did not change sets this, so dependants stay fresh.
    output_hash: str | None = None
    # Earlier stages that should run again in this round because this stage produced something they
    # consume (a lap: e.g. recovered detections for Autofit). The stage itself judges whether a lap is
    # worth its cost; the engine only bounds how many a round may take (MAX_LAPS).
    rerun: tuple[str, ...] = ()


@dataclass
class Context:
    workspace: Workspace
    round: int
    stage: str

    @property
    def db(self):
        return self.workspace.db

    def log(self, message: str, level: str = "info") -> None:
        self.db.log(message, level=level, stage=self.stage)
        print(f"[{self.stage}] {message}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    description: str
    module: str  # data_engine.stages.<module> exposing run(context) -> StageResult
    requires: tuple[str, ...] = ()
    recipe: tuple[str, ...] = ()  # recipe sections hashed into the input hash
    constraints: tuple[str, ...] = ()  # constraint kinds that are inputs
    owns: tuple[str, ...] = ()  # unknown kinds this stage reports
    every_round: bool = False  # part of the recursive loop

    def run(self, context: Context) -> StageResult:
        return importlib.import_module(f"data_engine.stages.{self.module}").run(context)


STAGES: list[Stage] = [
    Stage("ingest", "Ingest", "Resolve camera files into verified frame sequences; map cameras to calibration.",
          "ingest", owns=("stream_layout",)),
    Stage("detect", "2D detection", "Fine-tuned YOLO + ViTPose on every native frame (TensorRT).",
          "detect", requires=("ingest",), recipe=("detect",)),
    Stage("calibration", "Calibration check", "Check each camera's view against the calibration capture (mount shift).",
          "calibration", requires=("ingest",), constraints=("calibration",), owns=("calibration",)),
    Stage("align", "Time alignment", "Per-camera time offsets from pose motion (and audio) over the whole recording.",
          "align", requires=("detect", "calibration"), constraints=("alignment",), owns=("alignment",)),
    Stage("tracklets", "2D tracklets", "Post-filter detections and build over-split single-camera tracklets (BoT-SORT).",
          "tracklets", requires=("detect",), recipe=("postfilter", "tracklets"), constraints=("split", "not_person")),
    Stage("autofit", "Autofit 3D tracking",
          "Independent detection-level 3D tracking (Autofit, CPU): evidence that cuts and links 2D tracklets.",
          "autofit", requires=("tracklets", "align", "calibration")),
    Stage("solve", "3D association and identity",
          "Cross-camera association, 3D tracks, identity across gaps and triangulation, iterated to a fixpoint.",
          "solve", requires=("tracklets", "align", "calibration", "autofit"),
          recipe=("geometry", "association", "temporal", "refine", "swap_check"),
          constraints=("same", "different", "split", "not_person", "duplicate", "undeterminable"),
          owns=("association", "identity_link", "unassigned", "tracklet_purity", "pose_ownership", "coverage",
                "alignment", "identity_swap"),
          every_round=True),
    Stage("recover", "Coverage recovery",
          "Read poses where tracked people must be (guided ViTPose crops) and triangulate them: fills gaps "
          "left by detector misses without interpolating.",
          # RECIPE["autofit_refresh"] is not an input: it decides about a lap after the recovery and changes no
          # recovered position, so tuning it must not make an hour of recovery out of date.
          "recover", requires=("solve",), recipe=("coverage", "geometry"), every_round=True),
    Stage("tasks", "Human tasks", "Turn the remaining unknowns into the smallest set of human questions.",
          "tasks", requires=("solve",), every_round=True),
    Stage("publish", "Publish", "Publish alignment, 2D tracklets and 3D tracks as data_management label sets.",
          "publish", requires=("solve",)),
]
STAGE = {stage.name: stage for stage in STAGES}
MAX_LAPS = 4  # hard bound on laps per round, whatever the stages ask for
RERUN_KEY = "rerun_requested"  # stages asked to run again; persisted so an interrupted round resumes the lap


class Engine:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.db = workspace.db

    # ------------------------------------------------------------------ freshness
    def _constraint_digest(self, kinds: tuple[str, ...]) -> str:
        if not kinds:
            return ""
        rows = [(c["id"], c["kind"]) for kind in kinds for c in self.db.constraints(kind)]
        return stable_hash(sorted(rows))

    def input_hash(self, stage: Stage) -> str:
        upstream = []
        for name in stage.requires:
            run = self.db.latest_stage_run(name, "done")
            identity = (run["metrics"].get("_output_hash") or run["id"]) if run else None
            upstream.append((name, identity))
        dataset = self.workspace.dataset_json
        return stable_hash({
            "recipe": {section: RECIPE[section] for section in stage.recipe},
            "upstream": upstream,
            # The resolved stream layout is an input of everything indexed by native frame.
            "dataset": file_sha256(dataset) if stage.requires and dataset.is_file() else None,
            "constraints": self._constraint_digest(stage.constraints),
            "models": self.db.get_meta(f"models:{stage.name}", None),
        })

    def stage_state(self, stage: Stage) -> str:
        run = self.db.latest_stage_run(stage.name)
        if run is None:
            return "pending"
        if run["status"] == "running":
            return "running"
        if run["status"] in ("failed", "blocked"):
            return run["status"]
        if run["status"] == "skipped":
            return "skipped"
        return "done" if run["input_hash"] == self.input_hash(stage) else "stale"

    # ------------------------------------------------------------------ answers -> constraints
    def apply_answers(self, round_number: int) -> int:
        from .tasks.apply import apply_task_answer
        applied = 0
        for task in self.db.tasks(status="answered"):
            apply_task_answer(self.workspace, task, round_number)
            self.db.set_task_status(task["id"], "applied", applied_round=round_number)
            applied += 1
        return applied

    # ------------------------------------------------------------------ run
    def run_stage(self, stage: Stage, round_number: int) -> StageResult:
        context = Context(self.workspace, round_number, stage.name)
        input_hash = self.input_hash(stage)
        run_id = self.db.start_stage(stage.name, round_number, input_hash)
        context.log("started")
        try:
            result = stage.run(context)
        except Exception as exc:  # recorded, never swallowed: the stage is 'failed' on the dashboard
            self.db.finish_stage(run_id, "failed", message=f"{type(exc).__name__}: {exc}")
            context.log(f"failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}", level="error")
            return StageResult(status="failed", message=str(exc))
        foreign = [kind for kind in result.unknowns if kind not in stage.owns]
        if foreign:
            message = f"Stage {stage.name} reported unknown kinds it does not own: {foreign}"
            self.db.finish_stage(run_id, "failed", message=message)
            raise RuntimeError(message)
        try:
            for kind, items in result.unknowns.items():
                change = self.db.report_unknowns(stage.name, kind, items, round_number)
                context.log(f"unknowns[{kind}]: {len(items)} open (+{change['opened']} / -{change['closed']})")
        except Exception as exc:  # the stage ran but its report could not be stored: failed, not left running
            self.db.finish_stage(run_id, "failed", message=f"could not record unknowns: {type(exc).__name__}: {exc}")
            context.log(f"failed while recording unknowns: {type(exc).__name__}: {exc}; " + traceback.format_exc(), level="error")
            return StageResult(status="failed", message=f"could not record unknowns: {exc}")
        manifest = self.workspace.write_manifest(
            stage.name, round_number, status=result.status,
            config={section: RECIPE[section] for section in stage.recipe}, inputs=result.inputs,
            outputs=result.outputs, notes=result.message)
        metrics = {**result.metrics, "_output_hash": result.output_hash} if result.output_hash else result.metrics
        self.db.finish_stage(run_id, result.status, metrics=metrics, message=result.message, manifest=manifest)
        context.log(f"{result.status}: {result.message}" if result.message else result.status)
        return result

    def advance(self, *, until: str | None = None, trigger: str = "manual", force: tuple[str, ...] = ()) -> dict:
        """Run every stage whose inputs changed, then turn remaining unknowns into tasks."""
        # A stage still marked running belongs to a process that died (the caller holds the
        # run lock, so nothing else is running): record that instead of blocking forever.
        for row in self.db.query("SELECT id, stage FROM stage_runs WHERE status = 'running'"):
            self.db.finish_stage(row["id"], "failed", message="interrupted: the process running this stage ended")
            self.db.log("marked interrupted stage run as failed", level="warning", stage=row["stage"])
        pending_answers = len(self.db.tasks(status="answered"))
        if self.db.current_round() < 0:
            round_number = self.db.open_round("init")
        else:
            round_number = self.db.open_round("human_answers" if pending_answers and trigger == "manual" else trigger)
        applied = self.apply_answers(round_number) if pending_answers else 0
        if applied:
            self.db.log(f"round {round_number}: applied {applied} human answers")
        ran, stopped, laps = [], None, 0
        forced, every_round = set(force), bool(applied)
        while True:
            lap_from = None
            for stage in STAGES:
                if stage.name == "publish":
                    continue  # publication is an explicit action, never a side effect of a round
                state = self.stage_state(stage)
                blocked_by = [name for name in stage.requires if self.stage_state(STAGE[name]) != "done"]
                if blocked_by:
                    stopped = f"{stage.name} waits for {', '.join(blocked_by)}"
                    break
                requested = list(self.db.get_meta(RERUN_KEY, []))
                rerun = state != "done" or stage.name in forced or stage.name in requested \
                    or (stage.every_round and every_round)
                if rerun:
                    result = self.run_stage(stage, round_number)
                    ran.append(stage.name)
                    if result.status != "done":
                        stopped = f"{stage.name}: {result.status} - {result.message}"
                        break
                    forced.discard(stage.name)
                    if stage.name in requested:
                        self.db.set_meta(RERUN_KEY, [name for name in requested if name != stage.name])
                    wanted = [name for name in result.rerun if name in STAGE]
                    if wanted:
                        self.db.set_meta(RERUN_KEY, sorted(set(self.db.get_meta(RERUN_KEY, [])) | set(wanted)))
                        if until in (None, "tasks") and laps < MAX_LAPS:
                            lap_from = stage.name
                            break
                        self.db.log(f"{stage.name} asked for another lap ({', '.join(wanted)}); left for the next round",
                                    level="warning", stage=stage.name)
                if until == stage.name:
                    break
            if lap_from is None:
                break
            laps += 1
            every_round = False  # the lap reruns what its inputs make stale; nothing else
            self.db.log(f"round {round_number}: lap {laps} - {lap_from} produced input for "
                        f"{', '.join(self.db.get_meta(RERUN_KEY, []))}")
        summary = self.round_summary(round_number, ran, applied, stopped)
        summary["laps"] = laps
        self.db.close_round(round_number, summary)
        return summary

    def round_summary(self, round_number: int, ran: list[str], applied: int, stopped: str | None) -> dict:
        counts = self.db.unknown_counts()
        open_by_kind = {kind: values.get("open", 0) for kind, values in counts.items() if values.get("open")}
        severity = {kind: round(values.get("open_severity", 0.0), 1) for kind, values in counts.items()
                    if values.get("open")}
        previous = [r for r in self.db.rounds() if r["round"] < round_number and r["summary"].get("open_severity_total") is not None]
        total = sum(open_by_kind.values())
        severity_total = round(sum(severity.values()), 1)
        shrinking = None
        if previous and "solve" in ran:
            last = previous[-1]["summary"]
            # Severity is label volume at stake; it is the measure that must fall. The count
            # only breaks ties (the same volume concentrated in fewer questions is progress).
            shrinking = (severity_total < last["open_severity_total"] or total == 0
                         or (severity_total == last["open_severity_total"] and total < last["open_total"]))
            if not shrinking:
                self.db.log(f"round {round_number} did not shrink the unknown set "
                            f"({last['open_total']} -> {total}); more human input or a different lever is needed",
                            level="warning")
        solve = self.db.latest_stage_run("solve", "done")
        return {"stages_run": ran, "answers_applied": applied, "stopped": stopped, "open_by_kind": open_by_kind,
                "open_severity_by_kind": severity, "open_total": total, "open_severity_total": severity_total,
                "shrinking": shrinking, "coverage": (solve or {}).get("metrics", {}).get("coverage", {}),
                "tasks_open": sum(v.get("open", 0) for v in self.db.task_counts().values())}

    # ------------------------------------------------------------------ status
    def progress(self, stage: Stage) -> dict | None:
        path = self.workspace.path("logs", f"{stage.name}_progress.json")
        if not path.is_file():
            return None
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        progress["age_seconds"] = round(time.time() - progress.get("updated", 0), 1)
        if "eta_minutes" in progress:
            progress.setdefault("eta_seconds", round(progress["eta_minutes"] * 60))
        if progress.get("rate_fps") is not None:
            progress.setdefault("message", f"{progress['rate_fps']} frames/s")
        return progress

    # ------------------------------------------------------------------ what ran when
    @staticmethod
    def _sequence(runs: list[dict], finished: float | None = None) -> list[dict]:
        """Stage runs of one round in the order they ran, each with the lap it belongs to (0 = first pass).

        A round walks the stages in order; when it goes back to an earlier stage, a lap begins. A stage started
        after the round had finished (one stage run on its own, e.g. the label-set drafts) is recorded under that
        round but is not part of it: ``after_round``."""
        position = {stage.name: index for index, stage in enumerate(STAGES)}
        steps, lap, previous = [], 0, -1
        for run in runs:
            here = position.get(run["stage"], 0)
            if here <= previous:
                lap += 1
            previous = here
            stage = STAGE.get(run["stage"])
            steps.append({
                "stage": run["stage"], "title": stage.title if stage else run["stage"], "status": run["status"],
                "lap": lap, "started": run["started"], "finished": run["finished"],
                # the round's own last stage can start and end on the clock tick the round finishes at
                "after_round": bool(finished is not None and run["started"] >= finished
                                    and (run["finished"] is None or run["finished"] > finished)),
                "duration_s": round(run["finished"] - run["started"], 1) if run["finished"] else None,
                "message": run["message"] or ""})
        return steps

    def timeline(self) -> dict:
        """The current (or last) round step by step - done, running, expected next - and recent rounds.

        ``expected`` is a prediction: the stages after the running one that are out of date or consume what a
        running or expected stage produces. A stage whose output turns out unchanged leaves its dependants as
        they are, and the recover stage can add a lap, so the list can shrink or grow.
        """
        rounds = self.db.rounds()
        if not rounds:
            return {"round": None, "steps": [], "expected": [], "history": []}
        by_round: dict[int, list[dict]] = {}
        for run in self.db.stage_runs():
            by_round.setdefault(run["round"], []).append(run)
        current = rounds[-1]
        steps = self._sequence(by_round.get(current["round"], []), current["finished"])
        running = next((step for step in steps if step["status"] == "running"), None)
        if running is not None:
            running["progress"] = self.progress(STAGE[running["stage"]]) if running["stage"] in STAGE else None
        expected = []
        if current["finished"] is None and steps and steps[-1]["status"] in ("running", "done"):
            names = [stage.name for stage in STAGES if stage.name != "publish"]
            last = steps[-1]
            requested = set(self.db.get_meta(RERUN_KEY, []))
            changing = {last["stage"]} if last["status"] == "running" else set()
            # a lap that was just requested starts again at the requested stage
            start = min((names.index(name) for name in requested if name in names), default=names.index(last["stage"]) + 1) \
                if last["status"] == "done" else names.index(last["stage"]) + 1
            lap = last["lap"] + (1 if start <= names.index(last["stage"]) else 0)
            for name in names[start:]:
                stage = STAGE[name]
                state = self.stage_state(stage)
                if name in requested:
                    reason = "asked for by " + last["title"] if last["status"] == "done" else "requested"
                elif changing & set(stage.requires):
                    reason = "its input is being recomputed"
                elif state != "done":
                    reason = {"stale": "out of date", "pending": "not run yet"}.get(state, state)
                else:
                    continue
                changing.add(name)
                expected.append({"stage": name, "title": stage.title, "lap": lap, "reason": reason})
        history = []
        for entry in rounds[-8:]:
            sequence = self._sequence(by_round.get(entry["round"], []), entry["finished"])
            summary = entry["summary"] or {}
            history.append({
                "round": entry["round"], "trigger": entry["trigger"], "started": entry["started"],
                "finished": entry["finished"],
                "duration_s": round(entry["finished"] - entry["started"], 1) if entry["finished"] else None,
                "laps": max((step["lap"] for step in sequence if not step["after_round"]), default=0),
                "steps": [{key: step[key] for key in ("stage", "title", "status", "lap", "duration_s", "after_round")}
                          for step in sequence],
                "answers_applied": summary.get("answers_applied"), "open_total": summary.get("open_total"),
                "stopped": summary.get("stopped")})
        return {"round": current["round"], "trigger": current["trigger"], "started": current["started"],
                "finished": current["finished"],
                "laps": max((step["lap"] for step in steps if not step["after_round"]), default=0),
                "steps": steps, "expected": expected, "history": history,
                "lap_requested": sorted(self.db.get_meta(RERUN_KEY, []))}

    def status(self) -> dict:
        dataset = self.workspace.dataset if self.workspace.dataset_json.is_file() else None
        stages = []
        for stage in STAGES:
            run = self.db.latest_stage_run(stage.name)
            state = self.stage_state(stage)
            stages.append({
                "name": stage.name, "title": stage.title, "description": stage.description,
                "status": state, "every_round": stage.every_round, "requires": list(stage.requires),
                "round": run["round"] if run else None, "started": run["started"] if run else None,
                "finished": run["finished"] if run else None,
                "duration_s": round(run["finished"] - run["started"], 1) if run and run["finished"] else None,
                "metrics": run["metrics"] if run else {}, "message": run["message"] if run else "",
                "progress": self.progress(stage) if state == "running" else None,
            })
        counts = self.db.unknown_counts()
        by_kind = []
        for kind in KINDS.values():
            values = counts.get(kind.name, {})
            by_kind.append({"kind": kind.name, "title": kind.title, "meaning": kind.meaning,
                            "machine_steps": list(kind.machine_steps), "task_type": kind.task_type,
                            "open": values.get("open", 0), "resolved": values.get("resolved", 0),
                            "undeterminable": values.get("undeterminable", 0),
                            "open_severity": round(values.get("open_severity", 0.0), 1)})
        task_counts = self.db.task_counts()
        by_type = [{"type": t.name, "title": t.title, "ui": t.ui, "question": t.question,
                    **{s: task_counts.get(t.name, {}).get(s, 0) for s in ("open", "answered", "applied", "skipped", "superseded")}}
                   for t in TASK_TYPES.values()]
        open_unknowns = sum(k["open"] for k in by_kind)
        open_tasks = sum(t["open"] for t in by_type)
        pending_answers = sum(t["answered"] for t in by_type)
        states = {s["name"]: s["status"] for s in stages}
        timeline = self.timeline()
        if any(v == "running" for v in states.values()):
            title = next(s["title"] for s in stages if s["status"] == "running")
            step = next((step for step in timeline["steps"] if step["status"] == "running"), None)
            lap = f", lap {step['lap']}" if step and step["lap"] else ""
            after = [item["title"] for item in timeline["expected"]]
            phase = "running"
            if step and step["after_round"]:
                detail = f"{title} is running on its own (round {timeline['round']} has finished; this is not a new round)"
            else:
                detail = f"Round {timeline['round']}{lap}: {title} is running" + (f"; then {', '.join(after)}" if after else "")
        elif any(v in ("failed", "blocked") for v in states.values()):
            bad = next(s for s in stages if s["status"] in ("failed", "blocked"))
            phase, detail = "blocked", f"{bad['title']}: {bad['message']}"
        elif states["solve"] != "done":
            phase = "new" if states["ingest"] == "pending" else "machine_pending"
            detail = "Machine stages have not all run yet"
        elif pending_answers:
            phase, detail = "answers_pending", f"{pending_answers} answered tasks wait for the next round"
        elif open_tasks:
            phase, detail = "waiting_for_human", f"{open_tasks} human tasks are open"
        elif open_unknowns:
            phase, detail = "machine_pending", f"{open_unknowns} unknowns are open without tasks; run a round"
        elif states["publish"] == "done":
            phase, detail = "published", "Complete labels are published"
        else:
            phase, detail = "ready_to_publish", "No open unknowns; labels can be published"
        actions = []
        for stage in stages:
            if stage["status"] in ("pending", "stale") and stage["name"] != "publish":
                actions.append({"kind": "machine", "text": f"Run {stage['title']} ({stage['status']})",
                                "command": f"data-engine run {self.workspace.dataset_id}"})
                break
        if pending_answers:
            actions.append({"kind": "machine", "text": f"Apply {pending_answers} answers and run the next round",
                            "command": f"data-engine run {self.workspace.dataset_id}"})
        for task in by_type:
            if task["open"]:
                actions.append({"kind": "human", "text": f"Answer {task['open']} '{task['title']}' tasks",
                                "task_type": task["type"], "count": task["open"]})
        if phase == "ready_to_publish":
            actions.append({"kind": "operator", "text": "Publish label sets",
                            "command": f"data-engine publish {self.workspace.dataset_id}"})
        solve = self.db.latest_stage_run("solve", "done")
        events = [dict(row) for row in self.db.query("SELECT * FROM events ORDER BY id DESC LIMIT 60")]
        return {
            "dataset_id": self.workspace.dataset_id, "phase": phase, "phase_detail": detail,
            "round": self.db.current_round(),
            "cameras": [{"camera_id": c.camera_id, "frames": c.frame_count, "fps": round(c.fps, 4),
                         "resolution": [c.width, c.height], "calibration_id": c.calibration_id,
                         "issues": c.issues, "notes": c.notes} for c in (dataset.cameras.values() if dataset else [])],
            "stages": stages, "rounds": self.db.rounds(),
            "unknowns": {"by_kind": by_kind, "open_total": open_unknowns},
            "tasks": {"by_type": by_type, "open_total": open_tasks, "answered_pending": pending_answers},
            "coverage": (solve or {}).get("metrics", {}).get("coverage", {}),
            "next_actions": actions, "events": events, "timeline": timeline,
        }
