"""Command line: ``data-engine <command>`` (also ``python -m data_engine.cli``).

Exit codes: 0 ok, 1 failure, 2 refused (run lock held, unknown dataset, bad arguments).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

OK, FAILED, REFUSED = 0, 1, 2


# ------------------------------------------------------------------ helpers
def _site():
    from .config import load_site
    return load_site()


def _existing(dataset_id: str):
    """The workspace of an initialised dataset, else None (never creates directories)."""
    from .workspace import Workspace
    site = _site()
    if not (site.data_root / dataset_id / "engine.sqlite3").is_file():
        print(f"unknown dataset {dataset_id!r}: no workspace under {site.data_root} (run `data-engine init {dataset_id}`)",
              file=sys.stderr)
        return None
    return Workspace(dataset_id, site)


def _when(timestamp) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)) if timestamp else "-"


def _duration(seconds) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.1f}min"


def _open_by_kind(status: dict) -> dict[str, int]:
    return {k["kind"]: k["open"] for k in status["unknowns"]["by_kind"] if k["open"]}


# ------------------------------------------------------------------ commands
def cmd_list(args) -> int:
    from .engine import Engine
    from .workspace import list_workspaces
    workspaces = list_workspaces(_site())
    if not workspaces:
        print("no workspaces")
        return OK
    print(f"{'dataset':45s} {'phase':18s} {'round':>5s} {'unknowns':>8s} {'tasks':>5s}")
    for workspace in workspaces:
        try:
            status = Engine(workspace).status()
        except Exception as exc:  # one broken workspace must not hide the others
            print(f"{workspace.dataset_id:45s} unreadable: {type(exc).__name__}: {exc}")
            continue
        print(f"{workspace.dataset_id:45s} {status['phase']:18s} {status['round']:>5d} "
              f"{status['unknowns']['open_total']:>8d} {status['tasks']['open_total']:>5d}")
    return OK


def cmd_init(args) -> int:
    from .dataset import parse_dataset_id
    from .workspace import Workspace
    try:
        parse_dataset_id(args.dataset_id.split("@", 1)[0])
    except ValueError:
        print(f"bad dataset id {args.dataset_id!r}: expected <scene>_<start>_<end>", file=sys.stderr)
        return REFUSED
    workspace = Workspace(args.dataset_id, _site())
    workspace.db.set_meta("created", time.strftime("%Y-%m-%dT%H:%M:%S%z"), overwrite=False)
    workspace.db.set_meta("dataset_id", args.dataset_id, overwrite=False)
    if getattr(args, "limit_frames", None):
        workspace.db.set_meta("limit_frames", int(args.limit_frames))
    print(workspace.root)
    return OK


def summary_text(summary: dict) -> str:
    lines = []
    lines.append("stages run:      " + (", ".join(summary["stages_run"]) or "none (everything up to date)"))
    lines.append(f"answers applied: {summary['answers_applied']}")
    by_kind = summary["open_by_kind"]
    lines.append("open unknowns:   " + (", ".join(f"{k} {v}" for k, v in sorted(by_kind.items())) or "none")
                 + f"  (total {summary['open_total']}, severity {summary['open_severity_total']})")
    shrinking = summary.get("shrinking")
    lines.append("open set shrank: " + ("n/a (no solve this round or first round)" if shrinking is None
                                        else "yes" if shrinking else "NO - more human input or another lever is needed"))
    lines.append(f"open tasks:      {summary['tasks_open']}")
    lines.append("round stopped:   " + (summary["stopped"] or "no (all runnable stages finished)"))
    return "\n".join(lines)


def cmd_run(args) -> int:
    from . import runlock
    from .engine import STAGE, Engine
    workspace = _existing(args.dataset_id)
    if workspace is None:
        return REFUSED
    if not runlock.acquire(workspace, "data-engine " + args.command_line):
        holder = runlock.holder(workspace) or {}
        print(f"refused: a run is already in progress (pid {holder.get('pid')}, started {_when(holder.get('started'))}, "
              f"command: {holder.get('command')})", file=sys.stderr)
        return REFUSED
    try:
        engine = Engine(workspace)
        summary = engine.advance(until=args.until, trigger="manual", force=tuple(args.force or ()))
        print("\n" + summary_text(summary))
        failed = [name for name in summary["stages_run"]
                  if (engine.db.latest_stage_run(name) or {}).get("status") == "failed"]
        return FAILED if failed else OK
    except Exception:
        traceback.print_exc()
        return FAILED
    finally:
        runlock.release(workspace)


def cmd_status(args) -> int:
    from .engine import Engine
    workspace = _existing(args.dataset_id)
    if workspace is None:
        return REFUSED
    status = Engine(workspace).status()
    if args.json:
        print(json.dumps(status, indent=1, default=str))
        return OK
    from . import runlock
    print(f"{status['dataset_id']}")
    print(f"phase: {status['phase']} - {status['phase_detail']}   round: {status['round']}")
    holder = runlock.holder(workspace)
    if holder:
        print(f"run in progress: pid {holder['pid']} since {_when(holder['started'])} ({holder.get('command')})")
    if status["cameras"]:
        print(f"cameras: {len(status['cameras'])}, frames {sum(c['frames'] for c in status['cameras'])}")
    print("\nstages:")
    for stage in status["stages"]:
        message = (stage["message"] or "").replace("\n", " ")
        print(f"  {stage['name']:12s} {stage['status']:8s} round {stage['round'] if stage['round'] is not None else '-':>3} "
              f"{_duration(stage['duration_s']):>8s}  {message[:110]}")
    print("\nopen unknowns: " + (", ".join(f"{k} {v}" for k, v in _open_by_kind(status).items()) or "none"))
    task_types = [t for t in status["tasks"]["by_type"] if t["open"] or t["answered"]]
    print("tasks: " + (", ".join(f"{t['type']} {t['open']} open" + (f"/{t['answered']} answered" if t["answered"] else "")
                                 for t in task_types) or "none"))
    print("\nnext actions:")
    for action in status["next_actions"] or [{"kind": "-", "text": "nothing to do"}]:
        print(f"  [{action['kind']}] {action['text']}" + (f"   $ {action['command']}" if action.get("command") else ""))
    return OK


def cmd_tasks(args) -> int:
    from .taxonomy import TASK_TYPES
    workspace = _existing(args.dataset_id)
    if workspace is None:
        return REFUSED
    status = None if args.status == "all" else args.status
    tasks = workspace.db.tasks(status=status, task_type=args.type, limit=args.limit)
    if not tasks:
        print("no tasks")
        return OK
    print(f"{'id':>5s} {'type':20s} {'prio':>6s}  {'subject':40s} title")
    for task in tasks:
        title = task["payload"].get("title") or (TASK_TYPES[task["type"]].title if task["type"] in TASK_TYPES else "")
        print(f"{task['id']:>5d} {task['type']:20s} {task['priority']:>6.1f}  {task['subject'][:40]:40s} {title}")
    return OK


def cmd_publish(args) -> int:
    from . import runlock
    from .engine import STAGE, Engine
    workspace = _existing(args.dataset_id)
    if workspace is None:
        return REFUSED
    if not runlock.acquire(workspace, "data-engine " + args.command_line):
        print("refused: a run is already in progress", file=sys.stderr)
        return REFUSED
    try:
        engine = Engine(workspace)
        round_number = engine.db.current_round()
        if round_number < 0:
            print("refused: no round has run yet", file=sys.stderr)
            return REFUSED
        result = engine.run_stage(STAGE["publish"], round_number)
        print(f"publish: {result.status}" + (f" - {result.message}" if result.message else ""))
        return OK if result.status == "done" else FAILED
    except Exception:
        traceback.print_exc()
        return FAILED
    finally:
        runlock.release(workspace)


def cmd_serve(args) -> int:
    from .web.app import serve  # lazy: the web module is optional for every other command
    serve(args.host, args.port)
    return OK


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    from .engine import STAGE
    parser = argparse.ArgumentParser(prog="data-engine", description="Data Engine: recursive human-in-the-loop label engine.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list workspaces").set_defaults(func=cmd_list)

    p = sub.add_parser("init", help="create the workspace and database for a dataset (idempotent)")
    p.add_argument("dataset_id", help="<scene>_<start>_<end>, optionally with @<tag> for a pilot slice")
    p.add_argument("--limit-frames", type=int, help="pilot slice: use only the first N frames of every camera")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run one round")
    p.add_argument("dataset_id")
    p.add_argument("--until", choices=list(STAGE), help="stop after this stage")
    p.add_argument("--force", nargs="+", action="extend", choices=list(STAGE), metavar="STAGE",
                   help="rerun these stages even if up to date")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show phase, stages, unknowns, tasks and next actions")
    p.add_argument("dataset_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("tasks", help="list human tasks")
    p.add_argument("dataset_id")
    p.add_argument("--status", default="open", help="open | answered | applied | skipped | superseded | all (default open)")
    p.add_argument("--type")
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("publish", help="run the publish stage")
    p.add_argument("dataset_id")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8600)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: 0 for --help, 2 for bad arguments (already "refused")
        return int(exc.code or 0)
    args.command_line = " ".join(argv if argv is not None else sys.argv[1:])
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return FAILED


if __name__ == "__main__":
    sys.exit(main())
