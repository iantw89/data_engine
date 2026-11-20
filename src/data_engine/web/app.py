"""Web application: JSON API plus the static dashboard / task-runner front end.

Every request opens its own ``Workspace`` (and so its own SQLite connection); only the
``FrameSource`` and detection cache used for rendering panels are shared per dataset.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import Site, load_site
from ..engine import STAGES, Engine
from ..render import error_image, render_panel
from ..taxonomy import ANSWER_MODEL, IMAGE_FLAGS, KINDS, TASK_TYPES, UI_KINDS
from ..workspace import Workspace, list_workspaces
from .validate import AnswerError, validate_answer

STATIC = Path(__file__).parent / "static"
SRC_ROOT = Path(__file__).resolve().parents[2]
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_advance_lock = threading.Lock()


# --------------------------------------------------------------------------- processes
def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def spawn_process(command: list[str], log_path: Path, env: dict, cwd: Path) -> int:
    """Start ``command`` detached from this server with output appended to ``log_path``; return its pid."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env=env, cwd=str(cwd), close_fds=True, **kwargs)
    return process.pid


def _tail(path: Path, lines: int = 60, max_bytes: int = 24000) -> str:
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            data = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


def run_state(workspace: Workspace, *, tail: bool = False) -> dict:
    """Is a round in progress? By the detached process's lock file and by any stage 'running'."""
    logs = workspace.path("logs")
    info: dict[str, Any] = {"running": False, "reason": None, "pid": None, "started": None, "log": None,
                            "stages_running": []}
    # The CLI owns logs/run.lock (data_engine.runlock). A run started from this server also
    # leaves logs/web_run.json with its pid and log file, which covers the moment between
    # spawning the process and it taking the lock.
    from ..runlock import holder
    held = holder(workspace)
    if held:
        info.update(running=True, reason="process", pid=held.get("pid"), started=held.get("started"))
    marker = logs / "web_run.json"
    if marker.is_file():
        try:
            spawned = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            spawned = {}
        info["log"] = spawned.get("log")
        if spawned.get("pid") and pid_alive(int(spawned["pid"])):
            info.update(running=True, reason="process", pid=info["pid"] or spawned["pid"],
                        started=info["started"] or spawned.get("started"))
    for stage in STAGES:
        run = workspace.db.latest_stage_run(stage.name)
        if run and run["status"] == "running":
            info["stages_running"].append(stage.name)
    if info["stages_running"] and not info["running"]:
        info.update(running=True, reason="stage")
    if tail:
        log_path = (workspace.root / info["log"]) if info["log"] else None
        if log_path is None or not log_path.is_file():
            candidates = sorted(logs.glob("run_*.log")) if logs.is_dir() else []
            log_path = candidates[-1] if candidates else None
        info["log"] = str(log_path.relative_to(workspace.root)) if log_path else None
        info["tail"] = _tail(log_path) if log_path else ""
    return info


# --------------------------------------------------------------------------- render registry
class RenderRegistry:
    """One long-lived Workspace per dataset, used only for rendering (shared FrameSource and detections)."""

    def __init__(self, site: Site):
        self.site = site
        self._workspaces: dict[str, Workspace] = {}
        self._signature: dict[str, tuple[tuple, float]] = {}
        self._lock = threading.Lock()

    def _detections_signature(self, workspace: Workspace) -> tuple:
        files = list(workspace.detections_dir.glob("*/*.npz")) if workspace.detections_dir.is_dir() else []
        return len(files), max((f.stat().st_mtime_ns for f in files), default=0)

    def get(self, dataset_id: str) -> Workspace:
        with self._lock:
            workspace = self._workspaces.get(dataset_id)
            if workspace is None:
                workspace = self._workspaces[dataset_id] = Workspace(dataset_id, self.site)
                self._signature[dataset_id] = (self._detections_signature(workspace), time.time())
            else:
                signature, checked = self._signature[dataset_id]
                if time.time() - checked > 10:  # pick up detection shards written while the server runs
                    current = self._detections_signature(workspace)
                    if current != signature:
                        workspace.__dict__.pop("detections", None)
                    self._signature[dataset_id] = (current, time.time())
            return workspace

    def close(self) -> None:
        with self._lock:
            for workspace in self._workspaces.values():
                frames = workspace.__dict__.get("frames")
                if frames is not None:
                    frames.close()
                if "db" in workspace.__dict__:
                    workspace.db.close()
            self._workspaces.clear()


# --------------------------------------------------------------------------- helpers
def _task_summary(task: dict) -> dict:
    payload = task["payload"]
    return {"id": task["id"], "type": task["type"], "status": task["status"], "priority": task["priority"],
            "round": task["round"], "subject": task["subject"], "title": payload.get("title"),
            "ui": _task_ui(task),
            "time_seconds": payload.get("time_seconds"), "created": task["created"],
            "answered": task["answered"], "answered_by": task["answered_by"], "applied_round": task["applied_round"]}


def _task_ui(task: dict) -> str:
    ui = task["payload"].get("ui")
    if not ui and task["type"] in TASK_TYPES:
        ui = TASK_TYPES[task["type"]].ui
    return ui or ""


def _task_detail(workspace: Workspace, task: dict) -> dict:
    info = TASK_TYPES.get(task["type"])
    return {**task, "ui": _task_ui(task),
            "type_info": {"title": info.title, "question": info.question, "ui": info.ui} if info else None,
            "unknown_ids": workspace.db.task_unknown_ids(task["id"])}


def _open_by_type(workspace: Workspace) -> dict[str, int]:
    return {row["type"]: int(row["n"]) for row in workspace.db.query(
        "SELECT type, COUNT(*) AS n FROM tasks WHERE status = 'open' GROUP BY type")}


# --------------------------------------------------------------------------- app
def create_app(site: Site | None = None) -> FastAPI:
    site = site or load_site()
    registry = RenderRegistry(site)
    @asynccontextmanager
    async def lifespan(_app):
        yield
        registry.close()

    app = FastAPI(title="Data Engine", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.site, app.state.registry = site, registry

    def workspace_dep(dataset_id: str):
        if not ID_PATTERN.match(dataset_id) or not (site.data_root / dataset_id / "engine.sqlite3").is_file():
            raise HTTPException(404, f"unknown dataset '{dataset_id}'")
        workspace = Workspace(dataset_id, site)
        try:
            yield workspace
        finally:
            if "db" in workspace.__dict__:
                workspace.db.close()

    WS = Depends(workspace_dep)

    # ------------------------------------------------------------------ datasets / status
    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        rows = []
        for workspace in list_workspaces(site):
            try:
                status = Engine(workspace).status()
                rows.append({"id": workspace.dataset_id, "phase": status["phase"],
                             "phase_detail": status["phase_detail"], "round": status["round"],
                             "open_unknowns": status["unknowns"]["open_total"],
                             "open_tasks": status["tasks"]["open_total"],
                             "answered_pending": status["tasks"]["answered_pending"],
                             "cameras": len(status["cameras"])})
            except Exception as exc:  # one broken workspace must not hide the others
                rows.append({"id": workspace.dataset_id, "phase": "error", "phase_detail": f"{type(exc).__name__}: {exc}",
                             "round": None, "open_unknowns": None, "open_tasks": None, "answered_pending": None,
                             "cameras": None})
            finally:
                if "db" in workspace.__dict__:
                    workspace.db.close()
        return rows

    @app.get("/api/datasets/{dataset_id}/status")
    def status(dataset_id: str, workspace: Workspace = WS) -> dict:
        result = Engine(workspace).status()
        result["run"] = run_state(workspace)
        result["server_time"] = time.time()
        return result

    @app.get("/api/taxonomy")
    def taxonomy() -> dict:
        return {"kinds": [asdict(k) for k in KINDS.values()], "task_types": [asdict(t) for t in TASK_TYPES.values()],
                "ui_kinds": list(UI_KINDS), "answer_model": ANSWER_MODEL,
                "image_flags": {name: {"label": label, "meaning": meaning, "produces": produces}
                                for name, (label, meaning, produces) in IMAGE_FLAGS.items()}}

    # ------------------------------------------------------------------ tasks
    @app.get("/api/datasets/{dataset_id}/tasks")
    def list_tasks(dataset_id: str, status: str | None = None, type: str | None = None,
                   limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                   workspace: Workspace = WS) -> dict:
        clauses, parameters = [], []
        if status:
            clauses.append("status = ?"), parameters.append(status)
        if type:
            clauses.append("type = ?"), parameters.append(type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        total = workspace.db.query(f"SELECT COUNT(*) AS n FROM tasks{where}", parameters)[0]["n"]
        rows = workspace.db.query(f"SELECT id FROM tasks{where} ORDER BY priority DESC, id LIMIT ? OFFSET ?",
                                  [*parameters, limit, offset])
        tasks = [_task_summary(workspace.db.task(row["id"])) for row in rows]
        return {"total": total, "offset": offset, "limit": limit, "tasks": tasks}

    @app.get("/api/datasets/{dataset_id}/tasks/next")
    def next_task(dataset_id: str, type: str | None = None, after: int | None = None,
                  workspace: Workspace = WS) -> dict:
        db = workspace.db
        parameters: list[Any] = []
        sql = "SELECT id, priority FROM tasks WHERE status = 'open'"
        if type:
            sql += " AND type = ?"
            parameters.append(type)
        rows = db.query(sql + " ORDER BY priority DESC, id", parameters)
        chosen = None
        if rows:
            ids = [row["id"] for row in rows]
            if after is not None:
                anchor = db.query("SELECT priority FROM tasks WHERE id = ?", (after,))
                if after in ids:
                    following = ids[ids.index(after) + 1:]
                elif anchor:  # the anchor was answered/skipped: continue from its place in the order
                    key = (-anchor[0]["priority"], after)
                    following = [row["id"] for row in rows if (-row["priority"], row["id"]) > key]
                else:
                    following = ids
                chosen = following[0] if following else next((i for i in ids if i != after), None)
            else:
                chosen = ids[0]
        task = _task_detail(workspace, db.task(chosen)) if chosen is not None else None
        return {"task": task, "remaining": len(rows), "remaining_by_type": _open_by_type(workspace)}

    @app.get("/api/datasets/{dataset_id}/tasks/{task_id}")
    def get_task(dataset_id: str, task_id: int, workspace: Workspace = WS) -> dict:
        task = workspace.db.task(task_id)
        if task is None:
            raise HTTPException(404, f"no task {task_id}")
        return _task_detail(workspace, task)

    @app.post("/api/datasets/{dataset_id}/tasks/{task_id}/answer")
    def answer_task(dataset_id: str, task_id: int, body: dict = Body(...), workspace: Workspace = WS) -> dict:
        db = workspace.db
        task = db.task(task_id)
        if task is None:
            raise HTTPException(404, f"no task {task_id}")
        if task["status"] not in ("open", "answered"):
            raise HTTPException(409, f"task {task_id} is '{task['status']}' and can no longer be answered")
        user = str(body.get("user") or "web").strip()[:80] or "web"
        try:
            cleaned = validate_answer(_task_ui(task), task["payload"], body.get("answer"))
        except AnswerError as exc:
            raise HTTPException(422, str(exc)) from exc
        db.answer_task(task_id, cleaned, user)
        return {"task": _task_detail(workspace, db.task(task_id)), "remaining_by_type": _open_by_type(workspace)}

    @app.post("/api/datasets/{dataset_id}/tasks/{task_id}/skip")
    def skip_task(dataset_id: str, task_id: int, workspace: Workspace = WS) -> dict:
        task = workspace.db.task(task_id)
        if task is None:
            raise HTTPException(404, f"no task {task_id}")
        if task["status"] != "open":
            raise HTTPException(409, f"task {task_id} is '{task['status']}', only open tasks can be skipped")
        workspace.db.set_task_status(task_id, "skipped")
        return {"task": _task_detail(workspace, workspace.db.task(task_id)),
                "remaining_by_type": _open_by_type(workspace)}

    @app.post("/api/datasets/{dataset_id}/tasks/{task_id}/reopen")
    def reopen_task(dataset_id: str, task_id: int, body: dict | None = Body(default=None),
                    workspace: Workspace = WS) -> dict:
        db = workspace.db
        task = db.task(task_id)
        if task is None:
            raise HTTPException(404, f"no task {task_id}")
        if task["status"] not in ("answered", "skipped"):
            raise HTTPException(409, f"task {task_id} is '{task['status']}'; only answered or skipped tasks "
                                     "can be reopened (applied answers are already constraints)")
        try:
            cursor = db.execute(
                "UPDATE tasks SET status = 'open', answer = NULL, answered_by = NULL, answered = NULL "
                "WHERE id = ? AND status IN ('answered', 'skipped')", (task_id,))
        except Exception as exc:  # a same-subject task was opened meanwhile (unique index)
            raise HTTPException(409, f"cannot reopen: {exc}") from exc
        if cursor.rowcount:
            user = str((body or {}).get("user") or "web")[:80]
            db.log(f"task {task_id} reopened by {user} (was {task['status']})", stage="tasks")
        return {"task": _task_detail(workspace, db.task(task_id)), "remaining_by_type": _open_by_type(workspace)}

    # ------------------------------------------------------------------ panels
    @app.get("/api/datasets/{dataset_id}/panel")
    def panel(dataset_id: str, spec: str, full: int = 0, w: int = Query(720, ge=64, le=4096),
              _workspace: Workspace = WS) -> Response:
        try:
            parsed = json.loads(spec)
            if not isinstance(parsed, dict) or "camera" not in parsed or "frame" not in parsed:
                raise ValueError("spec needs camera and frame")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(422, f"bad panel spec: {exc}") from exc
        try:
            data = render_panel(registry.get(dataset_id), parsed, max_side=w, full=bool(full))
        except Exception as exc:
            # A placeholder (not a 5xx) keeps the review screen usable; never cached.
            return Response(error_image(f"{type(exc).__name__}: {exc}"), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store", "X-Render-Error": type(exc).__name__})
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------------ unknowns / constraints
    @app.get("/api/datasets/{dataset_id}/unknowns")
    def unknowns(dataset_id: str, status: str | None = "open", kind: str | None = None,
                 limit: int = Query(200, ge=1, le=5000), workspace: Workspace = WS) -> dict:
        rows = workspace.db.unknowns(None if status in (None, "", "all") else status, kind or None)
        return {"total": len(rows), "unknowns": rows[:limit]}

    @app.get("/api/datasets/{dataset_id}/constraints")
    def constraints(dataset_id: str, kind: str | None = None, active_only: bool = True,
                    limit: int = Query(500, ge=1, le=10000), workspace: Workspace = WS) -> dict:
        rows = workspace.db.constraints(kind or None, active_only)
        return {"total": len(rows), "constraints": rows[-limit:]}

    # ------------------------------------------------------------------ running rounds
    @app.post("/api/datasets/{dataset_id}/advance")
    def advance(dataset_id: str, workspace: Workspace = WS) -> dict:
        with _advance_lock:
            state = run_state(workspace)
            if state["running"]:
                what = (f"process {state['pid']}" if state["reason"] == "process"
                        else f"stage(s) {', '.join(state['stages_running'])} marked running")
                raise HTTPException(409, f"a run is already in progress ({what})")
            logs = workspace.path("logs")
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_ROOT), os.environ.get("PYTHONPATH", "")]))}
            pid = spawn_process([sys.executable, "-m", "data_engine.cli", "run", dataset_id], log_path, env,
                                SRC_ROOT.parent)
            started = time.time()
            (logs / "web_run.json").write_text(json.dumps(
                {"pid": pid, "started": started, "log": str(log_path.relative_to(workspace.root))}), encoding="utf-8")
            workspace.db.log(f"run started from the web UI (pid {pid})", stage="web")
        return {"started": True, "pid": pid, "log": str(log_path.relative_to(workspace.root)), "started_at": started}

    @app.get("/api/datasets/{dataset_id}/run")
    def run(dataset_id: str, workspace: Workspace = WS) -> dict:
        return run_state(workspace, tail=True)

    # ------------------------------------------------------------------ pages and static files
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return page()

    @app.get("/d/{dataset_id}", include_in_schema=False)
    def dataset_page(dataset_id: str) -> FileResponse:
        return page()

    @app.get("/d/{dataset_id}/tasks", include_in_schema=False)
    def tasks_page(dataset_id: str) -> FileResponse:
        return page()

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC), check_dir=False), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8610, site: Site | None = None) -> None:
    import uvicorn
    uvicorn.run(create_app(site), host=host, port=port, log_level="warning")
