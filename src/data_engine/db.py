"""Per-dataset state database (SQLite).

The database holds everything small and relational: which stages ran, what is still
unknown, which human tasks exist and what was answered, and the constraints derived from
answers. Large arrays (detections, tracklets, 3D tracks) are artifact files referenced by
stage runs.

Vocabulary
- stage run   one execution of a stage in a round; has a manifest with input/config hashes.
- unknown     one thing the engine cannot currently decide, with what would resolve it.
              Unknowns are the measure of progress: every round must shrink the open set.
- task        a question for a human; resolves one or more unknowns.
- constraint  durable evidence (human answers, or machine decisions worth pinning) that
              later runs must respect. Endpoints are *detection keys*, never tracklet or
              track numbers, because those are renumbered whenever a stage is rerun.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS rounds (
    round INTEGER PRIMARY KEY,
    started REAL NOT NULL,
    finished REAL,
    trigger TEXT NOT NULL,          -- 'init' | 'human_answers' | 'manual' | 'retrain'
    summary TEXT NOT NULL DEFAULT '{}'  -- JSON: open unknowns by kind at round end, coverage
);

CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    round INTEGER NOT NULL,
    status TEXT NOT NULL,           -- 'running' | 'done' | 'failed' | 'blocked' | 'skipped'
    started REAL NOT NULL,
    finished REAL,
    input_hash TEXT,                -- hash of recipe section + upstream outputs + constraints used
    manifest TEXT,                  -- path of run_manifest.json relative to the workspace
    metrics TEXT NOT NULL DEFAULT '{}',
    message TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS stage_runs_stage ON stage_runs(stage, id);

CREATE TABLE IF NOT EXISTS unknowns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- see data_engine.unknowns.KINDS
    subject TEXT NOT NULL,          -- stable key of what is unknown (detection-key based)
    stage TEXT NOT NULL,            -- stage that reported it
    status TEXT NOT NULL,           -- 'open' | 'resolved' | 'undeterminable'
    severity REAL NOT NULL DEFAULT 1.0,  -- how much label volume depends on it (observations)
    detail TEXT NOT NULL DEFAULT '{}',   -- JSON evidence: why it is unknown, what would resolve it
    opened_round INTEGER NOT NULL,
    closed_round INTEGER,
    resolution TEXT,                -- 'machine:<rule>' | 'human:<task id>' | 'undeterminable:<reason>'
    UNIQUE(kind, subject)
);
CREATE INDEX IF NOT EXISTS unknowns_status ON unknowns(status, kind);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,             -- see data_engine.tasks.TASK_TYPES
    status TEXT NOT NULL,           -- 'open' | 'answered' | 'applied' | 'superseded' | 'skipped'
    priority REAL NOT NULL DEFAULT 0,
    round INTEGER NOT NULL,
    subject TEXT NOT NULL,          -- dedupe key; an open task is never duplicated
    payload TEXT NOT NULL,          -- JSON: everything the UI needs to render the question
    answer TEXT,                    -- JSON answer as given
    answered_by TEXT,
    created REAL NOT NULL,
    answered REAL,
    applied_round INTEGER,
    external TEXT                   -- JSON: e.g. {"cvat_task": 312, "cvat_job": 540}
);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status, priority);
CREATE UNIQUE INDEX IF NOT EXISTS tasks_subject_open ON tasks(type, subject) WHERE status IN ('open', 'answered');

CREATE TABLE IF NOT EXISTS task_unknowns (
    task_id INTEGER NOT NULL REFERENCES tasks(id),
    unknown_id INTEGER NOT NULL REFERENCES unknowns(id),
    PRIMARY KEY (task_id, unknown_id)
);

CREATE TABLE IF NOT EXISTS constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,             -- 'same' | 'different' | 'split' | 'not_person' | 'duplicate'
                                    -- | 'alignment' | 'undeterminable'
    a TEXT NOT NULL,                -- JSON list of detection keys (or a camera id for alignment)
    b TEXT NOT NULL DEFAULT '[]',   -- JSON list of detection keys
    source TEXT NOT NULL,           -- 'human' | 'machine:<rule>'
    task_id INTEGER REFERENCES tasks(id),
    detail TEXT NOT NULL DEFAULT '{}',
    round INTEGER NOT NULL,
    created REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,   -- superseded constraints stay for audit, inactive
    superseded_by INTEGER REFERENCES constraints(id)
);
CREATE INDEX IF NOT EXISTS constraints_kind ON constraints(kind, active);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time REAL NOT NULL,
    level TEXT NOT NULL,            -- 'info' | 'warning' | 'error'
    stage TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL
);
"""


def _loads(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION, overwrite=False)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self):
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def query(self, sql: str, parameters: Iterable = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, tuple(parameters)).fetchall()

    def execute(self, sql: str, parameters: Iterable = ()) -> sqlite3.Cursor:
        cursor = self.connection.execute(sql, tuple(parameters))
        self.connection.commit()
        return cursor

    # ------------------------------------------------------------------ meta / events
    def set_meta(self, key: str, value: Any, overwrite: bool = True) -> None:
        verb = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
        self.connection.execute(f"{verb} INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self.connection.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM meta WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def log(self, message: str, *, level: str = "info", stage: str = "") -> None:
        self.execute("INSERT INTO events(time, level, stage, message) VALUES (?, ?, ?, ?)",
                     (time.time(), level, stage, message))

    # ------------------------------------------------------------------ rounds
    def current_round(self) -> int:
        row = self.query("SELECT MAX(round) AS r FROM rounds")[0]
        return -1 if row["r"] is None else int(row["r"])

    def open_round(self, trigger: str) -> int:
        number = self.current_round() + 1
        self.execute("INSERT INTO rounds(round, started, trigger) VALUES (?, ?, ?)", (number, time.time(), trigger))
        return number

    def close_round(self, number: int, summary: dict) -> None:
        self.execute("UPDATE rounds SET finished = ?, summary = ? WHERE round = ?",
                     (time.time(), json.dumps(summary), number))

    def rounds(self) -> list[dict]:
        return [{**dict(row), "summary": _loads(row["summary"], {})}
                for row in self.query("SELECT * FROM rounds ORDER BY round")]

    # ------------------------------------------------------------------ stage runs
    def start_stage(self, stage: str, round_number: int, input_hash: str | None) -> int:
        cursor = self.execute(
            "INSERT INTO stage_runs(stage, round, status, started, input_hash) VALUES (?, ?, 'running', ?, ?)",
            (stage, round_number, time.time(), input_hash))
        return int(cursor.lastrowid)

    def finish_stage(self, run_id: int, status: str, *, metrics: dict | None = None,
                     message: str = "", manifest: str | None = None) -> None:
        self.execute(
            "UPDATE stage_runs SET status = ?, finished = ?, metrics = ?, message = ?, manifest = ? WHERE id = ?",
            (status, time.time(), json.dumps(metrics or {}), message, manifest, run_id))

    def latest_stage_run(self, stage: str, status: str | None = None) -> dict | None:
        sql = "SELECT * FROM stage_runs WHERE stage = ?" + (" AND status = ?" if status else "")
        rows = self.query(sql + " ORDER BY id DESC LIMIT 1", (stage, status) if status else (stage,))
        return {**dict(rows[0]), "metrics": _loads(rows[0]["metrics"], {})} if rows else None

    def stage_runs(self) -> list[dict]:
        return [{**dict(row), "metrics": _loads(row["metrics"], {})}
                for row in self.query("SELECT * FROM stage_runs ORDER BY id")]

    # ------------------------------------------------------------------ unknowns
    def report_unknowns(self, stage: str, kind: str, items: Iterable[dict], round_number: int) -> dict:
        """Replace the open set of ``kind`` reported by ``stage`` with ``items``.

        Each item: {"subject": str, "severity": float, "detail": dict}. Unknowns that were
        open and are no longer reported are closed as machine-resolved; unknowns already
        closed by a human or marked undeterminable are never reopened by a machine report.
        """
        unique: dict[str, dict] = {}
        for item in items:  # one row per subject (the table's key): of two reports the more severe is kept
            known = unique.get(item["subject"])
            if known is None or float(item.get("severity", 1.0)) > float(known.get("severity", 1.0)):
                unique[item["subject"]] = item
        items = list(unique.values())
        subjects = set(unique)
        opened = closed = kept = 0
        with self.transaction() as connection:
            existing = {row["subject"]: row for row in connection.execute(
                "SELECT id, subject, status, resolution FROM unknowns WHERE kind = ?", (kind,))}
            for item in items:
                row = existing.get(item["subject"])
                detail = json.dumps(item.get("detail", {}))
                severity = float(item.get("severity", 1.0))
                if row is None:
                    connection.execute(
                        "INSERT INTO unknowns(kind, subject, stage, status, severity, detail, opened_round) "
                        "VALUES (?, ?, ?, 'open', ?, ?, ?)",
                        (kind, item["subject"], stage, severity, detail, round_number))
                    opened += 1
                elif row["status"] == "open":
                    connection.execute("UPDATE unknowns SET severity = ?, detail = ?, stage = ? WHERE id = ?",
                                       (severity, detail, stage, row["id"]))
                    kept += 1
                elif row["status"] == "resolved" and str(row["resolution"]).startswith("machine:"):
                    # A machine resolution that no longer holds is honestly reopened.
                    connection.execute(
                        "UPDATE unknowns SET status = 'open', severity = ?, detail = ?, closed_round = NULL, "
                        "resolution = NULL WHERE id = ?", (severity, detail, row["id"]))
                    opened += 1
            for subject, row in existing.items():
                if subject not in subjects and row["status"] == "open":
                    connection.execute(
                        "UPDATE unknowns SET status = 'resolved', closed_round = ?, resolution = ? WHERE id = ?",
                        (round_number, f"machine:{stage}", row["id"]))
                    closed += 1
        return {"opened": opened, "closed": closed, "kept": kept}

    def close_unknown(self, unknown_id: int, round_number: int, resolution: str,
                      status: str = "resolved") -> None:
        self.execute("UPDATE unknowns SET status = ?, closed_round = ?, resolution = ? WHERE id = ?",
                     (status, round_number, resolution, unknown_id))

    def unknowns(self, status: str | None = "open", kind: str | None = None) -> list[dict]:
        clauses, parameters = [], []
        if status:
            clauses.append("status = ?"), parameters.append(status)
        if kind:
            clauses.append("kind = ?"), parameters.append(kind)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return [{**dict(row), "detail": _loads(row["detail"], {})}
                for row in self.query(f"SELECT * FROM unknowns{where} ORDER BY severity DESC, id", parameters)]

    def unknown_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for row in self.query("SELECT kind, status, COUNT(*) AS n, SUM(severity) AS s FROM unknowns GROUP BY kind, status"):
            counts.setdefault(row["kind"], {})[row["status"]] = int(row["n"])
            counts[row["kind"]][row["status"] + "_severity"] = float(row["s"] or 0)
        return counts

    # ------------------------------------------------------------------ tasks
    def add_task(self, task_type: str, subject: str, payload: dict, *, priority: float,
                 round_number: int, unknown_ids: Iterable[int] = ()) -> int | None:
        """Create an open task unless an open/answered one with the same subject exists."""
        with self.transaction() as connection:
            if connection.execute(
                    "SELECT 1 FROM tasks WHERE type = ? AND subject = ? AND status IN ('open', 'answered')",
                    (task_type, subject)).fetchone():
                return None
            cursor = connection.execute(
                "INSERT INTO tasks(type, status, priority, round, subject, payload, created) "
                "VALUES (?, 'open', ?, ?, ?, ?, ?)",
                (task_type, priority, round_number, subject, json.dumps(payload), time.time()))
            task_id = int(cursor.lastrowid)
            connection.executemany("INSERT OR IGNORE INTO task_unknowns(task_id, unknown_id) VALUES (?, ?)",
                                   [(task_id, int(u)) for u in unknown_ids])
        return task_id

    def task(self, task_id: int) -> dict | None:
        rows = self.query("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._task(rows[0]) if rows else None

    @staticmethod
    def _task(row: sqlite3.Row) -> dict:
        return {**dict(row), "payload": _loads(row["payload"], {}), "answer": _loads(row["answer"]),
                "external": _loads(row["external"])}

    def tasks(self, status: str | None = None, task_type: str | None = None, limit: int | None = None) -> list[dict]:
        clauses, parameters = [], []
        if status:
            clauses.append("status = ?"), parameters.append(status)
        if task_type:
            clauses.append("type = ?"), parameters.append(task_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        tail = f" LIMIT {int(limit)}" if limit else ""
        return [self._task(row) for row in
                self.query(f"SELECT * FROM tasks{where} ORDER BY priority DESC, id{tail}", parameters)]

    def task_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for row in self.query("SELECT type, status, COUNT(*) AS n FROM tasks GROUP BY type, status"):
            counts.setdefault(row["type"], {})[row["status"]] = int(row["n"])
        return counts

    def answer_task(self, task_id: int, answer: dict, answered_by: str) -> None:
        self.execute("UPDATE tasks SET status = 'answered', answer = ?, answered_by = ?, answered = ? "
                     "WHERE id = ? AND status IN ('open', 'answered')",
                     (json.dumps(answer), answered_by, time.time(), task_id))

    def set_task_status(self, task_id: int, status: str, applied_round: int | None = None) -> None:
        self.execute("UPDATE tasks SET status = ?, applied_round = COALESCE(?, applied_round) WHERE id = ?",
                     (status, applied_round, task_id))

    def task_unknown_ids(self, task_id: int) -> list[int]:
        return [int(row["unknown_id"]) for row in
                self.query("SELECT unknown_id FROM task_unknowns WHERE task_id = ?", (task_id,))]

    # ------------------------------------------------------------------ constraints
    def add_constraint(self, kind: str, a: list, b: list | None, *, source: str, round_number: int,
                       task_id: int | None = None, detail: dict | None = None) -> int:
        cursor = self.execute(
            "INSERT INTO constraints(kind, a, b, source, task_id, detail, round, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, json.dumps(a), json.dumps(b or []), source, task_id, json.dumps(detail or {}),
             round_number, time.time()))
        return int(cursor.lastrowid)

    def constraints(self, kind: str | None = None, active_only: bool = True) -> list[dict]:
        clauses, parameters = [], []
        if kind:
            clauses.append("kind = ?"), parameters.append(kind)
        if active_only:
            clauses.append("active = 1")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return [{**dict(row), "a": json.loads(row["a"]), "b": json.loads(row["b"]),
                 "detail": _loads(row["detail"], {})}
                for row in self.query(f"SELECT * FROM constraints{where} ORDER BY id", parameters)]

    def supersede_constraint(self, constraint_id: int, by: int) -> None:
        """Human evidence is never deleted; a later explicit answer supersedes it with provenance."""
        self.execute("UPDATE constraints SET active = 0, superseded_by = ? WHERE id = ?", (by, constraint_id))
