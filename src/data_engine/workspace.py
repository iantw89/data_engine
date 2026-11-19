"""A dataset's working directory: state database, immutable inputs and per-round artifacts.

Layout (under ``<data_root>/<dataset_id>/``)
    dataset.json                 resolved cameras/files (ingest)
    engine.sqlite3               state machine, unknowns, tasks, constraints
    detections/raw/<camera>/     immutable detection shards (detect)
    calibration/                 calibration used for this dataset + mount check
    alignment/                   alignment.json, sync_table.npz, clocks.npz, report.json
    tracklets/                   per-camera tracklets
    rounds/r<N>/                 association / identity / 3D artifacts of round N
    manifests/                   one run manifest per stage run
    frame_index/                 per-file frame time caches
    published/                   label drafts handed to data_management
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from .config import Site, load_site
from .dataset import Dataset
from .db import Database
from .store import DetectionStore


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Workspace:
    def __init__(self, dataset_id: str, site: Site | None = None, *, root: Path | None = None):
        # A workspace id is "<dataset id>" or "<dataset id>@<tag>" for a pilot slice of the
        # same recording (e.g. its first minute); source_dataset is the catalogue dataset.
        self.dataset_id = dataset_id
        self.source_dataset = dataset_id.split("@", 1)[0]
        self.site = site or load_site()
        self.root = Path(root) if root is not None else self.site.data_root / dataset_id
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def dataset_json(self) -> Path:
        return self.path("dataset.json")

    @property
    def detections_dir(self) -> Path:
        return self.path("detections", "raw")

    def round_dir(self, round_number: int) -> Path:
        directory = self.path("rounds", f"r{round_number:03d}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def latest_round_file(self, name: str) -> Path | None:
        """Newest round directory that contains ``name`` (artifacts persist across rounds)."""
        rounds = sorted(self.path("rounds").glob("r[0-9]*"), reverse=True) if self.path("rounds").is_dir() else []
        for directory in rounds:
            if (directory / name).is_file():
                return directory / name
        return None

    # ------------------------------------------------------------------ lazily opened resources
    @cached_property
    def db(self) -> Database:
        return Database(self.path("engine.sqlite3"))

    @cached_property
    def dataset(self) -> Dataset:
        return Dataset.load(self.dataset_json)

    @cached_property
    def detections(self) -> DetectionStore:
        return DetectionStore(self.detections_dir, self.dataset)

    @cached_property
    def frames(self):
        from .video import FrameSource
        return FrameSource(self.dataset, self.path("frame_index"))

    @property
    def exists(self) -> bool:
        return self.path("engine.sqlite3").is_file()

    # ------------------------------------------------------------------ manifests
    def write_manifest(self, stage: str, round_number: int, *, config: dict, inputs: dict,
                       outputs: dict, status: str, notes: str = "") -> str:
        """Record what produced a stage's outputs: recipe, input and output hashes, code identity.

        Mirrors cvat_labeling's run_manifest rule: every generation run is reproducible
        from its manifest and a changed input or recipe is visible as a changed hash.
        """
        from .provenance import code_identity
        manifest = {
            "dataset": self.dataset_id, "stage": stage, "round": round_number, "status": status,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": config, "inputs": inputs,
            "outputs": outputs, "code": code_identity(), "notes": notes,
            "human_input_policy": "human answers enter only as recorded constraints (see constraints table)",
        }
        directory = self.path("manifests")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{stage}_r{round_number:03d}_{int(time.time())}.json"
        target.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        return str(target.relative_to(self.root))


def list_workspaces(site: Site | None = None) -> list[Workspace]:
    site = site or load_site()
    if not site.data_root.is_dir():
        return []
    return [Workspace(path.name, site) for path in sorted(site.data_root.iterdir())
            if (path / "engine.sqlite3").is_file()]
