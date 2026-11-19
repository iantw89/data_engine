"""Engine configuration: sibling repositories, interpreters, models and the frozen recipe.

Everything that can change a label lives in ``RECIPE`` and is hashed into each stage's
run manifest; changing a value therefore invalidates the stages that consumed it instead
of silently altering already-produced labels.

Site paths (repositories, interpreters, weights, data root) come from defaults that can be
overridden with ``engine.local.json`` next to the repository root or ``$DATA_ENGINE_CONFIG``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_ROOT = REPO_ROOT.parent


def _venv_python(repo: Path) -> Path:
    windows = repo / ".venv" / "Scripts" / "python.exe"
    return windows if windows.exists() else repo / ".venv" / "bin" / "python"


SITE_DEFAULTS: dict[str, Any] = {
    # Where per-dataset workspaces live (state database, detections, manifests, task media).
    "data_root": r"C:\data_engine_data",
    "repos": {
        "data_management": str(GIT_ROOT / "data_management"),
        "cvat_labeling": str(GIT_ROOT / "cvat_labeling"),
        "autofit": str(GIT_ROOT / "Autofit"),
    },
    # Stage workers may need another repository's environment (TensorRT engines, Autofit).
    "interpreters": {
        "engine": str(_venv_python(REPO_ROOT)),
        "data_management": str(_venv_python(GIT_ROOT / "data_management")),
        "autofit": str(_venv_python(GIT_ROOT / "Autofit")),
    },
    "video_root": r"G:\My Drive\FitnessData\Reolink",
    "calibration": r"C:\cvat_labeling_data\20250505071549\pseudo3d_20261031\calibration\v5_tracks_ransac.json",
    "models": {
        "yolo_onnx": str(GIT_ROOT / "data_management" / "models" / "yolov8x-pose-p6-curated-v6-1280.onnx"),
        "vitpose_onnx": str(GIT_ROOT / "data_management" / "models" / "vitpose-curated-v6.onnx"),
        "yolo_pt": str(GIT_ROOT / "cvat_labeling" / "finetuning" / "runs" / "curated_v7_al" / "weights" / "best.pt"),
        "vitpose_hf": r"C:\cvat_labeling_data\20250505071549\pilot\vitpose_curated_v6_20261101\training\best",
    },
    "cvat": {
        "url": "http://localhost:8080",
        "account_file": r"C:\cvat_labeling_data\20250505071549\cvat_admin_credentials.json",
    },
}

# The frozen machine recipe. Values adopted on the golden recording are cited; values that
# are new in the engine are marked NEW and are provisional until validated.
RECIPE: dict[str, Any] = {
    "detect": {
        # cvat_labeling prelabel_policy single-view-v6: YOLO conf .1 / NMS .7, ViTPose per box.
        # The TensorRT path keeps candidates >= .15; the post-filter floor (.2) is above both.
        "imgsz": 1280, "candidate_floor": 0.15, "nms_iou": 0.7,
    },
    "postfilter": {
        # Adopted 2026-11-09 (detection_postprocess.keep_indices).
        "score_floor": 0.2, "pair_iou": 0.3, "pair_containment": 0.6,
        "min_common_joints": 6, "joint_peak": 0.3, "pose_tau": 0.06,
    },
    "tracklets": {
        # Adopted 2026-11-10 association runtime (10 fps BoT-SORT).
        "track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.6,
        "match_thresh": 0.95, "track_buffer": 1800, "fuse_score": True,
        "gmc_method": "none", "proximity_thresh": 0.5, "appearance_thresh": 0.8,
        "with_reid": False, "frame_rate": 10,
        # NEW: tracklets are deliberately over-split at risk events; links re-join them.
        "split_gap_frames": 10, "split_jump_heights": 0.75,
        # NEW: a dropout of up to 5 s is bridged when the box reappears in place (IoU >= .5)
        # and no other detection overlaps the old box (competitor clearance).
        "bridge_gap_frames": 50, "bridge_iou": 0.5, "bridge_competitor_iou": 0.3,
    },
    "geometry": {
        # identity_triangulation contract (06): peak >= .3, >=2 views, ray angle >= 8 deg,
        # reprojection <= 12 px at 1280x960; physical veto floor -30 in.
        "joint_peak": 0.3, "min_views": 2, "min_ray_angle_deg": 8.0,
        "max_reprojection_px": 12.0, "floor_veto_in": -30.0,
        "root_z_range_in": [15.0, 55.0],
        # NEW: bodies triangulated above the cameras or far outside their footprint are implausible.
        "room_margin_in": 150.0,
    },
    "association": {
        # Golden gates (03): geometry-only link >=10 shared observations, p80 <= 5 px /
        # .025 box height, runner-up margin >= .05. NEW: evidence is accumulated per pair.
        "min_shared_frames": 10, "accept_p80_px": 5.0, "accept_p80_rel": 0.025,
        "review_p80_px": 10.0, "review_p80_rel": 0.05, "margin": 0.05, "margin_f2": 0.25,
    },
    "refine": {
        # NEW, adopted 2026-11-18 on the golden bench (see docs/DESIGN.md decision log).
        # A tracked person's measured pose is compared with single-camera pieces up to 1 s away
        # (a stationary person who drops to one view); tracker hand-offs at one spot are merged.
        "temporal_reach_frames": 10, "merge_handoffs": True,
    },
    "temporal": {
        # Golden review filter: endpoint distance <= 12 m/s * elapsed + 0.5 m.
        "max_speed_mps": 12.0, "slack_m": 0.5,
        # NEW: automatic links need a unique continuation within a short gap and tight radius;
        # candidates offered for human review are searched up to candidate_gap_s.
        "auto_gap_s": 5.0, "auto_speed_mps": 2.0, "auto_radius_m": 0.75, "candidate_gap_s": 120.0,
        # NEW, adopted 2026-11-18 on the golden bench: someone measured again at the same spot
        # (within 0.4 m) up to 60 s later is the same person if nobody else was measured within
        # 1 m of the spot in between and no other track starts or ends there.
        "same_spot_gap_s": 60.0, "same_spot_radius_m": 0.4, "same_spot_clear_m": 1.0,
    },
}


def recipe_hash(section: str | None = None) -> str:
    payload = RECIPE if section is None else RECIPE[section]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return result


@dataclass(frozen=True)
class Site:
    values: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    @property
    def data_root(self) -> Path:
        return Path(self.values["data_root"])

    def repo(self, name: str) -> Path:
        return Path(self.values["repos"][name])

    def interpreter(self, name: str) -> Path:
        return Path(self.values["interpreters"][name])

    def model(self, name: str) -> Path:
        return Path(self.values["models"][name])


def load_site() -> Site:
    values = SITE_DEFAULTS
    candidates = [os.environ.get("DATA_ENGINE_CONFIG"), REPO_ROOT / "engine.local.json"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            values = _merge(values, json.loads(Path(candidate).read_text(encoding="utf-8")))
            break
    return Site(values)
