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
        "fast_reid": str(GIT_ROOT / "fast-reid"),  # optional: appearance cue of the swap check
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
        "reid_fastreid": str(GIT_ROOT / "fast-reid" / "pretrained_download" / "msmt_sbs_R50.pth"),
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
        # NEW 2026-11-19: a hand-off that position alone does not single out (a crowd, or someone
        # walking) is merged when single-camera tracklets in at least this many cameras continue
        # from the ending track into the starting one and into no other candidate.
        "handoff_tracklets": 2,
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
    "swap_check": {
        # NEW 2026-11-18 (solve/swap_check.py, experiments/switches): did two tracks exchange people?
        # Pairs ever within radius_m are examined step by step. A step is unambiguous when
        # exchanging the labels would cost margin_m more movement than keeping them (more per
        # second without a common measurement); it is an "exchange" when keeping them costs
        # exchange_m more. Steps in between are crossings, merged over merge_frames.
        "radius_m": 1.0, "median_frames": 5, "margin_m": 0.3, "margin_per_second_m": 0.5, "exchange_m": 0.3,
        "step_gap_frames": 10, "max_gap_frames": 100, "merge_frames": 20, "window_frames": 100,
        # Two identities whose common median distance is below duplicate_m are one person tracked
        # twice; a track ending within ending_radius_m of a continuing one is checked one-sided.
        # The survivor must have been measured established_frames in the window before the ending.
        "duplicate_m": 0.3, "ending_radius_m": 0.5, "established_frames": 30,
        # A cue says "swap" above flag, "stay" below stay. Units: heading m/s, body within-person
        # sigma, appearance cosine. Body and appearance were fitted on the golden recording
        # (flag = 98th percentile of untouched encounters, stay = 2nd percentile of injected
        # swaps); heading is a round number. All provisional.
        "flag": {"heading": 0.6, "body": 0.038, "appearance": -0.089},
        "stay": {"heading": -0.6, "body": -0.505, "appearance": 0.043},
        "tracklet_contradictions": 1, "tracklet_agreements": 2,
    },
    "autofit_refresh": {
        # NEW 2026-11-19 (autofit_refresh.py, experiments/autofit/augmented.py): recovered poses that are
        # not a detection of the raw store are given to Autofit as detections (no identities), then the
        # round takes a lap: Autofit -> solve -> recover. Golden: 2,240 such poses (0.7% of Autofit's
        # input) took its track ids per labelled person from 45 to 39 and the solve from 31 to 27
        # identities with fewer impure observations (84 -> 50). A lap costs about a round, so it runs
        # only with enough new detections, and again only while laps keep removing identities.
        # min_new_share 0.001 -> 0.005 (2026-11-19, felix): laps with 0.7% (golden) and 5.9% (felix) of
        # Autofit's input removed 12% and 15% of the identities; a second felix lap with 0.28% (4,172
        # detections) cost 66 min and changed nothing in its favour (270 -> 279 identities, positions
        # +0.1%). Three laps are all the evidence for this threshold.
        "enabled": True, "min_new_detections": 500, "min_new_share": 0.005, "max_laps_per_round": 2,
        "min_gain": 0.02,
    },
    "relink": {
        # NEW 2026-11-19 (stages/recover.py, experiments/coverage/links_after_recovery.py): the recovery moves
        # where and when a track is first and last known. When that makes the linking rule above accept
        # links it could not accept on the measured positions (felix hour: 8), the round takes an identity
        # lap: the solve decides identity again on its kept association (seconds, not minutes), and the
        # recovery looks again only around the tracks that were joined. Laps repeat while they join tracks.
        "enabled": True, "max_laps_per_round": 6,
    },
    "coverage": {
        # NEW 2026-11-18 (solve/coverage.py, experiments/coverage): recover positions inside and next
        # to tracks by reading poses where the person must be. A neighbouring measured pose within
        # reach_frames says where to look (never what the position is); end_frames beyond each end of
        # a track are tried too. A camera is asked when at least min_body_in_view body joints and
        # min_box_in_view of the box are inside its image.
        "reach_frames": 30, "end_frames": 10, "min_body_in_view": 6, "min_box_in_view": 0.6,
        # Passes: each accepted recovery is the hint for its neighbours in the next pass (recursion
        # from the edges of what is known); stops when a pass finds nothing. max_passes is only a
        # safety bound (a track followed 10 frames per pass for 40 s needs 40 passes; felix was still
        # finding positions at pass 40); when it is reached the summary says so.
        "max_passes": 400,
        # Skeleton -> detector-style box, fractions of the keypoint-box height (measured on 297k detections).
        "box_margin": {"left": 0.10, "right": 0.10, "top": 0.13, "bottom": 0.14},
        # A crop is dropped when another measured person's box overlaps the hint box this much and
        # the pose read follows that person at least as well (removed 22 of 23 false accepts).
        "competitor_iou": 0.4,
        "min_pair_joints": 6, "search_radius_in": 100.0, "other_identity_clear_in": 24.0,
        # Acceptance on top of the triangulation gates. A: three or more cameras (0 false accepts in
        # 6,500 negative controls). B: two cameras, much stricter on image evidence, and only inside
        # a gap bracketed by measured poses at most max_hint_frames away.
        "tiers": {
            "A": {"min_cameras": 3, "min_body_joints": 9, "max_median_residual_px": 12.0, "max_bad_bones": 1,
                  "max_hint_distance_in": 15.0, "min_view_peak": 0.0, "max_hint_frames": 30},
            "B": {"min_cameras": 2, "min_body_joints": 8, "max_median_residual_px": 3.0, "max_bad_bones": 2,
                  "max_hint_distance_in": 12.0, "min_view_peak": 0.6, "max_hint_frames": 5},
        },
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
