"""Taxonomy of unknowns and of the human tasks that resolve them.

An *unknown* is something the engine cannot decide from its current evidence. Each kind
names (a) the machine steps that may still resolve it and (b) the human task type that
resolves it when machines cannot. A dataset is complete when no unknown is open: every
unknown is resolved, or explicitly marked undeterminable (occluded, out of view, not
enough evidence to measure) - the engine never invents a value to close one.

A *task type* fixes the question, the evidence shown, the answer schema and the
constraints an answer produces. Task payloads share one evidence vocabulary (``Panel``)
so one web UI can render every type; the pose-correction type is carried out in CVAT
because it needs a skeleton editor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- unknowns


@dataclass(frozen=True)
class UnknownKind:
    name: str
    title: str
    meaning: str
    machine_steps: tuple[str, ...]  # what the engine tries before asking a person
    task_type: str | None  # human task that resolves it (None: operator action outside the engine)
    blocks: tuple[str, ...]  # stages that cannot be trusted while it is open


KINDS: dict[str, UnknownKind] = {kind.name: kind for kind in [
    UnknownKind(
        "stream_layout", "Stream layout",
        "A camera's files cannot be ordered/trimmed into one verified frame sequence.",
        ("rebuild from main-channel files", "content-verified overlap trim"),
        "operator_decision", ("detect", "align")),
    UnknownKind(
        "calibration", "Calibration validity",
        "The camera view moved relative to the calibration capture, or the check was inconclusive.",
        ("rotation-only mount correction", "reprojection residual audit after association",
         "refit from identity correspondences"),
        "calibration_review", ("associate", "triangulate")),
    UnknownKind(
        "alignment", "Time alignment",
        "A camera's time offset is unresolved, unstable or stale over a span of the recording.",
        ("audio + pose-motion lag (data_management generator)", "reprojection-vs-lag scan on linked tracks"),
        "alignment_review", ("associate", "triangulate")),
    UnknownKind(
        "tracklet_purity", "Tracklet purity",
        "A single-camera tracklet may contain more than one person (identity switch).",
        ("split at risk events", "3D consistency split across windows"),
        "continuity", ("identity",)),
    UnknownKind(
        "association", "Cross-camera association",
        "Which simultaneous tracklets in other cameras are the same person is ambiguous.",
        ("pairwise geometry with margin", "N-view consistency", "elimination by established links"),
        "cross_view_match", ("triangulate",)),
    UnknownKind(
        "identity_link", "Identity across a gap",
        "A track fragment starts or ends inside the room without a unique continuation.",
        ("3D motion continuity", "single-view tracklet bridge", "appearance ReID", "elimination of occupied identities"),
        "reidentify", ()),
    UnknownKind(
        "identity_swap", "Possible identity swap",
        "Two tracked people came within reach of each other (or their tracks exchanged places) and nothing "
        "independent of the 3D tracker confirms who is who afterwards.",
        ("step-by-step 3D continuity", "heading through the crossing", "3D body proportions before/after",
         "single-camera tracklets in views where the two stay apart", "appearance ReID"),
        "swap_check", ()),
    UnknownKind(
        "unassigned", "Unassigned observations",
        "A tracklet of confident detections belongs to no track (single-view person, or a false detection).",
        ("projection assignment to existing 3D tracks", "single-view bridge between fragments"),
        "detection_triage", ()),
    UnknownKind(
        "pose_ownership", "Pose ownership",
        "Two detections in one camera at one instant claim the same person (duplicate or mixed pose).",
        ("near-identical-pose suppression", "reprojection ranking against the 3D track"),
        "pose_ownership", ()),
    UnknownKind(
        "coverage", "Missing observation",
        "A tracked person projects inside a camera view for a span but has no detection there.",
        ("high-recall re-detection", "low-score / filtered detection recovery by projection"),
        "visibility_check", ()),
    UnknownKind(
        "pose_quality", "Pose quality",
        "Keypoints of an observation disagree with the multi-view 3D pose beyond the reprojection gate.",
        ("drop worst view and re-triangulate", "re-run pose on a 3D-guided crop"),
        "pose_correction", ()),
    UnknownKind(
        "audit", "Audit sample",
        "A random sample of published-candidate labels awaiting human spot check.",
        (), "audit", ("publish",)),
]}


# --------------------------------------------------------------------------- evidence


@dataclass
class Panel:
    """One evidence image the UI asks the server to render.

    The server draws the frame (or a crop around ``box`` with ``pad``), the highlighted
    detections' measured keypoints, and optionally other detections faintly. Evidence
    always shows measured keypoints next to the unobscured image, because a box alone
    does not say which person owns a pose (golden-set learning).
    """
    camera: str
    frame: int  # native frame index
    box: list[float] | None = None  # native pixels; None = whole frame
    highlight: list[str] = field(default_factory=list)  # detection keys drawn in the panel colour
    others: bool = True  # draw the camera's other detections faintly
    project: list[int] = field(default_factory=list)  # track ids whose 3D pose is projected (dashed)
    label: str = ""  # short caption, e.g. "A", "B2", "t+1.5s"
    color: str = "green"
    context: bool = False  # also offer the full frame with the box marked


# --------------------------------------------------------------------------- what a person can say about one image

# Tasks that compare groups of images (ui 'reference_candidates') let the reviewer flag any single image.
# The first three take the image out of its group: the question is then answered with the images that
# remain. 'bad_pose' leaves the image in - who it shows is clear - and reports the keypoints.
IMAGE_FLAGS = {
    "other_person": ("Someone else", "A clean detection of a different person than the rest of its group: the "
                                     "existing grouping is wrong here.",
                     "different(image, rest of its group); if both lie in one single-camera tracklet the engine "
                     "asks where the tracklet changes person (continuity task)"),
    "mixed": ("Two people in one pose", "The skeleton takes joints from two people: it is nobody's detection.",
              "not_person(image, problem=mixed_pose, this detection only)"),
    "not_person": ("Not a person", "A reflection, a poster, equipment.",
                   "not_person(image): this detection, or its whole tracklet when every image of the group is flagged"),
    "bad_pose": ("Bad pose", "The right person, but the keypoints are wrong enough to report.",
                 "bad_pose(image): its keypoints are left out of 3D and it is listed for pose correction"),
}
ANSWER_MODEL = 2  # version of the 'reference_candidates' answer (1: reference_ok / none_of_these / problems)

# --------------------------------------------------------------------------- task types


@dataclass(frozen=True)
class TaskType:
    name: str
    title: str
    question: str
    ui: str  # 'decision' | 'strip' | 'reference_candidates' | 'frame_picker' | 'cvat'
    answer_schema: dict  # documented shape of the answer JSON
    produces: tuple[str, ...]  # constraint kinds an answer can create
    learnings: str  # what answers are reused for (model/threshold retraining)


TASK_TYPES: dict[str, TaskType] = {task.name: task for task in [
    TaskType(
        "operator_decision", "Operator decision",
        "The engine needs a decision about the input data or configuration.",
        "decision",
        {"choice": "one of payload.options[].value", "notes": "str"},
        ("alignment", "undeterminable"),
        "None; recorded for provenance."),
    TaskType(
        "calibration_review", "Calibration review",
        "Does the projected reference geometry line up with this camera's image?",
        "decision",
        {"choice": "use_standard | use_corrected | exclude_camera", "notes": "str"},
        (),
        "Accepted corrections become the dataset's derived calibration; residual statistics feed the next refit."),
    TaskType(
        "alignment_review", "Time alignment review",
        "Which frame of the target camera shows the same instant as the reference frame?",
        "frame_picker",
        {"choice": "index into payload.candidates | 'none' | 'unsure'", "notes": "str"},
        ("alignment",),
        "Human checkpoints are published as a human-confirmed frame_alignment set and bound the generator's search."),
    TaskType(
        "continuity", "Tracklet continuity",
        "Do all crops show the same person? If not, click the first crop showing a different person.",
        "strip",
        {"verdict": "continuous | break | unclear", "first_different": "int index into payload.panels (break)",
         "unclear": "[int] crop indices that cannot be judged", "notes": "str"},
        ("same", "split", "different"),
        "Break locations label tracker risk events; they tune the over-split rule and the BoT-SORT match threshold."),
    TaskType(
        "cross_view_match", "Cross-camera match",
        "Which tracked person, if any, is the person in reference A?",
        "reference_candidates",
        {"images": "{detection key: {'exclude': 'other_person' | 'mixed' | 'not_person' | null, 'bad_pose': bool}}",
         "candidates": "{candidate id: 'same' | 'different' | 'unsure'} - at most one 'same'", "notes": "str"},
        ("same", "different", "not_person", "bad_pose", "undeterminable"),
        "Same/different pairs with their geometry features retrain the association accept/reject thresholds."),
    TaskType(
        "reidentify", "Re-identification across a gap",
        "Which of these earlier tracks show the same person as reference A (seen again later)? More than one "
        "can be right when a person's track broke several times. Or 'none of these'.",
        "reference_candidates",
        {"images": "{detection key: {'exclude': ..., 'bad_pose': bool}}",
         "candidates": "{candidate id: 'same' | 'different' | 'unsure'} - several may be 'same', except pairs in "
                       "payload.cannot_both", "notes": "str"},
        ("same", "different", "not_person", "bad_pose", "undeterminable"),
        "Positive/negative fragment pairs are ReID training pairs for this camera set and calibrate the appearance margin."),
    TaskType(
        "swap_check", "Identity swap check",
        "Two people came close. Which of the candidates (after) is reference A (before)? Neither is a valid answer.",
        "reference_candidates",
        {"images": "{detection key: {'exclude': ..., 'bad_pose': bool}}",
         "candidates": "{candidate id: 'same' | 'different' | 'unsure'} - at most one 'same'; both 'different' = "
                       "neither", "notes": "str"},
        ("same", "different", "split", "not_person", "bad_pose", "undeterminable"),
        "Confirmed swaps and confirmed non-swaps calibrate the swap cues (body, heading, appearance) and are "
        "hard cases for the 3D tracker's association."),
    TaskType(
        "detection_triage", "Unassigned detection triage",
        "Is this a real person? (Seen by one camera only, and matching no tracked person.)",
        "reference_candidates",
        {"images": "{detection key: {'exclude': ..., 'bad_pose': bool}}",
         "verdict": "'person' | 'not_person' | 'unsure'", "notes": "str"},
        ("not_person", "bad_pose", "undeterminable"),
        "Not-person crops are hard negatives for the detector; confirmed people are recall positives."),
    TaskType(
        "pose_ownership", "Pose ownership",
        "Two poses in one camera at the same place. One person detected twice, or two people?",
        "reference_candidates",
        {"images": "{detection key: {'exclude': ..., 'bad_pose': bool}}",
         "candidates": "{'B': 'same' (one person) | 'different' (two people) | 'unsure'}",
         "better": "'A' | 'B' | 'equal' - which pose to use when they are one person", "notes": "str"},
        ("duplicate", "different", "not_person", "bad_pose"),
        "Duplicate/mixed-pose labels tune the near-identical-pose suppression threshold."),
    TaskType(
        "visibility_check", "Visibility check",
        "The dashed skeleton is where this person should be. Are they visible in this view?",
        "decision",
        {"choice": "visible | occluded | out_of_view | wrong_projection", "notes": "str"},
        ("undeterminable",),
        "'visible' answers queue a pose_correction task and are detector recall misses; 'occluded' trains the visibility model."),
    TaskType(
        "pose_correction", "Pose correction (CVAT)",
        "Correct or draw this person's 13-point skeleton.",
        "cvat",
        {"cvat_task": "int", "cvat_job": "int", "frames": "[{camera, frame, keypoints}] read back from CVAT"},
        ("same",),
        "Corrected skeletons are appended to the curated pose dataset for the next YOLO/ViTPose fine-tune."),
    TaskType(
        "audit", "Audit",
        "Is this label correct (right person, plausible pose)?",
        "decision",
        {"choice": "correct | wrong_identity | wrong_pose | not_person | unsure", "notes": "str"},
        ("different", "not_person"),
        "Audit precision per stage is the publish gate and the regression metric for retrained models."),
]}

UI_KINDS = ("decision", "strip", "reference_candidates", "frame_picker", "cvat")
