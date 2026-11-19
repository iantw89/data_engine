"""Dataset description shared by every stage.

``dataset.json`` is written once by the ingest stage (which needs the data_management
catalog) and then read everywhere else, so the engine core does not import data_management.

Conventions (same as data_management label sets):
- camera ids are current catalogue ids (``felix-3``);
- frame indices are 0-based *native* indices in presentation order across chunks, after
  content-verified overlap trims (data_management's index). A part's ``overlap`` frames are
  in this index but repeat earlier frames: the alignment keeps them off the timeline;
- pixel coordinates are native pixels unless a name says ``norm`` (1280x960 calibration scale).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Part:
    path: str
    trim: int  # leading frames of this file that duplicate the previous part
    count: int  # frames this part contributes to the stream
    verified: bool = True
    # Leading frames (after ``trim``) that repeat the previous part's last frames, found by pixel
    # comparison (``overlap.py``). They stay in the frame index but are not part of the timeline.
    overlap: int = 0


@dataclass
class Camera:
    camera_id: str
    calibration_id: str | None
    width: int
    height: int
    fps: float
    parts: list[Part] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return sum(part.count for part in self.parts)

    def locate(self, frame: int) -> tuple[str, int]:
        """Map a native stream frame to (file, index inside that file)."""
        if frame < 0:
            raise IndexError(frame)
        cumulative = 0
        for part in self.parts:
            if frame < cumulative + part.count:
                return part.path, part.trim + frame - cumulative
            cumulative += part.count
        raise IndexError(f"{self.camera_id}: frame {frame} outside 0..{cumulative - 1}")

    def repeated_frames(self) -> list[tuple[int, int]]:
        """``(frame, frame it repeats)`` for every frame that a part's ``overlap`` takes off the timeline."""
        pairs, cumulative = [], 0
        for part in self.parts:
            pairs += [(cumulative + j, cumulative - part.overlap + j) for j in range(part.overlap)]
            cumulative += part.count
        return pairs


@dataclass
class Dataset:
    dataset_id: str
    scene: str
    start: str
    end: str
    cameras: dict[str, Camera] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.scene, self.start, self.end)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> "Dataset":
        cameras = {
            cid: Camera(**{**cam, "parts": [Part(**part) for part in cam["parts"]]})
            for cid, cam in data["cameras"].items()
        }
        return Dataset(data["dataset_id"], data["scene"], data["start"], data["end"], cameras)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Dataset":
        return Dataset.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def parse_dataset_id(dataset_id: str) -> tuple[str, str, str]:
    scene, start, end = dataset_id.rsplit("_", 2)
    return scene, start, end
