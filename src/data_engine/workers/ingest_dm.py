"""Resolve a dataset from the data_management catalog. Runs in data_management's interpreter.

Writes ``dataset.json``: per camera the ordered source files with verified overlap trims.
A camera whose files mix the main (channel 0) and sub (channel 1) streams is rebuilt from
its main-channel files only; the catalog groups both channels under one camera id, which
makes the overlap check fail (seen on felix-6 / NVR channel 8-7, 2025-12-23).
Unresolved layouts are reported as issues, never guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.dataset import Camera, Dataset, Part, parse_dataset_id  # noqa: E402


def main_channel_stream(catalog, cam):
    """Return a CameraStream built from the camera's channel-0 files, or None if not mixed."""
    parts = sorted([(cam.parsed.start, cam.parsed.part, cam.path)] + cam.extra_parts)
    parsed = [catalog.parse_filename(os.path.basename(path)) for _, _, path in parts]
    if any(name is None for name in parsed):
        return None  # registered extra dataset (e.g. the golden recording): files are not NVR channel exports
    channels = {name.channel for name in parsed}
    if channels == {0} or 0 not in channels:
        return None
    main = [part for part in parts if catalog.parse_filename(os.path.basename(part[2])).channel == 0]
    first = main[0][2]
    stream = catalog.CameraStream(
        camera_id=cam.camera_id, camera_pair=cam.camera_pair, channel=0, path=first,
        parsed=catalog.parse_filename(os.path.basename(first)), file_size=os.path.getsize(first),
        meta=catalog.probe_video(first), camera_number=cam.camera_number,
    )
    stream.extra_parts = list(main[1:])
    catalog._aggregate_parts(stream, True)
    return stream


def resolve(dataset_id: str, video_root: str) -> Dataset:
    from data_management import catalog
    from data_management.recorded_capture import calibration_id_for

    key = parse_dataset_id(dataset_id)
    ranges = catalog.scan_dataset(video_root, probe=True).time_ranges + catalog.load_extra_datasets(probe=True)
    matches = [tr for tr in ranges if tuple(tr.key) == key]
    if len(matches) != 1:
        raise SystemExit(f"Dataset {dataset_id}: {len(matches)} catalog matches")
    tr = matches[0]
    dataset = Dataset(dataset_id, *key)
    for camera_id, cam in sorted(tr.cameras.items()):
        notes = []
        replacement = main_channel_stream(catalog, cam)
        if replacement is not None:
            notes.append("catalog stream mixed main and sub channel files; rebuilt from channel 0 only")
            cam = replacement
        issues = list(cam.layout_issues)
        single_file = not cam.resolved_parts and len(cam.part_paths) == 1 and cam.meta and cam.meta.frame_count
        if single_file:
            # One file per camera (registered extra datasets): the file is the stream.
            parts = [Part(cam.part_paths[0], 0, int(cam.meta.frame_count), True)]
        else:
            if not cam.resolved_parts or not cam.part_frame_counts:
                issues.append("no resolved chunk layout")
            parts = [
                Part(part.path, int(part.trim), int(part.count), bool(part.verified or index == 0))
                for index, part in enumerate(cam.resolved_parts) if part.count
            ]
        for part in parts[1:]:
            if not part.verified:
                notes.append(f"continuation without overlap evidence: {os.path.basename(part.path)}")
        dataset.cameras[camera_id] = Camera(
            camera_id=camera_id, calibration_id=calibration_id_for(cam),
            width=cam.meta.width if cam.meta else 0, height=cam.meta.height if cam.meta else 0,
            fps=cam.meta.fps if cam.meta else 0.0, parts=parts, issues=issues, notes=notes,
        )
    return dataset


def export_frame_index(dataset: Dataset, directory: Path) -> None:
    """Copy data_management's exact per-file frame times into the workspace for the engine's reader."""
    import numpy as np
    from data_management.frame_index import frame_times

    from data_engine.video import index_cache_path

    directory.mkdir(parents=True, exist_ok=True)
    for camera in dataset.cameras.values():
        for part in camera.parts:
            target = index_cache_path(part.path, directory)
            if not target.is_file():
                times = np.asarray(frame_times(part.path), dtype=np.float64)
                if part.trim + part.count > len(times):
                    raise SystemExit(f"{part.path}: layout needs {part.trim + part.count} frames, index has {len(times)}")
                np.save(target, times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = resolve(args.dataset, args.video_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dataset.save(Path(args.out))
    export_frame_index(dataset, Path(args.out).parent / "frame_index")
    print(json.dumps({cid: {"frames": cam.frame_count, "issues": cam.issues, "notes": cam.notes}
                      for cid, cam in dataset.cameras.items()}, indent=1))


if __name__ == "__main__":
    main()
