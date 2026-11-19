"""Mount check of a dataset against the calibration capture. Runs in data_management's interpreter.

Wraps ``data_management.mount_check.check_time_range`` (SIFT background match + rotation fit; CPU
only, ~9 decoded frames per camera) with one change: each camera stream is first fixed the way
``ingest_dm`` does it. A camera whose catalog stream mixes main- (channel 0) and sub-channel files
is rebuilt from its channel-0 files (``ingest_dm.main_channel_stream``); on the mixed stream the
background samples come from both channels and the check is inconclusive (seen on felix-6).

Nothing is written under data_management/data: the check's cache lives in ``<out>/mount_cache``.
Output ``<out>/mount_check.json``: mount_check's result plus ``data_engine`` provenance
(dataset id, cameras rebuilt from the main channel, frames per camera, sha256 of the inputs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_engine.dataset import Dataset, parse_dataset_id  # noqa: E402
from data_engine.workers.ingest_dm import main_channel_stream  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_time_range(catalog, dataset_id: str, video_root: str):
    key = parse_dataset_id(dataset_id)
    ranges = catalog.scan_dataset(video_root, probe=True).time_ranges + catalog.load_extra_datasets(probe=True)
    matches = [tr for tr in ranges if tuple(tr.key) == key]
    if len(matches) != 1:
        raise SystemExit(f"Dataset {dataset_id}: {len(matches)} catalog matches")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--refresh", action="store_true", help="ignore the mount check cache")
    args = parser.parse_args()

    from data_management import catalog, mount_check

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.load(Path(args.dataset_json))
    if dataset.dataset_id.split('@', 1)[0] != args.dataset:
        raise SystemExit(f"{args.dataset_json} describes {dataset.dataset_id}, not {args.dataset}")
    tr = find_time_range(catalog, args.dataset, args.video_root)
    rebuilt = []
    for camera_id, cam in sorted(tr.cameras.items()):
        replacement = main_channel_stream(catalog, cam)
        if replacement is not None:
            tr.cameras[camera_id] = replacement
            rebuilt.append(camera_id)
            print(f"{camera_id}: stream rebuilt from channel-0 files", flush=True)

    result = mount_check.check_time_range(tr, Path(args.calibration), samples=args.samples,
                                          cache_dir=out / "mount_cache", refresh=args.refresh)
    result = dict(result)
    result["data_engine"] = {
        "dataset": args.dataset, "main_channel_rebuilt": rebuilt,
        "frames": {cid: cam.frame_count for cid, cam in dataset.cameras.items()},
        "dataset_json_sha256": sha256(Path(args.dataset_json)), "calibration_sha256": sha256(Path(args.calibration)),
        "samples": args.samples,
    }
    (out / "mount_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for camera_id, check in sorted(result["cameras"].items()):
        shift = check.get("median_shift_px")
        print(f"{camera_id:10s} {check['status']:12s} shift_px={shift} rot_deg={check.get('rotation_deg')} "
              f"residual_px={check.get('residual_px')} {check.get('reason', '')}", flush=True)
    print(f"wrote {out / 'mount_check.json'}", flush=True)


if __name__ == "__main__":
    main()
