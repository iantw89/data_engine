"""Dataset calibration store access. Runs in data_management's interpreter.

Two modes, both thin wrappers over ``data_management.calibration_store``:

``--mode resolve --dataset ID --out DIR``
    Writes ``DIR/base.json``: the calibration data_management resolves for the dataset (its own stored
    calibration, else its default), with ``source`` ("dataset" | "default" | "unavailable"), path, sha256 and the
    camera ids the dataset overrides. Exit 0; exit 3 when the store refuses (stale override: the default changed
    after it was derived; corrupt record), with ``{"error": ...}`` in base.json. Nothing is written to the store.

``--mode store --dataset ID --calibration FILE --plan PLAN.json [--replace]``
    Registers FILE as the dataset's calibration. PLAN.json carries ``produced_by``, ``reasons`` (calibration id ->
    why), ``provenance`` and ``expected_base_sha256``: the calibration the file was derived from, which must be
    what the dataset resolves to right now (otherwise the file was built on a different base: exit 3, nothing
    written). Never overwrites without ``--replace``. Only ever started by ``data-engine store-calibration``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "base.json"
    try:
        from data_management import calibration_store as store
    except ImportError as exc:  # an older data_management without per-dataset calibrations
        target.write_text(json.dumps({"source": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}), encoding="utf-8")
        print(f"data_management has no calibration_store ({exc}); using the site default")
        return 0
    try:
        resolved = store.calibration_for_dataset(args.dataset)
    except FileNotFoundError as exc:  # no stored calibration and no data_management default file
        target.write_text(json.dumps({"source": "default", "path": None, "reason": str(exc)}), encoding="utf-8")
        print(f"no stored calibration for {args.dataset}")
        return 0
    except store.CalibrationStoreError as exc:
        target.write_text(json.dumps({"error": str(exc), "stale": isinstance(exc, store.StaleCalibrationError)}), encoding="utf-8")
        print(f"refused: {exc}")
        return 3
    target.write_text(json.dumps({**resolved.summary(), "stale": False}, indent=1), encoding="utf-8")
    print(f"{args.dataset}: {resolved.source} calibration {resolved.sha256[:12]}"
          + (f", cameras overridden {list(resolved.cameras_overridden)}" if resolved.cameras_overridden else ""))
    return 0


def store_calibration(args) -> int:
    from data_management import calibration_store as store

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    try:
        resolved = store.calibration_for_dataset(args.dataset)
        if resolved.sha256 != plan["expected_base_sha256"]:
            print(f"refused: the workspace calibration was derived from {plan['expected_base_sha256'][:12]} but "
                  f"{args.dataset} now resolves to {resolved.sha256[:12]} ({resolved.source}); re-run the calibration stage")
            return 3
        record = store.register_calibration(
            args.dataset, Path(args.calibration), produced_by=plan["produced_by"], reasons=plan.get("reasons"),
            provenance=plan.get("provenance"), replace=args.replace)
    except store.CalibrationStoreError as exc:
        print(f"refused: {exc}")
        return 3
    print(f"stored calibration {record['sha256'][:12]} for {args.dataset}: cameras {record['cameras_overridden']} "
          f"(derived from default {record['base']['sha256'][:12]})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("resolve", "store"))
    parser.add_argument("--dataset", required=True, help="source dataset id <scene>_<start>_<end> (no @tag)")
    parser.add_argument("--out")
    parser.add_argument("--calibration")
    parser.add_argument("--plan")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.mode == "resolve":
        if not args.out:
            parser.error("--out is required")
        sys.exit(resolve(args))
    if not (args.calibration and args.plan):
        parser.error("--calibration and --plan are required")
    sys.exit(store_calibration(args))


if __name__ == "__main__":
    main()
