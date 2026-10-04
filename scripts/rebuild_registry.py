#!/usr/bin/env python
"""Rebuild results/runs.jsonl from the saved weights in runs/.

Every ``runs/<id>.pt`` carries the run's config and its final metrics, so the registry
is a derived index, not the only copy of the data. This restores it when it is lost or
truncated -- for instance by a `git checkout` landing on it while a sweep is appending.

Merges rather than replaces: records already in the registry are kept as they are, and
only run ids missing from it are added. Writes atomically via a temporary file.

Usage: python scripts/rebuild_registry.py [--dry-run] [--weights runs] [--out results/runs.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clockpizza.train import REGISTRY, WEIGHTS_DIR  # noqa: E402


def records_from_weights(weights_dir: pathlib.Path) -> dict[str, dict]:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    out: dict[str, dict] = {}
    for path in sorted(weights_dir.glob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", mmap=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if "metrics" not in payload or "config" not in payload:
            continue
        record = {**payload["metrics"], "config": payload["config"]}
        record.setdefault("run_id", path.stem)
        out[record["run_id"]] = record
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_DIR, type=pathlib.Path)
    ap.add_argument("--out", default=REGISTRY, type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing: dict[str, dict] = {}
    if args.out.exists():
        with args.out.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing[r["run_id"]] = r
    print(f"registry {args.out}: {len(existing)} records")

    recovered = records_from_weights(args.weights)
    print(f"weights  {args.weights}: {len(recovered)} checkpoints")

    missing = {k: v for k, v in recovered.items() if k not in existing}
    print(f"missing from the registry: {len(missing)}")
    if missing:
        counts = collections.Counter(r["config"].get("experiment", "?") for r in missing.values())
        for exp, n in sorted(counts.items()):
            print(f"  {exp}: {n}")

    if not missing:
        print("nothing to do")
        return 0
    if args.dry_run:
        print("dry run; registry not modified")
        return 0

    merged = list(existing.values()) + list(missing.values())
    tmp = args.out.with_suffix(args.out.suffix + ".rebuilt")
    with tmp.open("w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(args.out)
    print(f"wrote {len(merged)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
