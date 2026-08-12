#!/usr/bin/env python3
"""Derive LVIS regression metrics with the official LVIS evaluation API."""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "tests/lvis_dataset/lvis_val_100.json"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "tests/lvis_dataset/lvis_results_100.json"


def derive_metrics(annotation_file: Path, prediction_file: Path) -> dict[str, float]:
    """Run official LVIS bbox evaluation and return its scalar metrics."""
    try:
        from lvis.eval import LVISEval
        from lvis.lvis import LVIS
    except ImportError as exc:
        raise RuntimeError("Install the official lvis-api package before deriving LVIS goldens.") from exc

    lvis_gt = LVIS(str(annotation_file))
    evaluator = LVISEval(lvis_gt, str(prediction_file), iou_type="bbox")
    evaluator.run()
    return {key: float(value) for key, value in evaluator.get_results().items()}


def main() -> None:
    """Parse LVIS fixture paths and print official metrics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", nargs="?", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("predictions", nargs="?", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    print(json.dumps(derive_metrics(args.annotations, args.predictions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
