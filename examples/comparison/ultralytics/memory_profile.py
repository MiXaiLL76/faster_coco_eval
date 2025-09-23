import argparse
import psutil
import os
from tabulate import tabulate

from pycocotools.coco import COCO as orig_COCO
from pycocotools.cocoeval import COCOeval as orig_COCOeval

from faster_coco_eval import COCO as faster_COCO
from faster_coco_eval import __version__ as faster_coco_version
from faster_coco_eval import COCOeval_faster as faster_COCOeval

def get_used_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return round(memory_info.rss / 1024 / 1024, 2)

def main(coco_gt : str, predictions : str, coco_lib : str = "pycocotools", separate_eval : bool = False):
    memory_flow = [{"1. Initial" : get_used_memory()}]

    if coco_lib == "pycocotools":
        COCO_CLASS = orig_COCO
        COCO_ARGS = {}

        EVAL_CLASS = orig_COCOeval
        EVAL_ARGS = {}
    else:
        print(f"faster_coco_eval __version__ = {faster_coco_version}")
        COCO_CLASS = faster_COCO
        COCO_ARGS = {"print_function" : print}

        EVAL_CLASS = faster_COCOeval
        EVAL_ARGS = {"print_function" : print, "separate_eval" : separate_eval}

    memory_flow.append({"2. Classes imported" : get_used_memory()})

    cocoGt = COCO_CLASS(coco_gt, **COCO_ARGS)
    memory_flow.append({"3. Ground truth loaded" : get_used_memory()})

    cocoDt = cocoGt.loadRes(predictions)
    memory_flow.append({"4. Predictions loaded" : get_used_memory()})

    cocoEval = EVAL_CLASS(cocoGt, cocoDt, "bbox", **EVAL_ARGS)
    memory_flow.append({"5. Evaluator created" : get_used_memory()})

    cocoEval.evaluate()

    memory_flow.append({"6. Evaluation completed" : get_used_memory()})

    cocoEval.accumulate()
    memory_flow.append({"7. Results accumulated" : get_used_memory()})

    cocoEval.summarize()
    memory_flow.append({"8. Summary generated" : get_used_memory()})

    # Print memory flow as table
    table_data = []
    prev_memory = None

    for entry in memory_flow:
        stage = list(entry.keys())[0]
        memory = list(entry.values())[0]

        if prev_memory is not None:
            diff = memory - prev_memory
            diff_str = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        else:
            diff_str = "-"

        table_data.append([stage, f"{memory:.2f} MB", diff_str])
        prev_memory = memory

    print("\nMemory Usage Profile:")
    print(tabulate(table_data, headers=["Stage", "Memory Usage", "Diff"], tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile memory usage of COCO evaluation")
    parser.add_argument("coco_gt", help="Path to COCO ground truth JSON file")
    parser.add_argument("predictions", help="Path to predictions JSON file")
    parser.add_argument("--coco-lib", choices=["pycocotools", "faster_coco_eval"],
                       default="pycocotools", help="COCO library to use")
    parser.add_argument("--separate-eval", action="store_true", default=False,
                       help="Use separate evaluation mode (faster_coco_eval only)")

    args = parser.parse_args()
    main(args.coco_gt, args.predictions, args.coco_lib, args.separate_eval)
