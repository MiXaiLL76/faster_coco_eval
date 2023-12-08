import argparse
import numpy as np
import pycocotools.mask as cocomask
import tqdm
import time
from mmeval.metrics import COCODetection  # type: ignore
from faster_coco_detection import FasterCOCODetection
from mmdet.datasets import CocoDataset
from mmdet.apis import DetInferencer


def do_mmeval_evaluate(config_file : str, checkpoint : str):
    model = DetInferencer(config_file, checkpoint, show_progress=False)

    coco_dataset = CocoDataset(
        # ann_file='annotations/instances_val2017_short.json',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root='./COCO/DIR'
    )

    faster_coco_metric = FasterCOCODetection(
        ann_file=coco_dataset.ann_file,
        metric=["bbox", "segm"],
        proposal_nums=[1, 10, 100],
    )
    coco_metric = COCODetection(
        ann_file=coco_dataset.ann_file,
        metric=["bbox", "segm"],
        proposal_nums=[1, 10, 100],
    )

    faster_coco_metric.dataset_meta = {
        "CLASSES": coco_dataset.METAINFO['classes']
    }
    coco_metric.dataset_meta = {
        "CLASSES": coco_dataset.METAINFO['classes']
    }

    for item in tqdm.tqdm(coco_dataset):
        pred_results = model(item['img_path'])['predictions'][0]
        pred_results['bboxes'] = np.array(pred_results['bboxes'])
        pred_results['img_id'] = item['img_id']
        coco_metric.add_predictions([pred_results])
        faster_coco_metric.add_predictions([pred_results])

    ts1 = time.time()
    coco_metric.compute()
    te1 = time.time()

    print(f"coco_metric.compute() : {te1-ts1:.3f}")

    ts2 = time.time()
    faster_coco_metric.compute()
    te2 = time.time()

    print(f"faster_coco_metric.compute() : {te2-ts2:.3f}")

    print((te2-ts2) / (te1-ts1))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='load a model for evaluation.', required=True)
    parser.add_argument('--config', type=str, help='load a config for evaluation.', required=True)
    args = parser.parse_args()

    do_mmeval_evaluate(args.config, args.load)