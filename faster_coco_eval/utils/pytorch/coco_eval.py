# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""COCO evaluator that works in distributed mode.

Mostly copy-paste from
https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that pycocotools is replaced by a faster library faster-coco-eval
"""

import contextlib
import copy
import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


class FasterCocoEvaluator:
    """COCO evaluator for distributed evaluation.

    Args:
        coco_gt (COCO): Ground truth COCO object.
        iou_types (List[str]): List of IoU types to evaluate, e.g., ['bbox', 'segm', 'keypoints'].
        lvis_style (bool, optional): Whether to use LVIS-style evaluation. Defaults to False.
    """

    def __init__(self, coco_gt: COCO, iou_types: List[str], lvis_style: bool = False):
        """Initializes the FasterCocoEvaluator.

        Args:
            coco_gt (COCO): Ground truth COCO object.
            iou_types (List[str]): List of IoU types to evaluate, e.g., ['bbox', 'segm', 'keypoints'].
            lvis_style (bool, optional): Whether to use LVIS-style evaluation. Defaults to False.
        """
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.world_size = None  # None = auto
        self.lvis_style = lvis_style

        self.iou_types = iou_types
        self.coco_eval: Dict[str, COCOeval_faster] = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                coco_gt, iouType=iou_type, lvis_style=lvis_style, print_function=print, separate_eval=True
            )

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.stats_as_dict = {k: [] for k in iou_types}

    def cleanup(self) -> None:
        """Cleans up and re-initializes the evaluator state for a new
        evaluation run.

        Returns:
            None
        """
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                self.coco_gt, iouType=iou_type, lvis_style=self.lvis_style, print_function=print, separate_eval=True
            )
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions: Dict[Any, Any]) -> None:
        """Updates the evaluator with new predictions.

        Args:
            predictions (Dict): Dictionary mapping image ids to predictions. The structure depends on the IoU type.

        Returns:
            None
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]
            # suppress prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)
                )
            )

    def synchronize_between_processes(self) -> None:
        """Synchronizes results across distributed processes.

        Returns:
            None
        """
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type], self.world_size)

            coco_eval = self.coco_eval[iou_type]
            coco_eval._evalImgs_cpp = eval_imgs
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

    def accumulate(self) -> None:
        """Accumulates evaluation results.

        Returns:
            None
        """
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> None:
        """Prints and stores evaluation statistics.

        Returns:
            None
        """
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
            self.stats_as_dict[iou_type] = coco_eval.stats_as_dict

    def prepare(self, predictions: Dict[Any, Any], iou_type: str) -> List[Dict[str, Any]]:
        """Prepares predictions for COCO evaluation.

        Args:
            predictions (Dict): Dictionary mapping image ids to predictions.
            iou_type (str): Type of IoU to prepare for, e.g., 'bbox', 'segm', or 'keypoints'.

        Returns:
            List[Dict]: List of detection results in COCO format.
        """
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions: Dict[Any, Any]) -> List[Dict[str, Any]]:
        """Converts bounding box predictions to COCO detection format.

        Args:
            predictions (Dict): Dictionary mapping image ids to prediction dicts. Each prediction dict must contain:
                - "boxes": Tensor of shape [N, 4] (x1, y1, x2, y2)
                - "scores": torch.Tensor of shape [N] of length N
                - "labels": torch.Tensor of shape [N] of length N

        Returns:
            List[Dict]: List of detection results in COCO format.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend([
                {
                    "image_id": original_id,
                    "category_id": int(labels[k]),
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ])
        return coco_results

    def prepare_for_coco_segmentation(self, predictions: Dict[Any, Any]) -> List[Dict[str, Any]]:
        """Converts mask predictions to COCO segmentation format.

        Args:
            predictions (Dict): Dictionary mapping image ids to prediction dicts. Each prediction dict must contain:
                - "scores": torch.Tensor of shape [N]
                - "labels": torch.Tensor of shape [N]
                - "masks": Tensor of shape [N, 1, H, W]

        Returns:
            List[Dict]: List of segmentation results in COCO format.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend([
                {
                    "image_id": original_id,
                    "category_id": int(labels[k]),
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ])
        return coco_results

    def prepare_for_coco_keypoint(self, predictions: Dict[Any, Any]) -> List[Dict[str, Any]]:
        """Converts keypoint predictions to COCO keypoint format.

        Args:
            predictions (Dict): Dictionary mapping image ids to prediction dicts. Each prediction dict must contain:
                - "boxes": Tensor of shape [N, 4]
                - "scores": torch.Tensor of shape [N]
                - "labels": torch.Tensor of shape [N]
                - "keypoints": Tensor of shape [N, K, 3]

        Returns:
            List[Dict]: List of keypoint results in COCO format.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend([
                {
                    "image_id": original_id,
                    "category_id": int(labels[k]),
                    "keypoints": keypoint,
                    "score": scores[k],
                }
                for k, keypoint in enumerate(keypoints)
            ])
        return coco_results


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from (xmin, ymin, xmax, ymax) to (xmin, ymin,
    width, height) format.

    Args:
        boxes (torch.Tensor): Bounding boxes of shape [N, 4].

    Returns:
        torch.Tensor: Converted bounding boxes of shape [N, 4].
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def all_gather(data: Any, world_size: int = None) -> List[Any]:
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data (Any): Any picklable object.
        world_size (int, optional): Number of processes in distributed mode. If None, auto-detect.

    Returns:
        List[Any]: List of data gathered from each rank.
    """
    if world_size is None:
        if not (dist.is_available() and dist.is_initialized()):
            world_size = 1
        else:
            world_size = dist.get_world_size()

    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    byte_array = bytearray(buffer)
    tensor = torch.ByteTensor(list(byte_array)).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def merge(
    img_ids: Union[List[Any], np.ndarray], eval_imgs: List[Any], world_size: int = None
) -> Tuple[List[Any], List[Any]]:
    """Merges evaluation results from all processes.

    Args:
        img_ids (List or np.ndarray): List or array of image ids.
        eval_imgs (List): List of evaluation image results.
        world_size (int, optional): Number of processes in distributed mode. If None, auto-detect.

    Returns:
        Tuple[List, List]: (merged image ids, merged eval images)
    """
    all_img_ids = all_gather(img_ids, world_size)
    all_eval_imgs = all_gather(eval_imgs, world_size)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx].ravel()

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
