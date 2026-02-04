"""
Extensive comparison tests between faster_coco_eval and pycocotools.

This test suite validates that faster_coco_eval produces identical results to
pycocotools across a wide range of scenarios with larger, more realistic datasets.
These tests address the requirement for more extensive validation beyond single examples.
"""

import json
import os
import os.path as osp
import tempfile
import unittest
from unittest import TestCase

import numpy as np
from parameterized import parameterized

try:
    from pycocotools.coco import COCO as origCOCO
    from pycocotools.cocoeval import COCOeval as origCOCOeval
except ImportError:
    origCOCO = None
    origCOCOeval = None

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO, COCOeval_faster


class TestExtensivePycocotoolsComparison(TestCase):
    """
    Extensive test suite comparing faster_coco_eval with pycocotools.
    
    Tests multiple scenarios with larger datasets to ensure equality:
    - Object detection (bbox) with many images and annotations
    - Instance segmentation (segm) with many images and annotations  
    - Keypoint detection with many images and annotations
    - Various category distributions and object sizes
    - Different score distributions and confidence levels
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_coco_annotations(
        self,
        num_images=100,
        num_categories=10,
        annotations_per_image=10,
        include_segmentation=False,
        include_keypoints=False,
    ):
        """
        Create a synthetic COCO dataset with configurable parameters.
        
        Args:
            num_images: Number of images in the dataset
            num_categories: Number of object categories
            annotations_per_image: Average number of annotations per image
            include_segmentation: Whether to include segmentation masks
            include_keypoints: Whether to include keypoint annotations
        
        Returns:
            Dictionary containing COCO-formatted annotations
        """
        np.random.seed(42)  # For reproducibility
        
        images = []
        annotations = []
        categories = []
        
        # Create categories
        for cat_id in range(num_categories):
            category = {
                "id": cat_id,
                "name": f"category_{cat_id}",
                "supercategory": f"super_{cat_id % 3}",
            }
            if include_keypoints:
                # Add keypoint definition (17 keypoints like COCO person)
                category["keypoints"] = [f"keypoint_{i}" for i in range(17)]
                category["skeleton"] = [[i, i + 1] for i in range(0, 16, 2)]
            categories.append(category)
        
        # Create images and annotations
        ann_id = 0
        for img_id in range(num_images):
            # Image dimensions vary
            img_width = np.random.randint(400, 800)
            img_height = np.random.randint(400, 800)
            
            images.append({
                "id": img_id,
                "width": img_width,
                "height": img_height,
                "file_name": f"image_{img_id:06d}.jpg",
            })
            
            # Variable number of annotations per image
            num_anns = np.random.randint(
                max(1, annotations_per_image - 5),
                annotations_per_image + 5
            )
            
            for _ in range(num_anns):
                # Random category
                cat_id = np.random.randint(0, num_categories)
                
                # Random bbox with various sizes
                # Create small, medium, and large objects (COCO size categories)
                size_type = np.random.choice(['small', 'medium', 'large'])
                if size_type == 'small':
                    w = np.random.randint(10, 32)
                    h = np.random.randint(10, 32)
                elif size_type == 'medium':
                    w = np.random.randint(32, 96)
                    h = np.random.randint(32, 96)
                else:
                    w = np.random.randint(96, min(200, img_width // 2))
                    h = np.random.randint(96, min(200, img_height // 2))
                
                x = np.random.randint(0, max(1, img_width - w))
                y = np.random.randint(0, max(1, img_height - h))
                
                area = w * h
                
                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(area),
                    "iscrowd": 0,
                }
                
                if include_segmentation:
                    # Create a simple segmentation mask
                    mask = np.zeros((img_height, img_width), order="F", dtype=np.uint8)
                    # Fill the bounding box region
                    mask[y:y+h, x:x+w] = 1
                    rle_mask = mask_util.encode(mask)
                    rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
                    annotation["segmentation"] = rle_mask
                
                if include_keypoints:
                    # Create random keypoints within the bbox
                    keypoints = []
                    num_keypoints = 17
                    num_visible = 0
                    for i in range(num_keypoints):
                        # Some keypoints are visible (v=2), some occluded (v=1), some not labeled (v=0)
                        visibility = int(np.random.choice([0, 1, 2], p=[0.1, 0.2, 0.7]))
                        if visibility > 0:
                            kp_x = x + np.random.randint(0, max(1, w))
                            kp_y = y + np.random.randint(0, max(1, h))
                        else:
                            kp_x = kp_y = 0
                        keypoints.extend([float(kp_x), float(kp_y), visibility])
                        if visibility == 2:
                            num_visible += 1
                    annotation["keypoints"] = keypoints
                    annotation["num_keypoints"] = num_visible
                
                annotations.append(annotation)
                ann_id += 1
        
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {"description": "Synthetic COCO dataset for testing"},
        }
        
        return coco_data

    def _create_predictions(self, coco_gt, iou_type="bbox", detection_rate=0.8):
        """
        Create synthetic predictions for a COCO dataset.
        
        Args:
            coco_gt: Ground truth COCO dataset dictionary
            iou_type: Type of predictions ('bbox', 'segm', or 'keypoints')
            detection_rate: Fraction of ground truth objects to detect
        
        Returns:
            List of prediction dictionaries
        """
        np.random.seed(123)  # Different seed for predictions
        
        predictions = []
        
        for ann in coco_gt["annotations"]:
            # Only detect a fraction of objects
            if np.random.random() > detection_rate:
                continue
            
            pred = {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
            }
            
            # Add score with some variation
            base_score = np.random.uniform(0.5, 0.99)
            pred["score"] = float(base_score)
            
            if iou_type in ["bbox", "segm"]:
                # Add some noise to bbox
                x, y, w, h = ann["bbox"]
                noise_factor = np.random.uniform(0.9, 1.1)
                pred["bbox"] = [
                    float(x + np.random.uniform(-2, 2)),
                    float(y + np.random.uniform(-2, 2)),
                    float(w * noise_factor),
                    float(h * noise_factor),
                ]
                pred["area"] = float(pred["bbox"][2] * pred["bbox"][3])
            
            if iou_type == "segm" and "segmentation" in ann:
                # Use ground truth segmentation with slight modification
                pred["segmentation"] = ann["segmentation"]
            
            if iou_type == "keypoints" and "keypoints" in ann:
                # Add noise to keypoint locations
                keypoints = []
                for i in range(0, len(ann["keypoints"]), 3):
                    kp_x, kp_y, v = ann["keypoints"][i:i+3]
                    if v > 0:
                        kp_x += np.random.uniform(-3, 3)
                        kp_y += np.random.uniform(-3, 3)
                    keypoints.extend([float(kp_x), float(kp_y), v])
                pred["keypoints"] = keypoints
            
            predictions.append(pred)
        
        # Add some false positives
        num_false_positives = int(len(predictions) * 0.1)
        for img in coco_gt["images"][:num_false_positives]:
            pred = {
                "image_id": img["id"],
                "category_id": np.random.randint(0, len(coco_gt["categories"])),
                "score": float(np.random.uniform(0.3, 0.7)),
            }
            
            if iou_type in ["bbox", "segm"]:
                w = np.random.randint(20, 100)
                h = np.random.randint(20, 100)
                x = np.random.randint(0, max(1, img["width"] - w))
                y = np.random.randint(0, max(1, img["height"] - h))
                pred["bbox"] = [float(x), float(y), float(w), float(h)]
                pred["area"] = float(w * h)
            
            if iou_type == "segm":
                # Create a dummy mask
                mask = np.zeros((img["height"], img["width"]), order="F", dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
                rle_mask = mask_util.encode(mask)
                rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
                pred["segmentation"] = rle_mask
            
            if iou_type == "keypoints":
                # Create dummy keypoints
                keypoints = []
                for i in range(17):
                    keypoints.extend([
                        float(np.random.randint(0, img["width"])),
                        float(np.random.randint(0, img["height"])),
                        2
                    ])
                pred["keypoints"] = keypoints
            
            predictions.append(pred)
        
        return predictions

    def _compare_evaluators(self, gt_file, predictions, iou_type, tolerance=1e-10):
        """
        Compare results from faster_coco_eval and pycocotools.
        
        Args:
            gt_file: Path to ground truth JSON file
            predictions: List of prediction dictionaries
            iou_type: Type of evaluation ('bbox', 'segm', or 'keypoints')
            tolerance: Tolerance for floating point comparison
        
        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: A tuple containing:
                - faster_coco_eval stats array
                - pycocotools stats array
                - boolean indicating if arrays are equal within tolerance
        """
        # Evaluate with faster_coco_eval
        coco_gt_fast = COCO(gt_file)
        coco_dt_fast = coco_gt_fast.loadRes(predictions)
        coco_eval_fast = COCOeval_faster(coco_gt_fast, coco_dt_fast, iou_type)
        coco_eval_fast.evaluate()
        coco_eval_fast.accumulate()
        coco_eval_fast.summarize()
        
        # Evaluate with pycocotools
        coco_gt_orig = origCOCO(gt_file)
        coco_dt_orig = coco_gt_orig.loadRes(predictions)
        coco_eval_orig = origCOCOeval(coco_gt_orig, coco_dt_orig, iou_type)
        coco_eval_orig.evaluate()
        coco_eval_orig.accumulate()
        coco_eval_orig.summarize()
        
        # Compare stats
        fast_stats = coco_eval_fast.stats
        orig_stats = coco_eval_orig.stats
        
        # Check if stats are equal within tolerance
        are_equal = np.allclose(fast_stats, orig_stats, rtol=tolerance, atol=tolerance)
        
        return fast_stats, orig_stats, are_equal

    @parameterized.expand([
        ("small_dataset", 10, 5, 5),
        ("medium_dataset", 50, 10, 10),
        ("large_dataset", 100, 20, 15),
    ])
    def test_bbox_detection_extensive(self, name, num_images, num_categories, anns_per_image):
        """Test bbox detection with various dataset sizes."""
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create dataset
        coco_data = self._create_coco_annotations(
            num_images=num_images,
            num_categories=num_categories,
            annotations_per_image=anns_per_image,
            include_segmentation=False,
            include_keypoints=False,
        )
        
        gt_file = osp.join(self.tmp_dir.name, f"gt_{name}.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Create predictions
        predictions = self._create_predictions(coco_data, iou_type="bbox")
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "bbox"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nDataset: {name} ({num_images} images, {len(coco_data['annotations'])} annotations, "
            f"{len(predictions)} predictions)\n"
            f"faster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )

    @parameterized.expand([
        ("small_dataset", 10, 5, 5),
        ("medium_dataset", 50, 10, 10),
        ("large_dataset", 100, 20, 15),
    ])
    def test_segmentation_extensive(self, name, num_images, num_categories, anns_per_image):
        """Test instance segmentation with various dataset sizes."""
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create dataset with segmentation
        coco_data = self._create_coco_annotations(
            num_images=num_images,
            num_categories=num_categories,
            annotations_per_image=anns_per_image,
            include_segmentation=True,
            include_keypoints=False,
        )
        
        gt_file = osp.join(self.tmp_dir.name, f"gt_{name}_segm.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Create predictions
        predictions = self._create_predictions(coco_data, iou_type="segm")
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "segm"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nDataset: {name} ({num_images} images, {len(coco_data['annotations'])} annotations, "
            f"{len(predictions)} predictions)\n"
            f"faster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )

    @parameterized.expand([
        ("small_dataset", 10, 3, 5),
        ("medium_dataset", 50, 5, 8),
        ("large_dataset", 100, 10, 10),
    ])
    def test_keypoints_extensive(self, name, num_images, num_categories, anns_per_image):
        """Test keypoint detection with various dataset sizes."""
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create dataset with keypoints
        coco_data = self._create_coco_annotations(
            num_images=num_images,
            num_categories=num_categories,
            annotations_per_image=anns_per_image,
            include_segmentation=False,
            include_keypoints=True,
        )
        
        gt_file = osp.join(self.tmp_dir.name, f"gt_{name}_kpts.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Create predictions
        predictions = self._create_predictions(coco_data, iou_type="keypoints")
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "keypoints"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nDataset: {name} ({num_images} images, {len(coco_data['annotations'])} annotations, "
            f"{len(predictions)} predictions)\n"
            f"faster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )

    def test_edge_case_no_predictions(self):
        """Test evaluation with no predictions.
        
        Note: Both pycocotools and faster_coco_eval have issues with truly empty
        prediction lists (loadRes() fails on empty lists when trying to inspect the
        first element to determine annotation type). This is a known limitation in
        the COCO API design. We use a very low-scoring prediction instead to test
        the low-prediction scenario.
        """
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create dataset
        coco_data = self._create_coco_annotations(
            num_images=10,
            num_categories=5,
            annotations_per_image=5,
        )
        
        gt_file = osp.join(self.tmp_dir.name, "gt_no_preds.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Use a very low score prediction instead of empty list
        # (Both APIs crash on truly empty prediction lists)
        predictions = [{
            "image_id": coco_data["images"][0]["id"],
            "category_id": 0,
            "bbox": [10.0, 10.0, 10.0, 10.0],
            "area": 100.0,
            "score": 0.01,  # Very low score to simulate near-empty results
        }]
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "bbox"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nfaster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )

    def test_edge_case_perfect_predictions(self):
        """Test evaluation with perfect predictions (all IOU=1.0)."""
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create small dataset
        coco_data = self._create_coco_annotations(
            num_images=10,
            num_categories=3,
            annotations_per_image=5,
        )
        
        gt_file = osp.join(self.tmp_dir.name, "gt_perfect.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Create perfect predictions (identical to ground truth)
        predictions = []
        for ann in coco_data["annotations"]:
            pred = {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "score": 1.0,
            }
            predictions.append(pred)
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "bbox"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nfaster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )

    def test_mixed_object_sizes(self):
        """Test evaluation with mixed small/medium/large objects."""
        if origCOCO is None:
            raise unittest.SkipTest("pycocotools not available")
        
        # Create dataset with controlled object sizes
        coco_data = self._create_coco_annotations(
            num_images=50,
            num_categories=10,
            annotations_per_image=15,
        )
        
        gt_file = osp.join(self.tmp_dir.name, "gt_mixed_sizes.json")
        with open(gt_file, "w") as f:
            json.dump(coco_data, f)
        
        # Create predictions
        predictions = self._create_predictions(coco_data, iou_type="bbox")
        
        # Compare evaluators
        fast_stats, orig_stats, are_equal = self._compare_evaluators(
            gt_file, predictions, "bbox"
        )
        
        # Assert equality
        self.assertTrue(
            are_equal,
            f"\nfaster_coco_eval stats: {fast_stats}\n"
            f"pycocotools stats:      {orig_stats}\n"
            f"Difference: {fast_stats - orig_stats}"
        )
        
        # Also verify that we have metrics for different size categories
        # Stats indices: [mAP, mAP@50, mAP@75, mAP_small, mAP_medium, mAP_large, ...]
        self.assertGreaterEqual(len(fast_stats), 6)


if __name__ == "__main__":
    unittest.main()
