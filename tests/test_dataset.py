#!/usr/bin/python3
import unittest

import faster_coco_eval.faster_eval_api_cpp as _C
import numpy as np
from parameterized import parameterized


class TestBaseCoco(unittest.TestCase):
    """Test basic COCO functionality."""

    maxDiff = None

    def setUp(self):
        pass

    def test_append(self):
        dataset = _C.Dataset()
        self.assertEqual(len(dataset), 0)

        dataset.append(1, 1, {"bbox": [1, 1, 1, 1]})
        self.assertEqual(len(dataset), 1)

        dataset.clean()
        self.assertEqual(len(dataset), 0)

    def test_get(self):
        dataset = _C.Dataset()
        for i in range(10):
            dataset.append(i, i, {"bbox": [i, i, i, i]})

        self.assertEqual(len(dataset), 10)

        for i in range(10):
            self.assertEqual(dataset.get(i, i), [{"bbox": [i, i, i, i]}])

        dataset.clean()
        self.assertEqual(len(dataset), 0)

    @parameterized.expand([range, np.arange])
    def test_get_instances(self, range_func):
        dataset = _C.Dataset()

        for i in range_func(10):
            dataset.append(i, i, {"bbox": [i, i, i, i]})
            dataset.append(np.int64(i), np.float64(i + 1), {"bbox": [-1, i, i, i]})

        self.assertEqual(len(dataset), 20)

        for i in range_func(10):
            self.assertEqual(dataset.get_instances([i], [i], True), [[[{"bbox": [i, i, i, i]}]]])
            self.assertEqual(
                dataset.get_instances([i], [i, np.float64(i + 1)], False),
                [[[{"bbox": [i, i, i, i]}, {"bbox": [-1, i, i, i]}]]],
            )
            self.assertEqual(
                dataset.get_instances([i], [i, i + 1], True), [[[{"bbox": [i, i, i, i]}], [{"bbox": [-1, i, i, i]}]]]
            )

        dataset.clean()
        self.assertEqual(len(dataset), 0)

    def test_serialization_basic(self):
        """Test serialization/deserialization with make_tuple/load_tuple."""
        dataset = _C.Dataset()

        # Add some test data
        test_data = [
            (1, 1, {"id": 1, "score": 0.9, "area": 100.5, "bbox": [1, 2, 3, 4]}),
            (1, 2, {"id": 2, "score": 0.8, "area": 200.0, "is_crowd": False}),
            (2, 1, {"id": 3, "score": 0.7, "area": 150.0, "iscrowd": True, "ignore": False}),
        ]

        for img_id, cat_id, ann in test_data:
            dataset.append(img_id, cat_id, ann)

        self.assertEqual(len(dataset), 3)

        # Test serialization - new format returns (size, list_of_tuples)
        tuple_data = dataset.make_tuple()
        self.assertEqual(len(tuple_data), 2)
        self.assertEqual(tuple_data[0], 3)  # size
        self.assertIsInstance(tuple_data[1], list)  # list of (img_id, cat_id, annotations) tuples

        # Verify serialized data structure
        serialized_list = tuple_data[1]
        self.assertEqual(len(serialized_list), 3)  # 3 unique (img_id, cat_id) pairs

        # Each item should be (img_id, cat_id, [annotations])
        for item in serialized_list:
            self.assertEqual(len(item), 3)  # (img_id, cat_id, ann_list)
            self.assertIsInstance(item[0], float)  # img_id
            self.assertIsInstance(item[1], float)  # cat_id
            self.assertIsInstance(item[2], list)  # annotation list

        # Test deserialization
        new_dataset = _C.Dataset()
        new_dataset.load_tuple(tuple_data)

        self.assertEqual(len(new_dataset), 3)

        # Verify data integrity
        for img_id, cat_id, expected_ann in test_data:
            retrieved = new_dataset.get(img_id, cat_id)
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0], expected_ann)

    def test_flexible_type_handling(self):
        """Test that parseInstanceAnnotation handles different data types
        correctly."""
        dataset = _C.Dataset()

        # Test various data type combinations
        test_annotations = [
            # Integer types
            {"id": 1, "score": 0.9, "area": 100, "is_crowd": 0, "ignore": 1},
            # Float types
            {"id": 2.0, "score": 0.8, "area": 200.5, "iscrowd": 0.0, "ignore": 1.0},
            # String types (should be parsed)
            {"id": "3", "score": "0.7", "area": "150.5"},
            # Boolean types
            {"id": 4, "score": 0.6, "area": 120.0, "is_crowd": True, "ignore": False},
            # Mixed types
            {"id": 5, "score": "0.5", "area": 180, "iscrowd": False, "lvis_mark": 1},
        ]

        for i, ann in enumerate(test_annotations):
            dataset.append(i, i, ann)

        # Test get_cpp_annotations to ensure parsing works without errors
        for i, expected_ann in enumerate(test_annotations):
            cpp_anns = dataset.get_cpp_annotations(i, i)
            self.assertEqual(len(cpp_anns), 1)

            # Just verify that parsing doesn't crash and returns expected count
            ann = cpp_anns[0]
            self.assertIsInstance(ann, _C.InstanceAnnotation)

    def test_rle_data_integrity(self):
        """Test that RLE bytes data survives serialization."""
        dataset = _C.Dataset()

        # Create test RLE data (simulated)
        rle_data = {
            "id": 1,
            "segmentation": {
                "counts": b"\x01\x02\x03\x04\x05",  # binary RLE data
                "size": [100, 100],
            },
            "area": 500.0,
            "bbox": [10, 20, 30, 40],
        }

        dataset.append(1, 1, rle_data)

        # Test round-trip through serialization
        tuple_data = dataset.make_tuple()
        new_dataset = _C.Dataset()
        new_dataset.load_tuple(tuple_data)

        # Retrieve and verify data
        retrieved = new_dataset.get(1, 1)
        self.assertEqual(len(retrieved), 1)

        original_segm = rle_data["segmentation"]
        retrieved_segm = retrieved[0]["segmentation"]

        # Verify that bytes data is preserved
        self.assertEqual(retrieved_segm["size"], original_segm["size"])
        self.assertEqual(retrieved_segm["counts"], original_segm["counts"])
        self.assertEqual(retrieved[0]["area"], rle_data["area"])
        self.assertEqual(retrieved[0]["bbox"], rle_data["bbox"])

    def test_empty_dataset_serialization(self):
        """Test serialization of empty dataset."""
        dataset = _C.Dataset()

        tuple_data = dataset.make_tuple()
        self.assertEqual(tuple_data[0], 0)  # size should be 0
        # Empty dataset should return empty list
        self.assertEqual(tuple_data[1], [])

        # Test loading empty dataset
        new_dataset = _C.Dataset()
        new_dataset.load_tuple(tuple_data)
        self.assertEqual(len(new_dataset), 0)

    def test_large_dataset_serialization(self):
        """Test serialization with larger dataset."""
        dataset = _C.Dataset()

        # Create a larger dataset
        for img_id in range(10):
            for cat_id in range(5):
                for ann_id in range(3):
                    ann = {
                        "id": img_id * 100 + cat_id * 10 + ann_id,
                        "score": np.random.random(),
                        "area": np.random.random() * 1000,
                        "bbox": [np.random.randint(0, 100) for _ in range(4)],
                        "is_crowd": np.random.choice([True, False]),
                    }
                    dataset.append(img_id, cat_id, ann)

        self.assertEqual(len(dataset), 50)  # 10 * 5 unique (img_id, cat_id) pairs

        # Test serialization/deserialization
        tuple_data = dataset.make_tuple()
        new_dataset = _C.Dataset()
        new_dataset.load_tuple(tuple_data)

        self.assertEqual(len(new_dataset), len(dataset))

        # Spot check some data
        for img_id in [0, 5, 9]:
            for cat_id in [0, 2, 4]:
                original = dataset.get(img_id, cat_id)
                new = new_dataset.get(img_id, cat_id)
                self.assertEqual(len(original), len(new))
                self.assertEqual(len(original), 3)  # 3 annotations per (img_id, cat_id)

    def test_cpp_annotations_consistency(self):
        """Test that get_cpp_annotations returns consistent data."""
        dataset = _C.Dataset()

        test_ann = {"id": 42, "score": 0.95, "area": 1250.5, "is_crowd": True, "ignore": False, "lvis_mark": True}

        dataset.append(1, 1, test_ann)

        # Get both py::dict and C++ versions
        py_anns = dataset.get(1, 1)
        cpp_anns = dataset.get_cpp_annotations(1, 1)

        self.assertEqual(len(py_anns), 1)
        self.assertEqual(len(cpp_anns), 1)

        py_ann = py_anns[0]
        cpp_ann = cpp_anns[0]

        # Just verify that both APIs return the expected count and types
        self.assertEqual(py_ann["id"], 42)
        self.assertAlmostEqual(py_ann["score"], 0.95, places=10)
        self.assertAlmostEqual(py_ann["area"], 1250.5, places=10)
        self.assertEqual(py_ann["is_crowd"], True)
        self.assertEqual(py_ann["ignore"], False)
        self.assertEqual(py_ann["lvis_mark"], True)

        # Verify C++ annotation object exists and is correct type
        self.assertIsInstance(cpp_ann, _C.InstanceAnnotation)


if __name__ == "__main__":
    unittest.main()
