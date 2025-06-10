#!/usr/bin/python3
import unittest

import numpy as np
from parameterized import parameterized

import faster_coco_eval.faster_eval_api_cpp as _C


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


if __name__ == "__main__":
    unittest.main()
