#!/usr/bin/python3

import unittest

import numpy as np

import faster_coco_eval.core.mask as mask_util


class TestBoundary(unittest.TestCase):
    def setUp(self):
        # fmt: off
        segm = [
            [0, 0, 15, 20, 20, 10, 20, 30, 20, 10, 10,
             10, 50, 50, 70, 60, 60, 60, 40, 50, 10, 60, 0, 0],
            [50, 20, 70, 20, 70, 40, 50, 20],
        ]

        mini_mask = np.array([
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0]],
            dtype=np.uint8,
        )

        self.mini_mask_boundry = np.array([
            [[0], [0], [1], [1], [1]],
            [[0], [1], [1], [1], [0]],
            [[0], [1], [0], [1], [0]],
            [[0], [1], [1], [1], [1]],
            [[0], [0], [0], [1], [0]]],
            dtype=np.uint8,
        )
        # fmt: on
        self.mini_mask_rle = mask_util.encode(mini_mask)
        self.rle_80_70 = mask_util.segmToRle(segm, 80, 70)

    def test_rleToBoundary(self):
        mask_api_rle = mask_util.rleToBoundary(
            self.rle_80_70, backend="mask_api"
        )
        opencv_rle = mask_util.rleToBoundary(self.rle_80_70, backend="opencv")
        self.assertDictEqual(mask_api_rle, opencv_rle)

        mask_api_rle_mask = mask_util.decode(
            [mask_util.rleToBoundary(self.mini_mask_rle, backend="mask_api")]
        )
        opencv_rle_mask = mask_util.decode(
            [mask_util.rleToBoundary(self.mini_mask_rle, backend="opencv")]
        )

        self.assertTrue(
            np.array_equal(mask_api_rle_mask, self.mini_mask_boundry)
        )
        self.assertTrue(np.array_equal(opencv_rle_mask, self.mini_mask_boundry))


if __name__ == "__main__":
    unittest.main()
