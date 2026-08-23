#!/usr/bin/python3

import sys
import threading
import unittest

import faster_coco_eval.mask_api_new_cpp as _mask
import numpy as np
from parameterized import parameterized

import faster_coco_eval.core.mask as mask_util
from faster_coco_eval import COCO


def _encode(x):
    """Encode a binary mask into a run-length encoded string."""
    return mask_util.encode(np.asfortranarray(x, np.uint8))


class _ListSubclass(list):
    """Exercise native dispatch with a Python list subclass."""


class _DictSubclass(dict):
    """Exercise native dispatch with a Python dict subclass."""


class _ArraySubclass(np.ndarray):
    """Exercise native dispatch with a NumPy array subclass."""


class TestMaskApi(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.rleObjs = []
        self.areas = []
        for h in range(64, 256 + 64, 64):
            for w in range(64, 256 + 64, 64):
                x = np.random.randint(0, 10, size=(h, w, 1), dtype=np.uint8)
                x[x < 4] = 0
                x[x >= 4] = 1
                self.areas.append(np.sum(x))
                self.rleObjs += _mask.encode(np.asfortranarray(x, np.uint8))

        self.bboxes = np.array([
            [3.0, 5.0, 4.0, 1.0],
            [5.0, 3.0, 3.0, 5.0],
            [8.0, 5.0, 8.0, 5.0],
            [3.0, 4.0, 7.0, 3.0],
            [2.0, 5.0, 2.0, 7.0],
            [8.0, 1.0, 9.0, 9.0],
            [8.0, 1.0, 9.0, 6.0],
            [7.0, 1.0, 9.0, 4.0],
            [4.0, 5.0, 4.0, 5.0],
            [2.0, 4.0, 5.0, 4.0],
        ])

        self.bbox_rles = [
            {"size": [20, 20], "counts": b"Q21c000000o7"},
            {"size": [20, 20], "counts": b"W35?000]7"},
            {"size": [20, 20], "counts": b"U55?0000000000000[2"},
            {"size": [20, 20], "counts": b"P23a000000000000T6"},
            {"size": [20, 20], "counts": b"]17=0k9"},
            {"size": [20, 20], "counts": b"Q59;000000000000000k1"},
            {"size": [20, 20], "counts": b"Q56>000000000000000k1"},
            {"size": [20, 20], "counts": b"]44`0000000000000000_2"},
            {"size": [20, 20], "counts": b"e25?00000[7"},
            {"size": [20, 20], "counts": b"\\14`00000000P8"},
        ]
        # fmt: off
        self.poly = np.array([
            [7., 8., 9., 3., 7., 1., 5., 1., 5., 8., 3., 4., 4., 7., 7., 8.],
            [4., 4., 7., 5., 5., 9., 3., 1., 2., 3., 1., 6., 6., 2., 2., 9.],
            [5., 3., 7., 1., 2., 5., 4., 9., 6., 6., 4., 8., 2., 9., 7., 4.],
            [5., 1., 1., 4., 7., 9., 2., 3., 6., 9., 7., 4., 3., 8., 1., 2.],
            [7., 5., 5., 8., 5., 3., 8., 8., 2., 6., 8., 7., 4., 6., 2., 6.],
            [8., 8., 4., 7., 6., 3., 9., 4., 9., 4., 7., 8., 7., 8., 2., 9.],
            [2., 8., 8., 6., 5., 8., 4., 8., 9., 4., 8., 3., 4., 5., 9., 7.],
            [4., 1., 8., 3., 8., 7., 9., 1., 3., 7., 5., 4., 9., 1., 9., 7.],
            [1., 4., 2., 9., 3., 4., 5., 5., 1., 4., 2., 1., 3., 1., 8., 8.],
            [6., 6., 4., 5., 9., 9., 1., 9., 8., 6., 5., 4., 3., 4., 7., 1.]
        ])
        # fmt: on

        self.bbox_rles_merged = {
            "size": [20, 20],
            "counts": b"\\18<00N1100N2000000000000000000k1",
        }
        self.bbox_rles_merged_1 = {"size": [20, 20], "counts": b"`<"}

        self.poly_rles = [
            {"size": [20, 20], "counts": b"U36>1OO2Lm6"},
            {"size": [20, 20], "counts": b"h02`012N_O0`00@1>OB3`0MQ8"},
            {"size": [20, 20], "counts": b"]1120=210AN>1BOT9"},
            {"size": [20, 20], "counts": b"f01c01^OO>0E0R11[O0AO]8"},
            {"size": [20, 20], "counts": b"e22a0O@0b0010X7"},
            {"size": [20, 20], "counts": b"f2110>212OO2Mj6"},
            {"size": [20, 20], "counts": b"S21c00M1AO?110@O?0l6"},
            {"size": [20, 20], "counts": b"V31c010000g6"},
            {"size": [20, 20], "counts": b"f04?OBN?131O01N10X7"},
            {"size": [20, 20], "counts": b"`11c00K0E1:0F090F2a0MX7"},
        ]

        self.uncompressed_rle = {
            "size": [1350, 1080],
            "counts": [0, 5, 5, 5, 5, 2, 3, 5, 2, 3, 5, 2, 3, 55],  # noqa: E501
        }
        self.compressed_rle = {
            "size": [1350, 1080],
            "counts": b"05500MN3ON3ONe1",
        }

    def test_frString(self):
        c_rle = _mask._frString(self.poly_rles)
        py_rle = _mask._toString(c_rle)
        for i in range(len(py_rle)):
            self.assertDictEqual(py_rle[i], self.poly_rles[i])

    @parameterized.expand([_mask, mask_util])
    def test_area(self, module):
        areas = module.area(self.rleObjs)
        self.assertEqual(areas.tolist(), self.areas)

    def test_area_solo(self):
        area = mask_util.area(self.rleObjs[0])
        self.assertEqual(area, self.areas[0])

    def test_rles(self):
        self.assertTrue(np.all([_mask.encode(_mask.decode([rle])) == [rle] for rle in self.rleObjs]))

    @staticmethod
    def _parallel_batch_masks(count=16, shape=(32, 32), seed=7):
        """Deterministic masks sized to cross the parallel batch threshold."""
        rng = np.random.RandomState(seed)
        masks = []
        for i in range(count):
            mask = np.zeros(shape, dtype=np.uint8)
            mask[4 + i : 4 + i + 8, 3 + i : 3 + i + 9] = 1
            mask[rng.rand(*shape) < 0.1] ^= 1
            masks.append(mask)
        return masks

    @staticmethod
    def _erode_reference(mask, dilation):
        """Dense reference for 3x3-style erosion used by the parallel path."""
        h, w = mask.shape
        padded = np.pad(mask, dilation, mode="constant")
        eroded = np.ones_like(mask)
        for dy in range(2 * dilation + 1):
            for dx in range(2 * dilation + 1):
                eroded &= padded[dy : dy + h, dx : dx + w]
        return eroded

    def test_parallel_batch_encode_parity_and_order(self):
        """Batch encode must match per-mask encode and keep slot order."""
        masks = self._parallel_batch_masks()
        stacked = np.asfortranarray(np.stack(masks, axis=2))

        batch = _mask.encode(stacked)
        solo = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]

        self.assertEqual(batch, solo)

    def test_parallel_batch_decode_parity_and_order(self):
        """Batch decode must match per-mask decode and keep slot order."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]

        decoded = _mask.decode(rles)

        self.assertEqual(decoded.shape, (32, 32, len(masks)))
        for index, mask in enumerate(masks):
            np.testing.assert_array_equal(decoded[:, :, index], mask)

    def test_parallel_batch_area_parity_and_order(self):
        """Batch area must match per-mask areas in order."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]

        areas = _mask.area(rles)

        self.assertEqual(areas.tolist(), [int(m.sum()) for m in masks])

    def test_parallel_batch_erode_parity_and_order(self):
        """Batch erosion must match the dense reference and keep slot order."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]

        eroded = _mask.erode_3x3(rles, 1)
        decoded = _mask.decode(eroded)

        for index, mask in enumerate(masks):
            np.testing.assert_array_equal(decoded[:, :, index], self._erode_reference(mask, 1))

    def test_parallel_batch_decode_propagates_malformed_item(self):
        """A malformed item in a parallel batch must surface its error."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]
        rles[5] = {"size": [32, 32], "counts": b"0"}  # runs sum below h*w

        with self.assertRaises(ValueError):
            _mask.decode(rles)

    def test_parallel_batch_erode_propagates_malformed_item(self):
        """A malformed item must fail erode_3x3 even in a parallel batch."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]
        rles[3] = {"size": [2, 2], "counts": b"05"}  # runs exceed h*w

        with self.assertRaises(ValueError):
            _mask.erode_3x3(rles, 1)

    def test_parallel_batch_area_propagates_malformed_item(self):
        """A malformed item must fail area even in a parallel batch."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]
        rles[2] = {"size": [2, 2], "counts": b"05"}  # runs exceed h*w

        with self.assertRaises(ValueError):
            _mask.area(rles)

    def test_parallel_batch_decode_multiple_malformed_items(self):
        """Multiple malformed items must still surface an error
        deterministically."""
        masks = self._parallel_batch_masks()
        rles = [mask_util.encode(np.asfortranarray(m[..., None]))[0] for m in masks]
        # Two undersized items in different chunks of a parallel batch.
        rles[1] = {"size": [32, 32], "counts": b"0"}
        rles[9] = {"size": [32, 32], "counts": b"0"}

        with self.assertRaises(ValueError):
            _mask.decode(rles)

    def test_decode_rejects_count_larger_than_each_mask(self):
        """Reject oversized uncompressed RLE counts at construction time."""
        with self.assertRaises(ValueError):
            _mask.frUncompressedRLE([
                {"size": [2, 2], "counts": [0, 4]},
                {"size": [2, 2], "counts": [0, 5]},
            ])

    def test_decode_rejects_count_smaller_than_mask(self):
        """Reject a second RLE whose runs sum to fewer pixels than h*w."""
        valid, undersized = _mask.frUncompressedRLE([
            {"size": [2, 2], "counts": [0, 4]},
            {"size": [2, 2], "counts": [0, 3]},
        ])

        with self.assertRaises(ValueError):
            _mask.decode([valid, undersized])

    def test_rejects_oversized_rle_before_mask_operations(self):
        """Reject malformed RLE counts at every dense-mask entry point."""
        oversized = {"size": [2, 2], "counts": b"05"}

        with self.assertRaises(ValueError):
            _mask.decode([oversized])
        with self.assertRaises(ValueError):
            _mask.merge([oversized, oversized])
        with self.assertRaises(ValueError):
            _mask.erode_3x3([oversized], 1)

    def test_rle_rejects_count_length_mismatch(self):
        """Reject an explicit run count length that disagrees with the data."""
        with self.assertRaises(ValueError):
            _mask.RLE(2, 2, 3, [0, 4])

    def test_fr_poly_preserves_64_bit_pixel_offsets(self):
        """Keep polygon RLE offsets above the uint32 range intact."""
        height = 2**32 + 1
        rle = _mask.frPoly([[0.0, 0.0, 0.0, 1.0]], height, 1)[0]

        uncompressed = _mask.toUncompressedRLE([rle])[0]

        self.assertEqual(sum(uncompressed["counts"]), height)

    def test_frBbox(self):
        self.assertEqual(self.bbox_rles, _mask.frBbox(self.bboxes, 20, 20))

    def test_frPoly(self):
        self.assertEqual(self.poly_rles, _mask.frPoly(self.poly, 20, 20))

    def test_frUncompressedRLE(self):
        self.assertEqual(
            self.compressed_rle,
            _mask.frUncompressedRLE([self.uncompressed_rle])[0],
        )

    @parameterized.expand([_mask, mask_util])
    def test_frPyObjects(self, module):
        self.assertEqual(
            self.poly_rles,
            module.frPyObjects([p for p in self.poly], 20, 20),
        )
        self.assertEqual(self.bbox_rles, module.frPyObjects(self.bboxes, 20, 20))

        self.assertEqual(
            self.compressed_rle,
            module.frPyObjects(self.uncompressed_rle, 1350, 1080),
        )
        self.assertEqual(
            self.compressed_rle,
            module.frPyObjects([self.uncompressed_rle], 1350, 1080)[0],
        )

    @parameterized.expand([_mask, mask_util])
    def test_merge(self, module):
        self.assertEqual(self.bbox_rles[0], module.merge([self.bbox_rles[0]]))
        self.assertEqual(self.bbox_rles_merged, module.merge(self.bbox_rles))
        self.assertEqual(self.bbox_rles_merged, module.merge(self.bbox_rles, 0))
        self.assertEqual(self.bbox_rles_merged_1, module.merge(self.bbox_rles, 1))

    @parameterized.expand([_mask, mask_util])
    def test_toBbox(self, module):
        self.assertEqual(self.bboxes.tolist(), module.toBbox(self.bbox_rles).tolist())

    def test_toBbox_solo(self):
        self.assertEqual(
            self.bboxes.tolist()[0],
            mask_util.toBbox(self.bbox_rles[0]).tolist(),
        )

    @parameterized.expand([_mask, mask_util])
    def test_iou(self, module):
        iou_11 = np.array([[1.0, 0.5], [0.13333333, 1.0]]).round(4)

        result_iou_11 = module.iou(self.bbox_rles[:2], self.bbox_rles[:2], [1, 1]).round(4)

        self.assertEqual(iou_11.tolist(), result_iou_11.tolist())

        iou_00 = np.array([[1.0, 0.11764706], [0.11764706, 1.0]]).round(4)
        result_iou_00 = module.iou(self.bbox_rles[:2], self.bbox_rles[:2], [0, 0]).round(4)

        self.assertEqual(iou_00.tolist(), result_iou_00.tolist())

        iou_10 = np.array([[1.0, 0.11764706], [0.13333333, 1.0]]).round(4)
        result_iou_10 = module.iou(self.bbox_rles[:2], self.bbox_rles[:2], [1, 0]).round(4)

        self.assertEqual(iou_10.tolist(), result_iou_10.tolist())

        iou_01 = np.array([[1.0, 0.5], [0.11764706, 1.0]]).round(4)
        result_iou_01 = module.iou(self.bbox_rles[:2], self.bbox_rles[:2], [0, 1]).round(4)
        self.assertEqual(iou_01.tolist(), result_iou_01.tolist())

        poly_iou = np.array([[1.0, 0.1562, 0.1], [0.1562, 1.0, 0.2174], [0.1, 0.2174, 1.0]])

        result_poly_iou = module.iou(self.poly_rles[:3], self.poly_rles[:3], [0, 0, 0]).round(4)
        self.assertEqual(poly_iou.tolist(), result_poly_iou.tolist())

    def test_iou_accepts_list_and_array_subclasses(self):
        """Dispatch bounding boxes from list and NumPy array subclasses."""
        expected = _mask.iou(self.bboxes[:2], self.bboxes[:2], [0, 0])
        list_boxes = _ListSubclass(
            [_ListSubclass(box) for box in self.bboxes[:2].tolist()],
        )
        array_boxes = self.bboxes[:2].view(_ArraySubclass)
        rle_list = _ListSubclass([_DictSubclass(self.compressed_rle)])

        np.testing.assert_array_equal(_mask.iou(list_boxes, list_boxes, [0, 0]), expected)
        np.testing.assert_array_equal(_mask.iou(array_boxes, array_boxes, [0, 0]), expected)
        np.testing.assert_array_equal(_mask.iou(rle_list, rle_list, [0]), np.ones((1, 1)))

    def test_fr_py_objects_accepts_list_and_dict_subclasses(self):
        """Dispatch segmentation input from list and dict subclasses."""
        rle = _DictSubclass(self.uncompressed_rle)

        self.assertEqual(
            self.compressed_rle,
            _mask.frPyObjects(_ListSubclass([rle]), 1350, 1080)[0],
        )
        self.assertEqual(self.compressed_rle, _mask.frPyObjects(rle, 1350, 1080))
        self.assertEqual(
            self.bbox_rles[:2],
            _mask.frPyObjects(self.bboxes[:2].view(_ArraySubclass), 20, 20),
        )

    def test_iou_releases_gil_during_cpp_compute(self):
        """Allow another Python thread to run during native IoU computation."""
        rows, columns = np.indices((256, 256))
        checkerboard = np.asfortranarray(((rows + columns) % 2).astype(np.uint8)[..., None])
        encoded = _mask.encode(checkerboard)[0]
        rles = [encoded] * 24
        started = threading.Event()
        finished = threading.Event()
        errors = []

        def compute_iou():
            started.set()
            try:
                _mask.iou(rles, rles, [0] * len(rles))
            except Exception as ex:  # pragma: no cover - surfaced below
                errors.append(ex)
            finally:
                finished.set()

        original_switch_interval = sys.getswitchinterval()
        worker = threading.Thread(target=compute_iou)
        observed_start = False
        overlapped = False
        try:
            # Prevent a Python bytecode switch between started.set() and iou().
            # The main thread can resume before completion only if native iou()
            # releases the GIL.
            sys.setswitchinterval(1.0)
            worker.start()
            observed_start = started.wait(timeout=1.0)
            overlapped = observed_start and not finished.is_set()
            worker.join(timeout=5.0)
        finally:
            sys.setswitchinterval(original_switch_interval)

        self.assertTrue(observed_start, "IoU worker did not start")
        self.assertFalse(worker.is_alive(), "IoU worker did not finish")
        if errors:
            raise errors[0]
        self.assertTrue(overlapped, "Python thread could not run while native IoU was active")

    def test_iou_size_mismatch_writes_sentinel_to_pair(self):
        """Return -1 in each pair's output cell when RLE sizes differ."""
        dt = _mask.frBbox([[0, 0, 2, 2]], 4, 4) * 2
        gt = [
            _mask.frBbox([[0, 0, 2, 2]], 4, 4)[0],
            _mask.frBbox([[0, 0, 2, 2]], 5, 5)[0],
        ]

        result = _mask.iou(dt, gt, [0, 0])

        np.testing.assert_array_equal(result, [[1.0, -1.0], [1.0, -1.0]])

    def test_iou_size_mismatch_rectangular_output_stays_in_bounds(self):
        """Return one sentinel per mismatched pair for a rectangular result."""
        dt = _mask.frBbox([[0, 0, 2, 2]], 4, 4)
        gt = _mask.frBbox([[0, 0, 2, 2]] * 5, 5, 5)

        result = _mask.iou(dt, gt, [0] * 5)

        np.testing.assert_array_equal(result, np.full((1, 5), -1.0))

    def testToBboxFullImage(self):
        mask = np.array([[0, 1], [1, 1]])
        bbox = mask_util.toBbox(_encode(mask))
        self.assertTrue((bbox == np.array([0, 0, 2, 2], dtype="float32")).all(), bbox)

    def testToBboxNonFullImage(self):
        mask = np.zeros((10, 10, 1), dtype=np.uint8)
        mask[2:4, 3:6, :] = 1
        bbox = mask_util.toBbox(_encode(mask)[0])
        self.assertTrue((bbox == np.array([3, 2, 3, 2], dtype="float32")).all(), bbox)

    def testInvalidRLECounts(self):
        rle = {
            "size": [1024, 1024],
            "counts": "jd`0=`o06J5L4M3L3N2N2N2N2N1O2N2N101N1O2O0O1O2N100O1O2N100O1O1O1O1O101N1O1O1O1O1O1O101N1O100O101O0O100000000000000001O00001O1O0O2O1N3N1N3N3L5Kh0XO6J4K5L5Id[o5N]dPJ7K4K4M3N2M3N2N1O2N100O2O0O1000O01000O101N1O1O2N2N2M3M3M4J7Inml5H[RSJ6L2N2N2N2O000000000000O2O1N2N2Mkm81SRG6L3L3N2O1N2N2O0O2O00001O0000000000O2O001N2O0O2N2N3M3L5JRjf6MPVYI8J4L3N3M2N1O2O1N101N1000000O10000001O000O101N101N1O2N2N2N3L4L7FWZ_50ne`J0000001O000000001O0000001O1O0N3M3N1O2N2N2O1N2O001N2`RO^O`k0c0[TOEak0;\\\\TOJbk07\\\\TOLck03[TO0dk01ZTO2dk0OYTO4gk0KXTO7gk0IXTO8ik0HUTO:kk0ETTO=lk0CRTO>Pl0@oSOb0Rl0\\\\OmSOe0Tl0[OjSOg0Ul0YOiSOi0Wl0XOgSOi0Yl0WOeSOk0[l0VOaSOn0kh0cNmYO",  # noqa: E501
        }
        with self.assertRaises(ValueError):
            mask_util.decode(rle)

    def testZeroLeadingRLE(self):
        # A foreground segment of length 0 was not previously handled correctly.
        # This input rle has 3 leading zeros.
        rle = {
            "size": [1350, 1080],
            "counts": "000lg0Zb01O00001O00001O001O00001O00001O001O00001O01O2N3M3M3M2N3M3N2M3M2N1O1O1O1O2N1O1O1O2N1O1O101N1O1O1O2N1O1O1O2N3M2N1O2N1O2O0O2N1O1O2N1O2N1O2N1O2N1O2N1O2O0O2N1O3M2N1O2N2N2N2N2N1O2N2N2N2N1O2N2N2N2N2N1N3N2N00O1O1O1O100000000000000O100000000000000001O0000001O00001O0O5L7I5K4L4L3M2N2N2N1O2m]OoXOm`0Sg0j^OVYOTa0lf0c^O]YO[a0ef0\\^OdYOba0bg0N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N2N3M2M4M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2N3M2M4M2N2N3M2M4M2N2N3M2M3N3M2N3M2M3N3M2N2N3L3N2N3M2N3L3N2N3M5J4M3M4L3M3L5L3M3M4L3L4\\EXTOd6jo0K6J5K6I4M1O1O1O1N2O1O1O001N2O00001O0O101O000O2O00001N101N101N2N101N101N101N2O0O2O0O2O0O2O1N101N2N2O1N2O1N2O1N101N2O1N2O1N2O0O2O1N2N2O1N2O0O2O1N2O1N2N2N1N4M2N2M4M2N3L3N2N3L3N3L3N2N3L3N2N3L3M4L3M3M4L3M5K5K5K6J5K5K6J7I7I7Ibijn0",  # noqa: E501
        }
        orig_bbox = mask_util.toBbox(rle)
        mask = mask_util.decode(rle)
        rle_new = mask_util.encode(mask)
        new_bbox = mask_util.toBbox(rle_new)
        self.assertTrue(np.equal(orig_bbox, new_bbox).all())

        orig_bbox = mask_util.toBbox(rle)
        masks = mask_util.decode([rle])
        rles_new = mask_util.encode(masks)
        new_bboxs = mask_util.toBbox(rles_new)
        self.assertTrue(np.equal(orig_bbox, new_bboxs[0]).all())

    def testSegmToRle(self):
        new_rle = mask_util.segmToRle(
            self.uncompressed_rle,
            self.uncompressed_rle["size"][0],
            self.uncompressed_rle["size"][1],
        )
        self.assertDictEqual(new_rle, self.compressed_rle)

        new_rle = mask_util.segmToRle(
            self.compressed_rle,
            self.compressed_rle["size"][0],
            self.compressed_rle["size"][1],
        )
        self.assertDictEqual(new_rle, self.compressed_rle)

        new_rle = mask_util.segmToRle([self.poly[0]], 20, 20)
        self.assertDictEqual(new_rle, self.poly_rles[0])

    def testAnnToRLE(self):
        fake_dataset = {
            "images": [
                {
                    "id": 1,
                    "width": 20,
                    "height": 20,
                    "file_name": "fake_image.jpg",
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": self.uncompressed_rle,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": self.compressed_rle,
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [self.poly[0].tolist()],
                },
            ],
            "categories": [
                {"id": 1, "name": "fake_category"},
            ],
        }

        fake_gt = COCO(fake_dataset)

        new_rle = fake_gt.annToRLE(fake_gt.anns[1])
        self.assertDictEqual(new_rle, self.compressed_rle)

        new_rle = fake_gt.annToRLE(fake_gt.anns[2])
        self.assertDictEqual(new_rle, self.compressed_rle)

        new_rle = fake_gt.annToRLE(fake_gt.anns[3])
        self.assertDictEqual(new_rle, self.poly_rles[0])

        mask = fake_gt.annToMask(fake_gt.anns[3])
        self.assertEqual(mask.tolist(), mask_util.decode(self.poly_rles[0]).tolist())


if __name__ == "__main__":
    unittest.main()
