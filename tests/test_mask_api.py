#!/usr/bin/python3

import unittest

import numpy as np
from parameterized import parameterized

import faster_coco_eval.core.mask as mask_util
import faster_coco_eval.mask_api_new_cpp as _mask
from faster_coco_eval import COCO


def _encode(x):
    """Encode a binary mask into a run-length encoded string."""
    return mask_util.encode(np.asfortranarray(x, np.uint8))


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
