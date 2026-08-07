#!/usr/bin/python3
import unittest

from faster_coco_eval import COCO


class TestOrphanAnnotations(unittest.TestCase):
    """Verify annotations remain consistently indexed without image metadata."""

    def test_orphan_annotation_is_available_from_every_annotation_index(self):
        """Keep orphan annotations loadable and discoverable by category."""
        annotation = {
            "id": 7,
            "image_id": 99,
            "category_id": 3,
            "bbox": [0, 0, 1, 1],
            "area": 1,
            "iscrowd": 0,
        }
        coco = COCO({
            "images": [],
            "categories": [{"id": 3, "name": "thing"}],
            "annotations": [annotation],
        })

        annotation_ids = coco.getAnnIds()

        self.assertEqual(annotation_ids, [annotation["id"]])
        self.assertEqual(coco.loadAnns(annotation_ids), [annotation])
        self.assertEqual(coco.imgToAnns[annotation["image_id"]], [annotation])
        self.assertEqual(coco.catToImgs[annotation["category_id"]], [annotation["image_id"]])


if __name__ == "__main__":
    unittest.main()
