#!/usr/bin/python3
import unittest
import warnings

from faster_coco_eval import COCO


class TestAnnotationIdZero(unittest.TestCase):
    """Regression test for #77 — gt/dt annotation id=0 collides with the
    internal "no match" sentinel used during evaluation matching, silently
    miscounting that annotation as unmatched.

    createIndex() should warn.
    """

    def _dataset(self, ann_ids):
        """Build a minimal single-image dataset with the given annotation
        ids."""
        return {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "x"}],
            "annotations": [
                {
                    "id": ann_id,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
                for ann_id in ann_ids
            ],
        }

    def test_warns_when_annotation_id_zero_present(self):
        """CreateIndex() warns when a dataset contains annotation id 0."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            COCO(self._dataset([0, 1]))

        self.assertTrue(
            any("id 0" in str(w.message) or "id=0" in str(w.message) for w in caught),
            f"expected a warning about annotation id 0, got: {[str(w.message) for w in caught]}",
        )

    def test_no_warning_when_ids_start_from_one(self):
        """CreateIndex() stays silent when no annotation id is 0."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            COCO(self._dataset([1, 2]))

        self.assertEqual(caught, [], f"unexpected warnings: {[str(w.message) for w in caught]}")


if __name__ == "__main__":
    unittest.main()
