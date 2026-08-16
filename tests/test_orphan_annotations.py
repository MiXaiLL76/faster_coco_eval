#!/usr/bin/python3
import unittest

from faster_coco_eval import COCO


class _SinglePassAnnotations(list):
    """Reject a second iteration so the indexing pass remains single-pass."""

    def __init__(self, annotations):
        """Store annotations and track iterations requested by
        `createIndex`."""
        super().__init__(annotations)
        self.iteration_count = 0

    def __iter__(self):
        """Yield annotations once, failing if an index rebuild traverses them
        again."""
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise AssertionError("createIndex iterated annotations more than once")
        return super().__iter__()


class TestOrphanAnnotations(unittest.TestCase):
    """Verify annotations remain consistently indexed without image
    metadata."""

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

    def test_create_index_builds_all_annotation_indexes_in_one_pass(self):
        """Build every annotation index during a single annotation
        traversal."""
        annotations = _SinglePassAnnotations([
            {"id": 1, "image_id": 1, "category_id": 3},
            {"id": 2, "image_id": 2, "category_id": 4},
        ])
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 3}, {"id": 4}],
            "annotations": annotations,
        }

        coco.createIndex()

        self.assertEqual(annotations.iteration_count, 1)
        self.assertEqual(list(coco.anns), [1, 2])
        self.assertEqual(coco.imgToAnns[1], [annotations[0]])
        self.assertEqual(coco.catToImgs[4], [2])

    def test_create_index_skips_category_index_without_categories(self):
        """Preserve the empty category index for category-less datasets."""
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3}],
        }

        coco.createIndex()

        self.assertEqual(coco.imgToAnns[1], [{"id": 1, "image_id": 1, "category_id": 3}])
        self.assertEqual(coco.catToImgs, {})


if __name__ == "__main__":
    unittest.main()
