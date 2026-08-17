COCOeval_faster
=================

RLE IoU worker limit
--------------------

``COCOeval_faster`` accepts ``rle_iou_max_workers`` (default: ``8``) to
cap concurrency during Python-side RLE IoU computation. The effective worker
count is limited by process capacity, the configured cap, and the number of
image/category pairs. A value of ``1`` keeps this phase serial. The pending
task window is bounded internally to at most twice the effective worker count.
This setting applies only to the Python RLE IoU phase; it does not control the
C++ evaluation or accumulation phase.

.. autoclass:: faster_coco_eval.core.COCOeval_faster
   :members:
   :undoc-members:
   :show-inheritance:
