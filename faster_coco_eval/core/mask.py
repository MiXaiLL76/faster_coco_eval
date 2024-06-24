# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76
import faster_coco_eval.mask_api_new_cpp as _mask

iou = _mask.iou
merge = _mask.merge
frPyObjects = _mask.frPyObjects


def encode(bimask):
    """Encode binary masks using RLE."""
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order="F"))[0]


def decode(rleObjs):
    """Decode binary masks encoded via RLE."""
    if type(rleObjs) is list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs):
    """Compute area of encoded masks."""
    if type(rleObjs) is list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs):
    """Get bounding boxes surrounding encoded masks."""
    if type(rleObjs) is list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]
