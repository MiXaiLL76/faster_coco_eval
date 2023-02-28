"""The following API functions are defined
----------
encode         : Encode binary masks using RLE.  
decode         : Decode binary masks encoded via RLE.  
merge          : Compute union or intersection of encoded masks.  
iou            : Compute intersection over union between masks.  
area           : Compute area of encoded masks.  
toBbox         : Get bounding boxes surrounding encoded masks.  
frPyObjects    : Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.  

Usage
----------
Rs     : encode( masks )
masks  : decode( Rs )
R      : merge( Rs, intersect=false )
o      : iou( dt, gt, iscrowd )
a      : area( Rs )
bbs    : toBbox( Rs )
Rs     : frPyObjects( [pyObjects], h, w )

In the API the following formats are used
----------
Rs      : [dict] Run-length encoding of binary masks
R       : dict Run-length encoding of binary mask
masks   : [hxwxn] Binary mask(s) (must have type np.ndarray(dtype=uint8) in column:major order)
iscrowd : [nx1] list of np.ndarray. 1 indicates corresponding gt image has crowd region to ignore
bbs     : [nx4] Bounding box(es) stored as [x y w h]
poly    : Polygon stored as [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
dt,gt   : May be either bounding boxes or encoded masks

Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).
"""

import faster_coco_eval.mask_api_cpp as _mask
import numpy as np


def merge(rleObjs):
    """Compute union or intersection of encoded masks."""
    return _mask.merge(rleObjs)


def frPyObjects(segm: list, h: int, w: int):
    """Convert polygon, bbox, and uncompressed RLE to encoded RLE mask."""
    return _mask.frPyObjects(segm, h, w)


def iou(d: list, g: list, iscrowd: list):
    """Compute intersection over union between masks.

    Parameters
    ----------
    d : array_like
        Detected (dt). May be either bounding boxes or encoded masks.
    g : array_like
        Ground truth (gt). May be either bounding boxes or encoded masks.
    iscrowd : [nx1] list of np.ndarray
        1 indicates corresponding gt image has crowd region to ignore.

    Returns
    -------
    iou_array : np.ndarray
        Intersection over union between masks or bbox.
   """
    return _mask.iou(d, g, iscrowd)


def encode(bimask: np.ndarray):
    """Encode binary masks using RLE."""
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]


def decode(rleObjs):
    """Decode binary masks encoded via RLE."""
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs):
    """Compute area of encoded masks."""
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs):
    """Get bounding boxes surrounding encoded masks."""
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]
