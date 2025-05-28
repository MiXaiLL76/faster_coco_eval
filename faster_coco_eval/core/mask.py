# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76
from typing import Dict, List, Union

import numpy as np

try:
    import cv2

    opencv_available = True
except ImportError:
    opencv_available = False

import faster_coco_eval.mask_api_new_cpp as _mask

ValidRleType = Union[List[np.ndarray], List[List[float]], np.ndarray, List[dict]]


def segmToRle(segm: Union[List[float], List[int], dict], w: int, h: int):
    """Convert segm array to run-length encoding.

    Args:
        segm (Union[List[float], List[int], dict]): Segmentation map, can be a list of floats, ints or a dictionary.
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        dict: Run-length encoding of the segmentation map.
    """
    return _mask.segmToRle(segm, w, h)


def rleToBoundaryCV(rle: dict, dilation_ratio: float = 0.02) -> dict:
    """Convert run-length encoding to boundary rle using OpenCV backend.

    Args:
        rle (dict): Run-length encoding of a binary mask.
        dilation_ratio (float, optional): Ratio of dilation to apply to the mask. Defaults to 0.02.

    Returns:
        dict: Run-length encoding of the boundary mask.
    """
    mask = _mask.decode([rle])[:, :, 0]
    h, w = rle["size"]

    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is
    # also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    boundary_mask = mask - mask_erode
    return _mask.encode(boundary_mask[..., None])[0]


def rleToBoundary(
    rle: dict,
    dilation_ratio: float = 0.02,
    backend: str = "mask_api",
) -> dict:
    """Convert run-length encoding to boundary rle.

    Args:
        rle (dict): Run-length encoding of a binary mask.
        dilation_ratio (float, optional): Ratio of dilation to apply to the mask. Defaults to 0.02.
        backend (str, optional): Backend to use for conversion. 'mask_api' uses the faster_eval_api_cpp backend,
            'opencv' uses OpenCV for conversion. Defaults to "mask_api".

    Returns:
        dict: Run-length encoding of the boundary mask.

    Raises:
        ImportError: If OpenCV is selected as backend but not available.
    """
    if backend == "mask_api":
        return _mask.toBoundary([rle], dilation_ratio)[0]
    else:
        if not opencv_available:
            raise ImportError("OpenCV is not available. Please install OpenCV to use this function.")
        return rleToBoundaryCV(rle, dilation_ratio)


def calculateRleForAllAnnotations(
    anns: List[dict],
    img_sizes: Dict[int, tuple],
    compute_rle: bool,
    compute_boundary: bool,
    boundary_dilation_ratio: float,
    boundary_cpu_count: int,
):
    """Calculate run-length encoding for all annotations.

    Args:
        anns (List[dict]): List of annotation dictionaries.
        img_sizes (Dict[int, tuple]): Dictionary mapping image ids to their sizes (height, width).
        compute_rle (bool): Whether to compute run-length encoding.
        compute_boundary (bool): Whether to compute boundary run-length encoding.
        boundary_dilation_ratio (float): Ratio of dilation to apply to the mask boundary.
        boundary_cpu_count (int): Number of CPUs to use for boundary computation.

    Returns:
        None
    """
    _mask.calculateRleForAllAnnotations(
        anns,
        img_sizes,
        compute_rle,
        compute_boundary,
        boundary_dilation_ratio,
        boundary_cpu_count,
    )


def iou(
    dt: ValidRleType,
    gt: ValidRleType,
    iscrowd: List[int],
) -> Union[list, np.ndarray]:
    """Compute intersection over union (IoU) between two sets of run-length
    encoded masks.

    Args:
        dt (ValidRleType): Detected masks, can be a list of RLEs or arrays.
        gt (ValidRleType): Ground truth masks, can be a list of RLEs or arrays.
        iscrowd (List[int]): List of flags indicating whether each ground truth mask is a crowd region.

    Returns:
        Union[list, np.ndarray]: Intersection over union between dt and gt masks.
    """
    return _mask.iou(dt, gt, iscrowd)


def merge(rleObjs: List[dict], intersect: int = 0):
    """Merge a list of run-length encoded objects.

    Args:
        rleObjs (List[dict]): List of run-length encoding dictionaries of binary masks.
        intersect (int, optional): Flag for type of merge to perform. Defaults to 0.

    Returns:
        dict: Run-length encoding of merged mask.
    """
    return _mask.merge(rleObjs, intersect)


def frPyObjects(
    objs: Union[
        ValidRleType,
        np.ndarray,
        List[float],
        dict,
    ],
    h: int,
    w: int,
) -> Union[dict, List[dict]]:
    """Convert a list of objects to RLE format suitable for use in mask API.

    Args:
        objs (Union[ValidRleType, np.ndarray, List[float], dict]): Objects to be converted (polygons, bboxes, etc).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        Union[dict, List[dict]]: Run-length encoding of the objects.
    """
    return _mask.frPyObjects(objs, h, w)


def encode(bimask: np.ndarray) -> dict:
    """Encode binary mask(s) using RLE.

    Args:
        bimask (np.ndarray): Binary mask. Can be 2D (H, W) or 3D (H, W, N).

    Returns:
        dict: Run-length encoding of the binary mask(s).
    """
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order="F"))[0]


def decode(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """Decode binary masks encoded via RLE.

    Args:
        rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

    Returns:
        np.ndarray: Decoded binary mask(s).
    """
    if type(rleObjs) is list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """Compute area of encoded masks.

    Args:
        rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

    Returns:
        np.ndarray: Area(s) of run-length encodings.
    """
    if type(rleObjs) is list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs: Union[dict, List[dict]]) -> np.ndarray:
    """Get bounding boxes surrounding encoded masks.

    Args:
        rleObjs (Union[dict, List[dict]]): Run-length encoding of binary mask(s).

    Returns:
        np.ndarray: Bounding box(es) of run-length encodings.
    """
    if type(rleObjs) is list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]
