# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76

import copy
import json
import logging
import os
import pathlib
import time
import warnings
from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np

from faster_coco_eval.core import mask as maskUtils
from faster_coco_eval.version import __author__, __version__

logger = logging.getLogger(__name__)


def _isArrayLike(obj):
    """Check if the object is array-like.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: True if object behaves like an array, False otherwise.
    """
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(
        self,
        annotation_file: Optional[Union[str, dict, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath]] = None,
        use_deepcopy: bool = False,
        print_function: Callable = logger.debug,
    ):
        """Constructor of Microsoft COCO helper class.

        Args:
            annotation_file (Optional[Union[str, dict, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath]], optional): Path to annotation file or annotation dict. Defaults to None.
            use_deepcopy (bool, optional): Whether to copy the dict annotations. Defaults to False.
            print_function (Callable, optional): Function to use for printing messages. Defaults to logger.debug.
        """  # noqa: E501
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = (
            {},
            {},
            {},
            {},
        )
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.print_function = print_function
        self.use_deepcopy = use_deepcopy
        self.annotation_file = annotation_file

        if annotation_file is not None:
            self._print_function("loading annotations into memory...")
            tic = time.time()
            if type(annotation_file) in [str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list]:
                self.dataset = COCO.load_json(annotation_file, self.use_deepcopy)
            else:
                raise TypeError(f"type {type(annotation_file)} is not supported")

            assert type(self.dataset) is dict, f"annotation file format {type(self.dataset)} not supported"
            self.print_function(f"Done (t={time.time() - tic:0.2f}s)")
            self.createIndex()

    @property
    def print_function(self) -> Callable:
        """Get the function used for printing/logging messages.

        Returns:
            Callable: Print/log function.
        """
        return self._print_function

    @print_function.setter
    def print_function(self, value: Callable):
        """Set the function used for printing/logging messages.

        Args:
            value (Callable): Function to use for printing messages.
        """
        self._print_function = value

    def createIndex(self):
        """Create index for coco annotation data.

        Creates internal indices for the COCO dataset to enable fast
        lookups. Builds mappings between images, annotations and
        categories.
        """
        tic = time.time()
        # create index
        self.print_function("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        annsImgIds_dict = set()
        if "images" in self.dataset:
            for img in self.dataset["images"]:
                if type(img["id"]) is not int:
                    img["id"] = int(img["id"])

                imgs[img["id"]] = img
                annsImgIds_dict.add(img["id"])

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                if type(ann["image_id"]) is not int:
                    ann["image_id"] = int(ann["image_id"])

                if ann["image_id"] in annsImgIds_dict:
                    imgToAnns[ann["image_id"]].append(ann)
                    anns[ann["id"]] = ann

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        self.print_function("index created!")
        self.print_function(f"Done (t={time.time() - tic:0.2f}s)")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """Print information about the annotation file.

        Prints the info section of the annotation file using the print
        function.
        """
        for key, value in self.dataset["info"].items():
            self.print_function(f"{key}: {value}")

    def getAnnIds(
        self,
        imgIds: List[int] = [],
        catIds: List[int] = [],
        areaRng: List[float] = [],
        iscrowd: bool = None,
    ) -> List[int]:
        """Get ann ids that satisfy given filter conditions.

        Args:
            imgIds (List[int], optional): Get anns for given images. Defaults to [].
            catIds (List[int], optional): Get anns for given categories. Defaults to [].
            areaRng (List[float], optional): Get anns for given area range (e.g. [0, inf]). Defaults to [].
            iscrowd (bool, optional): Get anns for given crowd label (False or True). Defaults to None.

        Returns:
            List[int]: Integer array of ann ids that satisfy the criteria.
        """
        imgIds = set(imgIds if _isArrayLike(imgIds) else [imgIds])
        catIds = set(catIds if _isArrayLike(catIds) else [catIds])

        check_area = len(areaRng) == 2
        check_crowd = iscrowd is not None
        check_cat = len(catIds) > 0
        check_img = len(imgIds) > 0

        if not (check_area or check_crowd or check_cat or check_img):
            anns = self.dataset["annotations"]
        else:
            anns = []

            if check_img:
                for img_id in imgIds:
                    anns.extend(self.img_ann_map[img_id])
            else:
                anns = self.dataset["annotations"]

            if check_cat:
                anns = list(filter(lambda ann: ann["category_id"] in catIds, anns))

            if check_area:
                anns = list(
                    filter(
                        lambda ann: (ann["area"] > areaRng[0] and ann["area"] < areaRng[1]),
                        anns,
                    )
                )

            if check_crowd:
                anns = list(
                    filter(
                        lambda ann: (int(ann.get("iscrowd", 0)) == int(iscrowd)),
                        anns,
                    )
                )

        ids = list(map(lambda ann: ann["id"], anns))
        return ids

    def getCatIds(
        self,
        catNms: List[str] = [],
        supNms: List[str] = [],
        catIds: List[int] = [],
    ) -> List[int]:
        """Get category ids that satisfy given filter conditions.

        Args:
            catNms (List[str], optional): Get categories for given cat names. Defaults to [].
            supNms (List[str], optional): Get categories for given supercategory names. Defaults to [].
            catIds (List[int], optional): Get categories for given ids. Defaults to [].

        Returns:
            List[int]: Integer array of cat ids.
        """
        catNms = set(catNms if _isArrayLike(catNms) else [catNms])
        supNms = set(supNms if _isArrayLike(supNms) else [supNms])
        catIds = set(catIds if _isArrayLike(catIds) else [catIds])

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]

            if len(catNms) > 0:
                cats = list(filter(lambda cat: cat.get("name") in catNms, cats))

            if len(supNms) > 0:
                cats = list(filter(lambda cat: cat.get("supercategory") in supNms, cats))

            if len(catIds) > 0:
                cats = list(filter(lambda cat: cat.get("id") in catIds, cats))

        ids = [cat["id"] for cat in cats]
        return ids

    def getImgIds(self, imgIds: List[int] = [], catIds: List[int] = []) -> List[int]:
        """Get image ids that satisfy given filter conditions.

        Args:
            imgIds (List[int], optional): Get images for given ids. Defaults to [].
            catIds (List[int], optional): Get images with all given categories. Defaults to [].

        Returns:
            List[int]: Integer array of img ids.
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids: Union[List[int], int] = []) -> List[dict]:
        """Load annotations with the specified ids.

        Args:
            ids (Union[List[int], int], optional): Integer ids specifying annotations. Defaults to [].

        Returns:
            List[dict]: Loaded annotation objects.
        """
        if _isArrayLike(ids):
            return [self.anns[i] for i in ids]
        elif type(ids) is int:
            return [self.anns[ids]]

    def loadCats(self, ids: Union[List[int], int] = []) -> List[dict]:
        """Load categories with the specified ids.

        Args:
            ids (Union[List[int], int], optional): Integer ids specifying categories. Defaults to [].

        Returns:
            List[dict]: Loaded category objects.
        """
        if _isArrayLike(ids):
            return [self.cats[i] for i in ids]
        elif type(ids) is int:
            return [self.cats[ids]]

    def loadImgs(self, ids: Union[List[int], int] = []) -> List[dict]:
        """Load images with the specified ids.

        Args:
            ids (Union[List[int], int], optional): Integer ids specifying images. Defaults to [].

        Returns:
            List[dict]: Loaded image objects.
        """
        if _isArrayLike(ids):
            return [self.imgs[i] for i in ids]
        elif type(ids) is int:
            return [self.imgs[ids]]

    @staticmethod
    def load_json(
        json_file: Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list],
        use_deepcopy: Optional[bool] = False,
    ) -> dict:
        """Load a json file.

        Args:
            json_file (Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list]): Path to the json file or data dict/list.
            use_deepcopy (Optional[bool], optional): If True, use deep copy. Defaults to False.

        Returns:
            dict: Loaded json data.
        """  # noqa: E501
        if type(json_file) in [str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath]:
            with open(json_file) as io:
                _data = json.load(io)
        else:
            if use_deepcopy:
                return copy.deepcopy(json_file)
            else:
                return json_file.copy()
        return _data

    def loadRes(
        self,
        resFile: Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list, np.ndarray],
        min_score: float = 0.0,
    ) -> "COCO":
        """Load result file and return a result api object.

        Args:
            resFile (Union[str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list, np.ndarray]): File name of result file or numpy array.
            min_score (float, optional): Minimum score to consider a result. Defaults to 0.0.

        Returns:
            COCO: Result api object.
        """  # noqa: E501
        res = COCO()
        res.dataset["info"] = copy.deepcopy(self.dataset.get("info", {}))
        res.dataset["images"] = [img for img in self.dataset["images"]]

        self.print_function("Loading and preparing results...")
        tic = time.time()
        if type(resFile) in [str, os.PathLike, pathlib.PosixPath, pathlib.WindowsPath, dict, list]:
            anns = COCO.load_json(resFile, getattr(self, "use_deepcopy", False))
        elif type(resFile) is np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            raise TypeError(f"type {type(resFile)} is not supported")

        assert type(anns) is list, "results in not an array of objects"

        if min_score != 0.0:
            anns = [ann for ann in anns if ann.get("score", 1) >= min_score]

        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set([ann["image_id"] for ann in anns])
            res.dataset["images"] = [img for img in res.dataset["images"] if img["id"] in imgIds]
            for index, ann in enumerate(anns):
                ann["id"] = index + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for index, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = index + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for index, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = index + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for index, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = index + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        self.print_function(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def showAnns(self, anns: List[dict], draw_bbox: Optional[bool] = False):
        """Display the specified annotations.

        Args:
            anns (List[dict]): Annotations to display.
            draw_bbox (Optional[bool], optional): Whether to display bbox. Defaults to False.
        """
        if len(anns) == 0:
            return 0

        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            datasetType = "instances"
        elif "caption" in anns[0]:
            datasetType = "captions"
        else:
            raise Exception("datasetType not supported")

        if datasetType == "instances":
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon

            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if isinstance(ann["segmentation"], list):
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if isinstance(ann["segmentation"]["counts"], list):
                            rle = maskUtils.frPyObjects([ann["segmentation"]], t["height"], t["width"])
                        else:
                            rle = [ann["segmentation"]]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if "keypoints" in ann and isinstance(ann["keypoints"], list):
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                    kp = np.array(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0], y[v > 0], "o", markersize=8, markerfacecolor=c, markeredgecolor="k", markeredgewidth=2
                    )
                    plt.plot(
                        x[v > 1], y[v > 1], "o", markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
                    poly = [
                        [bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y],
                    ]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == "captions":
            for ann in anns:
                print(ann["caption"])

    def download(self, tarDir=None, imgIds=[]):
        """Deprecated: Download images (no longer supported).

        Args:
            tarDir (Any, optional): Target directory. Not used.
            imgIds (list, optional): Image ids. Not used.

        Raises:
            DeprecationWarning: Always raised, function is deprecated.
        """
        warnings.warn("download deprecated in 1.3.0", DeprecationWarning)

    def loadNumpyAnnotations(self, data: np.ndarray) -> List[dict]:
        """Convert result data from array to anns.

        Args:
            data (np.ndarray): 2d array where each row contains [imageID, x1, y1, w, h, score, class]

        Returns:
            List[dict]: Converted annotations as a list of dicts.
        """
        self.print_function("Converting ndarray to lists...")
        assert type(data) is np.ndarray
        self.print_function(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                self.print_function(f"{i}/{N}")
            ann += [
                {
                    "image_id": int(data[i, 0]),
                    "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    "score": data[i, 5],
                    "category_id": int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann: dict) -> dict:
        """Convert annotation which can be polygons, uncompressed RLE to RLE.

        Args:
            ann (dict): Annotation information.

        Returns:
            dict: Run-length encoding of the annotation.
        """
        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if type(segm) is list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm["counts"]) is list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann: dict) -> np.ndarray:
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.

        Args:
            ann (dict): Annotation information.

        Returns:
            np.ndarray: Binary mask of the annotation.
        """
        rle = self.annToRLE(ann)
        mask = maskUtils.decode(rle)
        return mask

    def get_ann_ids(
        self,
        img_ids: List[int] = [],
        cat_ids: List[int] = [],
        area_rng: List[float] = [],
        iscrowd: bool = None,
    ) -> List[int]:
        """Get ann ids that satisfy given filter conditions.

        Args:
            img_ids (List[int], optional): Get anns for given imgs. Defaults to [].
            cat_ids (List[int], optional): Get anns for given cats. Defaults to [].
            area_rng (List[float], optional): Get anns with area less than this. Defaults to [].
            iscrowd (bool, optional): Get anns for given crowd label. Defaults to None.

        Returns:
            List[int]: Integer array of ann ids.
        """
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(
        self,
        cat_names: List[str] = [],
        sup_names: List[str] = [],
        cat_ids: List[int] = [],
    ) -> List[int]:
        """Get cat ids that satisfy given filter conditions.

        Args:
            cat_names (List[str], optional): Get cats for given names. Defaults to [].
            sup_names (List[str], optional): Get cats for given supercategory names. Defaults to [].
            cat_ids (List[int], optional): Get cats for given ids. Defaults to [].

        Returns:
            List[int]: Integer array of cat ids.
        """
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids: List[int] = [], cat_ids: List[int] = []) -> List[int]:
        """Get img ids that satisfy given filter conditions.

        Args:
            img_ids (List[int], optional): Get imgs for given ids. Defaults to [].
            cat_ids (List[int], optional): Get imgs with all given cats. Defaults to [].

        Returns:
            List[int]: Integer array of img ids.
        """
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids: List[int]) -> List[dict]:
        """Load anns with the specified ids.

        Args:
            ids (List[int]): Integer ids specifying anns.

        Returns:
            List[dict]: Loaded annotation objects.
        """
        return self.loadAnns(ids)

    def load_cats(self, ids: List[int]) -> List[dict]:
        """Load cats with the specified ids.

        Args:
            ids (List[int]): Integer ids specifying cats.

        Returns:
            List[dict]: Loaded category objects.
        """
        return self.loadCats(ids)

    def load_imgs(self, ids: List[int]) -> List[dict]:
        """Load imgs with the specified ids.

        Args:
            ids (List[int]): Integer ids specifying imgs.

        Returns:
            List[dict]: Loaded image objects.
        """
        return self.loadImgs(ids)

    @property
    def img_ann_map(self) -> dict:
        """Return a mapping from image ids to annotation ids.

        Returns:
            dict: Mapping from image ids to annotation ids.
        """
        return self.imgToAnns

    @img_ann_map.setter
    def img_ann_map(self, value: dict):
        """Set the mapping from image ids to annotation ids.

        Args:
            value (dict): Mapping from image ids to annotation ids.
        """
        self.imgToAnns = value

    @property
    def cat_img_map(self) -> dict:
        """Return a mapping from category ids to image ids.

        Returns:
            dict: Mapping from category ids to image ids.
        """
        return self.catToImgs

    @cat_img_map.setter
    def cat_img_map(self, value: dict):
        """Set the mapping from category ids to image ids.

        Args:
            value (dict): Mapping from category ids to image ids.
        """
        self.catToImgs = value

    def __repr__(self) -> str:
        """String representation for COCO class.

        Returns:
            str: Representation with author and version.
        """
        s = self.__class__.__name__ + "(annotation_file) # "
        s += f"__author__='{__author__}'; "
        s += f"__version__='{__version__}';"
        return s

    def to_dict(self, separate_fn: bool = False) -> dict:
        """Convert to a standard python dictionary.

        Args:
            separate_fn (bool, optional): Whether to separate the fn category. Defaults to False.

        Returns:
            dict: Standard python dictionary containing the COCO data.
        """
        cats = list(self.cats.values())
        anns = list(self.anns.values())

        if separate_fn:
            max_category_id = max(cats, key=lambda x: x["id"])["id"]
            fn_cats = [
                dict(
                    category,
                    **{
                        "id": (category["id"] + max_category_id),
                        "name": (category["name"] + "_fn"),
                    },
                )
                for category in cats
            ]

            for ann in anns:
                if ann.get("fn"):
                    ann["category_id"] = ann["category_id"] + max_category_id

            cats += fn_cats

        return {
            "info": {"description": "Created from faster-coco-eval"},
            "images": list(self.imgs.values()),
            "annotations": anns,
            "categories": cats,
        }

    def __iter__(self):
        """Iterate over the annotations.

        Yields:
            Tuple[str, Any]: Key-value pair of the annotation data.
        """
        for key, val in self.to_dict().items():  # noqa: UP028
            yield key, val

    def dump(self, output_file: Union[str, os.PathLike]):
        """Dump annotations to a json file.

        Args:
            output_file (Union[str, os.PathLike]): Path to the output json file.
        """
        with open(output_file, "w") as io:
            json.dump(dict(self), io, ensure_ascii=False, indent=4)
