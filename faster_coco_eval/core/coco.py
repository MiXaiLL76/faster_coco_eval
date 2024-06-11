__author__ = "tylin"
__version__ = "2.0"
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both # noqa: E501
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures. # noqa: E501
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import itertools
import json
import logging
import time
import warnings
from collections import defaultdict

import numpy as np

from . import mask as maskUtils

logger = logging.getLogger(__name__)


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(self, annotation_file=None):
        """Constructor of Microsoft COCO helper class for reading and
        visualizing annotations.

        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts
            images.
        :return:

        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = (
            dict(),
            dict(),
            dict(),
            dict(),
        )
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.score_tresh: float = 0.0

        if annotation_file is not None:
            logger.debug("loading annotations into memory...")
            tic = time.time()
            if type(annotation_file) is str:
                self.dataset = self.load_json(annotation_file)
            elif type(annotation_file) is dict:
                self.dataset = copy.deepcopy(annotation_file)
            else:
                self.dataset = None

            assert (
                type(self.dataset) is dict
            ), "annotation file format {} not supported".format(
                type(self.dataset)
            )
            logger.debug("Done (t={:0.2f}s)".format(time.time() - tic))
            self.createIndex()

    def createIndex(self):
        # create index
        logger.debug("creating index...")
        anns, cats, imgs, annToImgs = {}, {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        imgCatToAnnsIdx = defaultdict(dict)
        imgToAnnsIdx = defaultdict(dict)

        annsImgIds_dict = {}
        if "images" in self.dataset:
            for img in self.dataset["images"]:
                img["id"] = int(img["id"])
                imgs[img["id"]] = img
                annsImgIds_dict[img["id"]] = True

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                ann["image_id"] = int(ann["image_id"])
                if annsImgIds_dict.get(ann["image_id"]):
                    imgToAnns[ann["image_id"]].append(ann)
                    anns[ann["id"]] = ann
                    annToImgs[ann["id"]] = ann["image_id"]
                    imgCatToAnnsIdx[(ann["image_id"], ann["category_id"])][
                        ann["id"]
                    ] = len(
                        imgCatToAnnsIdx[(ann["image_id"], ann["category_id"])]
                    )
                    imgToAnnsIdx[ann["image_id"]][ann["id"]] = len(
                        imgToAnnsIdx[ann["image_id"]]
                    )

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        logger.debug("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.annToImgs = annToImgs
        self.imgCatToAnnsIdx = imgCatToAnnsIdx
        self.imgToAnnsIdx = imgToAnnsIdx
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """Print information about the annotation file.

        :return:

        """
        for key, value in self.dataset["info"].items():
            logger.debug("{}: {}".format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """Get ann ids that satisfy given filter conditions.

        default skips that filter
        :param imgIds (int array) : get anns for given imgs :param
            catIds (int array) : get anns for given cats
        :param areaRng (float array) : get anns for given area range
            (e.g. [0 inf])
        :param iscrowd (boolean) : get anns for given crowd label (False
            or True)
        :return: ids (int array)       : integer array of ann ids

        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId]
                    for imgId in imgIds
                    if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )
        if iscrowd is not None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """Filtering parameters.

        default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array) : get cats for given supercategory
            names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids

        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]

            if len(catNms) > 0:
                name_to_cat = {cat.get("name"): cat for cat in cats}
                cats = [
                    name_to_cat[label]
                    for label in catNms
                    if name_to_cat.get(label)
                ]

            if len(supNms) > 0:
                supercategory_to_cat = {
                    cat.get("supercategory"): cat for cat in cats
                }
                cats = [
                    supercategory_to_cat[label]
                    for label in supNms
                    if supercategory_to_cat.get(label)
                ]

            if len(catIds) > 0:
                id_to_cat = {cat.get("id"): cat for cat in cats}
                cats = [id_to_cat[idx] for idx in catIds if id_to_cat.get(idx)]

        ids = [cat["id"] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """Get img ids that satisfy given filter conditions.

        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids

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

    def loadAnns(self, ids=[]):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects

        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) is int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """Load cats with the specified ids.

        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects

        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) is int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects

        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) is int:
            return [self.imgs[ids]]

    @staticmethod
    def load_json(json_file):
        with open(json_file) as io:
            _data = json.load(io)
        return _data

    def loadRes(self, resFile, min_score=0):
        """Load result file and return a result api object.

        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object

        """
        self.score_tresh = min_score
        res = COCO()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        logger.debug("Loading and preparing results...")
        tic = time.time()
        if type(resFile) is str:
            anns = self.load_json(resFile)
        elif type(resFile) is np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = copy.deepcopy(resFile)

        assert type(anns) is list, "results in not an array of objects"

        anns = [ann for ann in anns if ann.get("score", 1) >= self.score_tresh]

        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
                [ann["image_id"] for ann in anns]
            )
            res.dataset["images"] = [
                img for img in res.dataset["images"] if img["id"] in imgIds
            ]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(
                self.dataset["categories"]
            )
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(
                self.dataset["categories"]
            )
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(
                self.dataset["categories"]
            )
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        logger.debug("DONE (t={:0.2f}s)".format(time.time() - tic))

        annsImgIds_dict = {image["id"]: True for image in res.dataset["images"]}
        anns = [ann for ann in anns if annsImgIds_dict.get(ann["image_id"])]

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def showAnns(self, anns, draw_bbox=False):
        warnings.warn("showAnns deprecated in 1.3.0", DeprecationWarning)

    def download(self, tarDir=None, imgIds=[]):
        warnings.warn("download deprecated in 1.3.0", DeprecationWarning)

    def loadNumpyAnnotations(self, data):
        """Convert result data from array to anns.

        :param data (numpy.ndarray): array [Nx7] where each row contains
            [imageID,x1,y1,w,h,score,class]
        :return: annotations (python nested list)

        """

        logger.debug("Converting ndarray to lists...")
        assert type(data) is np.ndarray
        logger.debug(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                logger.debug("{}/{}".format(i, N))
            ann += [
                {
                    "image_id": int(data[i, 0]),
                    "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    "score": data[i, 5],
                    "category_id": int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.

        :return: binary mask (numpy 2D array)

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

    def annToMask(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.

        :return: binary mask (numpy 2D array)

        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        """Get ann ids that satisfy given filter conditions.

        :param img_ids (int array) : get anns for given imgs
        :param cat_ids (int array) : get anns for given cats
        :param area_rng (float array) : get anns for given area range
            (e.g. [0 inf])
        :return: ids (int array)  : integer array of ann ids

        """
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        """Get cat ids that satisfy given filter conditions.

        :param cat_names (str array) : get cats for given cat names
        :param sup_names (str array) : get cats for given supercategory
            names
        :param cat_ids (int array) : get cats for given cat ids
        :return: ids (int array)  : integer array of cat ids

        """
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        """Get img ids that satisfy given filter conditions.

        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids

        """
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects

        """
        return self.loadAnns(ids)

    def load_cats(self, ids):
        """Load cats with the specified ids.

        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects

        """
        return self.loadCats(ids)

    def load_imgs(self, ids):
        """Load anns with the specified ids.

        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects

        """
        return self.loadImgs(ids)

    @property
    def img_ann_map(self):
        return self.imgToAnns

    @property
    def cat_img_map(self):
        return self.catToImgs

    @property
    def ann_img_map(self):
        return self.annToImgs

    @property
    def img_ann_idx_map(self):
        return self.imgToAnnsIdx

    @property
    def img_cat_ann_idx_map(self):
        return self.imgCatToAnnsIdx
