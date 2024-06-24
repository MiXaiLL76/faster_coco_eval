# Original work Copyright (c) Piotr Dollar and Tsung-Yi Lin, 2014.
# Modified work Copyright (c) 2024 MiXaiLL76

import json
import logging
import time
import warnings
from collections import defaultdict

import numpy as np

import faster_coco_eval.faster_eval_api_cpp as _C
from faster_coco_eval.core import mask as maskUtils
from faster_coco_eval.version import __author__, __version__

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
            {},
            {},
            {},
            {},
        )
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.score_tresh: float = 0.0
        self.print_function = logger.debug

        if annotation_file is not None:
            self._print_function("loading annotations into memory...")
            tic = time.time()
            if type(annotation_file) is str:
                self.dataset = self.load_json(annotation_file)
            elif type(annotation_file) is dict:
                self.dataset = _C.deepcopy(annotation_file)
            else:
                self.dataset = None

            assert (
                type(self.dataset) is dict
            ), "annotation file format {} not supported".format(
                type(self.dataset)
            )
            self.print_function("Done (t={:0.2f}s)".format(time.time() - tic))
            self.createIndex()

    @property
    def print_function(self):
        return self._print_function

    @print_function.setter
    def print_function(self, value):
        self._print_function = value

    def createIndex(self):
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
        self.print_function("Done (t={:0.2f}s)".format(time.time() - tic))

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def createSubIndex(self):
        """extra_calc sub index for math_matches."""
        tic = time.time()
        # create index
        self.print_function("creating sub_index...")
        annToImgs = {}
        imgCatToAnnsIdx = defaultdict(dict)
        imgCatToAnnsIdx_count = defaultdict(int)
        imgToAnnsIdx = defaultdict(dict)
        imgToAnnsIdx_count = defaultdict(int)

        for ann_id, ann in self.anns.items():
            annToImgs[ann_id] = ann["image_id"]

            imgCatToAnnsIdx[(ann["image_id"], ann["category_id"])][
                ann["id"]
            ] = imgCatToAnnsIdx_count[(ann["image_id"], ann["category_id"])]
            imgCatToAnnsIdx_count[(ann["image_id"], ann["category_id"])] += 1

            imgToAnnsIdx[ann["image_id"]][ann["id"]] = imgToAnnsIdx_count[
                ann["image_id"]
            ]
            imgToAnnsIdx_count[ann["image_id"]] += 1

        self.print_function("sub_index created!")
        self.print_function("Done (t={:0.2f}s)".format(time.time() - tic))

        # create class members
        self.annToImgs = annToImgs
        self.imgCatToAnnsIdx = imgCatToAnnsIdx
        self.imgToAnnsIdx = imgToAnnsIdx

    def info(self):
        """Print information about the annotation file.

        :return:

        """
        for key, value in self.dataset["info"].items():
            self.print_function("{}: {}".format(key, value))

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

        check_area = len(areaRng) > 0
        check_crowd = iscrowd is not None
        check_cat = len(catIds) > 0
        check_img = len(imgIds) > 0

        if not (check_area and check_crowd and check_cat and check_img):
            anns = self.dataset["annotations"]
        else:
            anns = []

            if check_img:
                for img_id in imgIds:
                    anns.extend(self.img_ann_map[img_id])
            else:
                anns = self.dataset["annotations"]

            if check_cat:
                anns = [ann for ann in anns if ann["category_id"] in catIds]

            if check_area:
                areaRng = [0, float("inf")]

                anns = [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]

            if check_crowd:
                anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

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

        self.print_function("Loading and preparing results...")
        tic = time.time()
        if type(resFile) is str:
            anns = self.load_json(resFile)
        elif type(resFile) is np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = _C.deepcopy(resFile)

        assert type(anns) is list, "results in not an array of objects"

        if self.score_tresh != 0:
            anns = [
                ann for ann in anns if ann.get("score", 1) >= self.score_tresh
            ]

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
            res.dataset["categories"] = _C.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = _C.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = _C.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        self.print_function("DONE (t={:0.2f}s)".format(time.time() - tic))

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

        self.print_function("Converting ndarray to lists...")
        assert type(data) is np.ndarray
        self.print_function(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                self.print_function("{}/{}".format(i, N))
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

    def __repr__(self):
        s = self.__class__.__name__
        s += "("
        s += "annotation_file"
        s += ") # "
        s += "__author__='{}'; ".format(__author__)
        s += "__version__='{}';".format(__version__)
        return s
