# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import datetime
import io
import itertools
import numpy as np
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from json import dump
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional, Sequence, Union

from mmeval.core.base_metric import BaseMetric
from mmeval.fileio import get_local_path, load
from mmeval.utils import is_list_of
from mmeval.metrics import COCODetection
try:
    from faster_coco_wrapper import COCO, COCOeval
    HAS_FASTER_COCOAPI = True
except ImportError:
    HAS_FASTER_COCOAPI = False


class FasterCOCODetection(COCODetection):
    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 iou_thrs: Union[float, Sequence[float], None] = None,
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (1, 10, 100),
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 gt_mask_area: bool = True,
                 backend_args: Optional[dict] = None,
                 print_results: bool = True,
                 extra_calc: bool = False,
                 **kwargs) -> None:
        if not HAS_FASTER_COCOAPI:
            raise RuntimeError('Failed to import `COCO` and `COCOeval` from '
                               '`mmeval.utils.coco_wrapper`. '
                               'Please try to install official pycocotools by '
                               '"pip install pycocotools"')
        super().__init__(**kwargs)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox' and 'segm'"
                    f'but got {metric}.')

        # do class wise evaluation, default False
        self.classwise = classwise
        
        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        elif isinstance(iou_thrs, float):
            iou_thrs = np.array([iou_thrs])
        elif is_list_of(iou_thrs, float):
            iou_thrs = np.array(iou_thrs)
        else:
            raise TypeError(
                '`iou_thrs` should be None, float, or a list of float')

        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.print_results = print_results
        self.extra_calc = extra_calc
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        self._coco_api: Optional[COCO]  # type: ignore
        if ann_file is not None:
            with get_local_path(
                    filepath=ann_file,
                    backend_args=backend_args) as local_path:
                self._coco_api = COCO(annotation_file=local_path)
        else:
            self._coco_api = None

        self.gt_mask_area = gt_mask_area
        # handle dataset lazy init
        self.cat_ids: list = []
        self.img_ids: list = []

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".

        Returns:
            str: The filename of the json file.
        """
        try:
            from faster_coco_wrapper import mask_util
        except ImportError:
            mask_util = None

        warnings.warn(
            'The area of the instance is default to use bbox area. '
            'Compared to load annotation file evaluate way, this will '
            'not affect the overall AP, but leads to different '
            'small/medium/large AP results.')

        classes = self.classes
        categories = [
            dict(id=id, name=name) for id, name in enumerate(classes)
        ]
        image_infos: list = []
        annotations: list = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            gt_bboxes = gt_dict['bboxes']
            gt_labels = gt_dict['labels']
            assert len(gt_bboxes) == len(gt_labels)
            if 'ignore_flags' in gt_dict:
                ignore_flags = gt_dict['ignore_flags']
                assert len(gt_bboxes) == len(ignore_flags)
            else:
                ignore_flags = np.zeros(len(gt_bboxes))
            if 'masks' in gt_dict:
                gt_masks = gt_dict['masks']
                assert len(gt_masks) == len(gt_bboxes)
            else:
                gt_masks = [None for _ in range(len(gt_bboxes))]

            for i in range(len(gt_bboxes)):
                label = gt_labels[i]
                coco_bbox = self.xyxy2xywh(gt_bboxes[i])
                ignore_flag = ignore_flags[i]
                mask = gt_masks[i]
                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=int(ignore_flag),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if mask is not None:
                    if mask_util and self.gt_mask_area:
                        # Using mask area can reduce the gap of
                        # small/medium/large AP results.
                        area = mask_util.area(mask)
                        annotation['area'] = float(area)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmeval CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        with open(converted_json_path, 'w') as f:
            dump(coco_json, f)
        return converted_json_path

    def compute_metric(self, results: list) -> dict:
        """Compute the COCO metrics.

        Args:
            results (List[tuple]): A list of tuple. Each tuple is the
                prediction and ground truth of an image. This list has already
                been synced across all ranks.

        Returns:
            dict: The computed metric.
            The keys are the names of the metrics, and the values are
            corresponding results.
        """
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        classes = self.classes
        # split gt and prediction list
        preds, gts = zip(*results)

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            self.logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if len(self.cat_ids) == 0:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=classes)  # type: ignore
        if len(self.img_ids) == 0:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results: OrderedDict = OrderedDict()
        table_results: OrderedDict = OrderedDict()
        if self.format_only:
            self.logger.info(
                f'Results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            self.logger.info(f'Evaluating {metric}...')

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                self.logger.warning('The testing results of the '
                                    'whole dataset is empty.')
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type, extra_calc=self.extra_calc)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = self.proposal_nums
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                f'AR@{self.proposal_nums[0]}': 6,
                f'AR@{self.proposal_nums[1]}': 7,
                f'AR@{self.proposal_nums[2]}': 8,
                f'AR_s@{self.proposal_nums[2]}': 9,
                f'AR_m@{self.proposal_nums[2]}': 10,
                f'AR_l@{self.proposal_nums[2]}': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            coco_eval.evaluate()
            coco_eval.accumulate()
            # Save coco summarize print information to logger
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            self.logger.info('\n' + redirect_string.getvalue())
            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            results_list = []
            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[coco_metric_names[metric_item]]
                results_list.append(f'{round(val * 100, 2):0.2f}')
                eval_results[key] = float(val)
            table_results[f'{metric}_result'] = results_list

            if self.classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = coco_eval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, cat_id in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self._coco_api.loadCats(cat_id)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{round(ap * 100, 2):0.2f}'))
                    eval_results[f'{metric}_{nm["name"]}_precision'] = ap

                table_results[f'{metric}_classwise_result'] = \
                    results_per_category
        if tmp_dir is not None:
            tmp_dir.cleanup()
        # if the testing results of the whole dataset is empty,
        # does not print tables.
        if self.print_results and len(table_results) > 0:
            self._print_results(table_results)
        return eval_results

    def _print_results(self, table_results: dict) -> None:
        """Print the evaluation results table.

        Args:
            table_results (dict): The computed metric.
        """
        for metric in self.metrics:
            result = table_results[f'{metric}_result']

            if metric == 'proposal':
                table_title = ' Recall Results (%)'
                if self.metric_items is None:
                    assert len(result) == 6
                    headers = [
                        f'AR@{self.proposal_nums[0]}',
                        f'AR@{self.proposal_nums[1]}',
                        f'AR@{self.proposal_nums[2]}',
                        f'AR_s@{self.proposal_nums[2]}',
                        f'AR_m@{self.proposal_nums[2]}',
                        f'AR_l@{self.proposal_nums[2]}'
                    ]
                else:
                    assert len(result) == len(self.metric_items)  # type: ignore # yapf: disable # noqa: E501
                    headers = self.metric_items  # type: ignore
            else:
                table_title = f' {metric} Results (%)'
                if self.metric_items is None:
                    assert len(result) == 6
                    headers = [
                        f'{metric}_mAP', f'{metric}_mAP_50',
                        f'{metric}_mAP_75', f'{metric}_mAP_s',
                        f'{metric}_mAP_m', f'{metric}_mAP_l'
                    ]
                else:
                    assert len(result) == len(self.metric_items)
                    headers = [
                        f'{metric}_{item}' for item in self.metric_items
                    ]
            table = Table(title=table_title)
            console = Console()
            for name in headers:
                table.add_column(name, justify='left')
            table.add_row(*result)
            with console.capture() as capture:
                console.print(table, end='')
            self.logger.info('\n' + capture.get())

            if self.classwise and metric != 'proposal':
                self.logger.info(
                    f'Evaluating {metric} metric of each category...')
                classwise_table_title = f' {metric} Classwise Results (%)'
                classwise_result = table_results[f'{metric}_classwise_result']

                num_columns = min(6, len(classwise_result) * 2)
                results_flatten = list(itertools.chain(*classwise_result))
                headers = ['category', f'{metric}_AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])

                table = Table(title=classwise_table_title)
                console = Console()
                for name in headers:
                    table.add_column(name, justify='left')
                for _result in results_2d:
                    table.add_row(*_result)
                with console.capture() as capture:
                    console.print(table, end='')
                self.logger.info('\n' + capture.get())

# Keep the deprecated metric name as an alias.
# The deprecated Metric names will be removed in 1.0.0!
COCODetectionMetric = COCODetection
