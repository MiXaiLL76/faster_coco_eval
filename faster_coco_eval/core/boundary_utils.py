"""
Copyright (c) 2021, Bowen Cheng
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""
import multiprocessing
import time

import cv2
import numpy as np

from . import mask as mask_utils


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation =
                                   dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is 
    # also considered as boundary.
    new_mask = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


# COCO/LVIS related util functions, to get the boundary for every annotations.
def augment_annotations_with_boundary_single_core(proc_id,
                                                  annotations,
                                                  ann_to_mask,
                                                  dilation_ratio=0.02):
    new_annotations = []

    for ann in annotations:
        mask = ann_to_mask(ann)
        # Find mask boundary.
        boundary = mask_to_boundary(mask, dilation_ratio)
        # Add boundary to annotation in RLE format.
        ann['boundary'] = mask_utils.encode(
            np.array(boundary[:, :, None], order="F", dtype="uint8"))[0]
        new_annotations.append(ann)

    return new_annotations


def add_boundary_multi_core(coco, cpu_num=16, dilation_ratio=0.02):
    print('Adding `boundary` to annotation.')
    tic = time.time()
    cpu_num = min(cpu_num, multiprocessing.cpu_count())

    annotations = coco.dataset["annotations"]
    annotations_split = np.array_split(annotations, cpu_num)
    print("Number of cores: {}, annotations per core: {}".format(
        cpu_num, len(annotations_split[0])
    ))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(augment_annotations_with_boundary_single_core,
                                (proc_id,
                                 annotation_set,
                                 coco.annToMask,
                                 dilation_ratio)
                                )
        processes.append(p)

    new_annotations = []
    for p in processes:
        new_annotations.extend(p.get())

    workers.close()
    workers.join()

    coco.dataset["annotations"] = new_annotations
    coco.createIndex()
    print('`boundary` added! (t={:0.2f}s)'.format(time.time() - tic))
