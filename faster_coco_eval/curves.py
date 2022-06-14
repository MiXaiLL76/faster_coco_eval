import tqdm
import numpy as np
import matplotlib.pyplot as plt

def thres(maxiou_confidence, threshold = 0.5):
    maxious = maxiou_confidence[:, 0]
    confidences = maxiou_confidence[:, 1]
    true_or_flase = (maxious > threshold)
    tf_confidence = np.array([true_or_flase, confidences])
    tf_confidence = tf_confidence.T
    tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
    return tf_confidence


def anns_as_bbox(anns, image_id=0, category=None):
    boxes = [
        list(ann['bbox']) + [ann.get('score', 1)] 
    for ann in anns if (ann['image_id'] == image_id) and ((ann['category_id'] == category) or (category is None))]

    boxes.sort(key = lambda x : x[-1], reverse=True)
    return boxes

def match(result_annotations, eval_all_coco, category=None):
    maxiou_confidence  = np.array([])
    num_detectedbox    = 0
    num_groundtruthbox = 0
    
    for image in eval_all_coco['images']:
        results = anns_as_bbox(result_annotations, image['id'], category)
        groundtruth = anns_as_bbox(eval_all_coco['annotations'], image['id'])
        num_detectedbox += len(results)
        num_groundtruthbox += len(groundtruth)
        
        pbar = tqdm.tqdm(total=len(results) * len(groundtruth), desc=f"{image['id']=}")
        for j in range(len(results)):
            iou_array = np.array([])
            detectedbox = results[j]
            confidence = detectedbox[-1]
            
            for k in range(len(groundtruth)):
                groundtruthbox = groundtruth[k]
                iou = cal_IoU(detectedbox, groundtruthbox)
                iou_array = np.append(iou_array, iou)
                pbar.update(1)
            
            maxiou = np.max(iou_array)
            maxiou_confidence = np.append(maxiou_confidence, [maxiou, confidence])
        pbar.close()
    
    maxiou_confidence = maxiou_confidence.reshape(-1, 2)
    maxiou_confidence = maxiou_confidence[np.argsort(-maxiou_confidence[:, 1])] # 按置信度从大到小排序

    return maxiou_confidence, num_detectedbox, num_groundtruthbox

def cal_IoU(detectedbox, groundtruthbox):
    leftx_det, topy_det, width_det, height_det, _ = detectedbox
    leftx_gt, topy_gt, width_gt, height_gt, _ = groundtruthbox

    centerx_det = leftx_det + width_det / 2
    centerx_gt = leftx_gt + width_gt / 2
    centery_det = topy_det + height_det / 2
    centery_gt = topy_gt + height_gt / 2

    distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
    distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

    if distancex <= 0 and distancey <= 0:
        intersection = distancex * distancey
        union = width_det * height_det + width_gt * height_gt - intersection
        iou = intersection / union
        # print(iou)
        return iou
    else:
        return 0

    
def plot_curve(match_results : list, threshold_iou=0.5, label_name=None):
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(15, 7)
    
    for _match in match_results:
        if len(_match) == 2:
            _label, _match = _match
        else:
            label_name = None
        
        if label_name is not None:
            label_string = f"[{label_name}={_label}] "
        else:
            label_string = ""
        
        maxiou_confidence, num_detectedbox, num_groundtruthbox = _match
        tf_confidence = thres(maxiou_confidence, threshold_iou)
        
        fp_list = []
        recall_list = []
        precision_list = []
        auc = 0
        mAP = 0
        for num in range(len(tf_confidence)):
            arr = tf_confidence[:(num + 1), 0]
            tp = np.sum(arr)
            fp = np.sum(arr == 0)
            recall = tp / num_groundtruthbox
            precision = tp / (tp + fp)
            auc = auc + recall
            mAP = mAP + precision

            fp_list.append(fp)
            recall_list.append(recall)
            precision_list.append(precision)

        auc = auc / len(fp_list)
        mAP = mAP * max(recall_list) / len(recall_list)

        axes[0].set_title('ROC')
        axes[0].set_xlabel('False Positives')
        axes[0].set_ylabel('True Positive rate')
        # plt.ylim(0, 1)
        axes[0].plot(fp_list, recall_list, label = f'{label_string}AUC: {auc:.3f}')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title('Precision-Recall')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        # plt.axis([0, 1, 0, 1])
        axes[1].plot(recall_list, precision_list, label = f'{label_string}mAP: {mAP:.3f}')
        axes[1].grid(True)
        axes[1].legend()
    
    plt.show()