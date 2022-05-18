from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec,Layer

def py_greedy_nms(dets, iou_thr=0.45):
    """Pure python implementation of traditional greedy NMS.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        iou_thr (float): Drop the boxes that overlap with current
            maximum > thresh.
    Returns:
        numpy.array: Retained boxes.
    """
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_thr)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return dets[keep, :]


def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]

    areas = (x2 - x1 ) * (y2 - y1 )
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 1], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 2], dets[1:, 2])
        yy1 = np.maximum(dets[0, 3], dets[1:, 3])
        xx2 = np.minimum(dets[0, 4], dets[1:, 4])
        yy2 = np.minimum(dets[0, 5], dets[1:, 5])

        w = np.maximum(xx2 - xx1 , 0.0)
        h = np.maximum(yy2 - yy1 , 0.0)
        inter = w * h
        iou = inter / (dets[0, 6] + dets[1:, 6] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 1] *= weight
        retained_idx = np.where(dets[1:, 1] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)

def my_nms(y_pred,config):

    shape = np.shape(y_pred)
    batch_size = shape[0]
    n_boxes = shape[1]
    n_classes = config["nclasses"]

    outputs = []

    for i in range(batch_size):
        batch_item = y_pred[i]
        predictions = []
        for j in range(1,n_classes):
            confidences = np.expand_dims(batch_item[...,j],axis = -1)
            class_id = np.zeros(np.shape(confidences))
            class_id.fill(j)
            box_coordinates = batch_item[...,n_classes:n_classes+4]
            single_class = np.concatenate([class_id,confidences,box_coordinates],axis = -1)
            threshold_met = single_class[:,1]>0.01
            single_class = single_class[threshold_met]

            if len(single_class):
          #      single_class = py_greedy_nms(single_class)
                single_class = py_soft_nms(single_class, method='gaussian', iou_thr=0.3, sigma=0.5, score_thr=0.001)
            else:
                single_class = np.zeros((1,6))

            predictions.append(single_class)

        predictions = np.concatenate(predictions,axis = 0)
        if len(predictions)>200:
            index = np.argsort(predictions[:,1])[::-1]
            predictions = predictions[index[0:200]]

        outputs.append(predictions)

    return outputs
