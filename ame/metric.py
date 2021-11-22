import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from ame.utils import *
from loguru import logger


def IOU_score(output, target):
    """
    Get average IOU mAP score for a dataset
    :param output:
    :param target:
    :return:
    """
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    scores = []
    for i in range(target.shape[0]):
        score = iou_map([output], [target])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score
