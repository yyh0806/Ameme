import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from ame.utils import *
from loguru import logger
from ame.functional import *


def IOU_score(output, target):
    iou_scores = []
    for thres in np.arange(0.5, 1.0, 0.05):
        iou_score = iou(output, target, 1e-7, float(thres))
        iou_scores.append(iou_score.cpu().detach().numpy())
    return np.mean(iou_scores)
