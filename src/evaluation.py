import numpy as np

def calculate_iou(segmented_image, ground_truth):
    intersection = np.logical_and(segmented_image, ground_truth)
    union = np.logical_or(segmented_image, ground_truth)
    iou = np.sum(intersection) / np.sum(union)
    return iou
