from typing import List

import numpy as np


def confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray) -> List[int]:
    """
    Calculate the confusion matrix for binary classification.

    Parameters:
    - true_labels (np.ndarray): The true labels.
    - predicted_labels (np.ndarray): The predicted labels.

    Returns:
    - List[int]: A list containing the counts of True Positives (TP), True Negatives (TN),
                 False Positives (FP), and False Negatives (FN) in this order.
    """
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))
    return tp, tn, fp, fn