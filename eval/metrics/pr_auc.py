from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np


def compute_pr_auc(y_true, y_proba, num_classes: int):
    """
    y_true: array shape (n_samples,)
    y_proba: array shape (n_samples, num_classes)
    """

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    pr_auc_macro = average_precision_score(
        y_true_bin,
        y_proba,
        average="macro"
    )

    pr_auc_weighted = average_precision_score(
        y_true_bin,
        y_proba,
        average="weighted"
    )

    return {
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_weighted": pr_auc_weighted,
    }