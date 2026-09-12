import numpy as np
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official PHM08 challenge scoring function.

    Asymmetric by design: overestimating remaining life (d > 0, predicting
    the engine is healthier than it is) is scored far more harshly than
    underestimating it (d < 0), because in a maintenance setting a late
    prediction risks an in-service failure while an early one only costs
    unnecessary maintenance. Plain RMSE treats both errors identically, so
    it doesn't capture this asymmetric risk.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def imminent_failure_prf(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 20
) -> dict:
    """Precision/recall/F1 for the rare "imminent failure" class (RUL <= threshold).

    RUL regression error alone hides how the model does on the rare cycles
    that actually matter operationally. Thresholding both true and predicted
    RUL into a binary "about to fail" label reframes this as the imbalanced
    classification problem it really is, which is the metric Phase 2's
    augmentation is meant to move.
    """
    true_positive_class = y_true <= threshold
    pred_positive_class = y_pred <= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_positive_class, pred_positive_class, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_positive": int(true_positive_class.sum()),
        "support_total": int(len(y_true)),
    }
