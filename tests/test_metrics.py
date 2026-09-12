import numpy as np

from guardian.eval.metrics import imminent_failure_prf, mae, phm08_score, rmse


def test_rmse_mae_zero_for_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_phm08_score_penalizes_overestimate_more_than_underestimate():
    y_true = np.array([50.0])
    over = phm08_score(y_true, np.array([60.0]))   # d = +10
    under = phm08_score(y_true, np.array([40.0]))  # d = -10
    assert over > under


def test_phm08_score_zero_for_perfect_prediction():
    y = np.array([10.0, 50.0])
    assert phm08_score(y, y) == 0.0


def test_imminent_failure_prf_perfect_classifier():
    y_true = np.array([5, 15, 50, 100])
    y_pred = np.array([6, 14, 55, 90])
    result = imminent_failure_prf(y_true, y_pred, threshold=20)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["support_positive"] == 2
    assert result["support_total"] == 4
