from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _model(d: np.ndarray, a: float, b: float, k: float) -> np.ndarray:
    return a + b * np.expm1(k * d)


@dataclass
class SensorCurve:
    """sensor(d) = a + b*(exp(k*d) - 1) for degradation progress d in [0, 1].

    The standard exponential-wear shape for turbofan degradation (Saxena et
    al., PHM08): flat at the healthy baseline `a`, curving away from it as
    the engine approaches failure. `b`'s sign captures whether the sensor
    rises or falls with degradation; `k` is a shared curvature.
    """

    a: float
    b: float
    k: float

    def __call__(self, d: np.ndarray) -> np.ndarray:
        return _model(np.asarray(d, dtype=float), self.a, self.b, self.k)


def fit_sensor_curve(d: np.ndarray, y: np.ndarray) -> SensorCurve:
    """Fit (a, b, k) to pooled (degradation progress, sensor value) pairs."""
    healthy = d < 0.05
    a0 = y[healthy].mean() if healthy.any() else y.mean()
    corr = np.corrcoef(d, y)[0, 1]
    sign = 1.0 if np.isnan(corr) or corr == 0 else np.sign(corr)
    b0 = sign * (y.max() - y.min() + 1e-6)
    popt, _ = curve_fit(_model, d, y, p0=[a0, b0, 2.0], maxfev=20000)
    return SensorCurve(*popt)
