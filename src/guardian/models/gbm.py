from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class GBMConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_samples: int = 20
    random_state: int = 0
    extra_params: dict = field(default_factory=dict)


class GBMBaseline:
    """LightGBM regressor over per-cycle sensor readings -> RUL.

    Each row (engine, cycle) is treated as an independent sample. This ignores
    temporal context within an engine's trajectory, which is exactly the gap
    the LSTM comparison is meant to probe.
    """

    def __init__(self, config: GBMConfig | None = None):
        self.config = config or GBMConfig()
        self.model = lgb.LGBMRegressor(
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            min_child_samples=self.config.min_child_samples,
            random_state=self.config.random_state,
            **self.config.extra_params,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "GBMBaseline":
        callbacks = []
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        self.model.fit(X, y, callbacks=callbacks, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
