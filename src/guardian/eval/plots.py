import matplotlib.pyplot as plt
import numpy as np


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_pred - y_true, bins=40)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted - True RUL")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
