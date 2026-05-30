from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
UTILS_DIR = ROOT_DIR / "公共工具"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from common_model_utils import (
    BASE_DIR,
    RESULTS_DIR,
    feature_target_split,
    load_split_csv,
    regression_metrics,
    save_basic_plots,
    save_metrics,
    save_predictions,
    update_model_comparison,
)


MODEL_NAME = "regression"
RIDGE_ALPHA = 1.0


def add_intercept(x):
    return np.hstack([np.ones((x.shape[0], 1), dtype=x.dtype), x])


def train_ridge_regression(x, y, alpha):
    x_i = add_intercept(x)
    reg = np.eye(x_i.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    return np.linalg.pinv(x_i.T @ x_i + reg) @ x_i.T @ y


def predict(x, coef):
    return add_intercept(x) @ coef


def main():
    train_df = load_split_csv("train_dataset.csv")
    test_df = load_split_csv("test_dataset.csv")
    validation_df = load_split_csv("validation_dataset.csv")

    x_train, y_train, _ = feature_target_split(train_df)
    x_test, y_test, _ = feature_target_split(test_df)
    x_val, y_val, _ = feature_target_split(validation_df)

    coef = train_ridge_regression(x_train.astype(np.float64), y_train.astype(np.float64), RIDGE_ALPHA)
    pred_train = predict(x_train, coef)
    pred_test = predict(x_test, coef)
    pred_val = predict(x_val, coef)

    metrics = {
        "model": MODEL_NAME,
        "ridge_alpha": RIDGE_ALPHA,
        "train": regression_metrics(y_train, pred_train),
        "test": regression_metrics(y_test, pred_test),
        "validation": regression_metrics(y_val, pred_val),
    }

    result_dir = RESULTS_DIR / MODEL_NAME
    save_metrics(result_dir, metrics)
    save_predictions(result_dir, validation_df["timestamp"].astype(str), y_val, pred_val)
    save_basic_plots(result_dir, validation_df["timestamp"].astype(str), y_val, pred_val)
    np.save(result_dir / "coefficients.npy", coef)
    update_model_comparison(MODEL_NAME, metrics["validation"])
    print(metrics)


if __name__ == "__main__":
    main()
