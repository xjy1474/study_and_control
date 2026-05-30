import json
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
    RESULTS_DIR,
    feature_target_split,
    load_split_csv,
    regression_metrics,
    save_basic_plots,
    save_metrics,
    save_predictions,
    update_model_comparison,
)


MODEL_NAME = "xgboost"
FALLBACK_ROUNDS = 120
LEARNING_RATE = 0.05


class StumpBoostRegressor:
    def __init__(self, n_estimators=120, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_value = 0.0
        self.stumps = []

    def fit(self, x, y):
        self.base_value = float(np.mean(y))
        pred = np.full_like(y, self.base_value, dtype=np.float64)
        for _ in range(self.n_estimators):
            residual = y - pred
            best = None
            best_loss = float("inf")
            for feature_idx in range(x.shape[1]):
                values = x[:, feature_idx]
                thresholds = np.quantile(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                for threshold in thresholds:
                    left = values <= threshold
                    right = ~left
                    if left.sum() == 0 or right.sum() == 0:
                        continue
                    left_value = residual[left].mean()
                    right_value = residual[right].mean()
                    update = np.where(left, left_value, right_value)
                    loss = np.mean((residual - update) ** 2)
                    if loss < best_loss:
                        best_loss = loss
                        best = (feature_idx, float(threshold), float(left_value), float(right_value))
            if best is None:
                break
            self.stumps.append(best)
            feature_idx, threshold, left_value, right_value = best
            pred += self.learning_rate * np.where(x[:, feature_idx] <= threshold, left_value, right_value)

    def predict(self, x):
        pred = np.full(x.shape[0], self.base_value, dtype=np.float64)
        for feature_idx, threshold, left_value, right_value in self.stumps:
            pred += self.learning_rate * np.where(x[:, feature_idx] <= threshold, left_value, right_value)
        return pred


def train_native_xgboost(x_train, y_train):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=300)

    class NativeBoosterWrapper:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def predict(self, x):
            return self.wrapped.predict(xgb.DMatrix(x))

    model = NativeBoosterWrapper(booster)
    return model, "native_xgboost"


def main():
    train_df = load_split_csv("train_dataset.csv")
    test_df = load_split_csv("test_dataset.csv")
    validation_df = load_split_csv("validation_dataset.csv")

    x_train, y_train, feature_cols = feature_target_split(train_df)
    x_test, y_test, _ = feature_target_split(test_df)
    x_val, y_val, _ = feature_target_split(validation_df)

    try:
        model, backend = train_native_xgboost(x_train, y_train)
    except Exception:
        model = StumpBoostRegressor(n_estimators=FALLBACK_ROUNDS, learning_rate=LEARNING_RATE)
        model.fit(x_train.astype(np.float64), y_train.astype(np.float64))
        backend = "numpy_gradient_boosted_stumps_fallback"

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    pred_val = model.predict(x_val)

    metrics = {
        "model": MODEL_NAME,
        "backend": backend,
        "feature_columns": feature_cols,
        "train": regression_metrics(y_train, pred_train),
        "test": regression_metrics(y_test, pred_test),
        "validation": regression_metrics(y_val, pred_val),
    }

    result_dir = RESULTS_DIR / MODEL_NAME
    save_metrics(result_dir, metrics)
    save_predictions(result_dir, validation_df["timestamp"].astype(str), y_val, pred_val)
    save_basic_plots(result_dir, validation_df["timestamp"].astype(str), y_val, pred_val)
    if backend == "numpy_gradient_boosted_stumps_fallback":
        with open(result_dir / "fallback_model.json", "w", encoding="utf-8") as f:
            json.dump({"base_value": model.base_value, "stumps": model.stumps}, f, ensure_ascii=False, indent=2)
    update_model_comparison(MODEL_NAME, metrics["validation"])
    print(metrics)


if __name__ == "__main__":
    main()
