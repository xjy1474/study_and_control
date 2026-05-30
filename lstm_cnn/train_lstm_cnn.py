import os
import sys

import numpy as np
import pandas as pd

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
UTILS_DIR = ROOT_DIR / "公共工具"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from common_model_utils import (
    BASE_DIR,
    RESULTS_DIR,
    load_split_csv,
    make_sequence_data,
    regression_metrics,
    save_basic_plots,
    save_metrics,
    save_predictions,
    update_model_comparison,
)


MODEL_NAME = "lstm_cnn"
LOOKBACK = 24
HORIZON = 1
EPOCHS = 40
BATCH_SIZE = 32


def build_combined_df():
    return pd.concat(
        [
            load_split_csv("train_dataset.csv"),
            load_split_csv("test_dataset.csv"),
            load_split_csv("validation_dataset.csv"),
        ],
        ignore_index=True,
    )


def split_sequences(x, y, timestamps, source_len):
    train_end = int(source_len * 0.8)
    test_end = int(source_len * 0.9)
    target_indices = np.arange(LOOKBACK + HORIZON - 1, source_len)
    train_mask = target_indices < train_end
    test_mask = (target_indices >= train_end) & (target_indices < test_end)
    val_mask = target_indices >= test_end
    return (
        (x[train_mask], y[train_mask], timestamps[train_mask]),
        (x[test_mask], y[test_mask], timestamps[test_mask]),
        (x[val_mask], y[val_mask], timestamps[val_mask]),
    )


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except Exception as exc:
        raise RuntimeError(
            "缺少 tensorflow，无法训练 LSTM+CNN。请使用已安装 tensorflow 的 Python 环境运行本脚本。"
        ) from exc

    df = build_combined_df()
    x, y, timestamps, feature_cols = make_sequence_data(df, lookback=LOOKBACK, horizon=HORIZON)
    (x_train, y_train, _), (x_test, y_test, _), (x_val, y_val, ts_val) = split_sequences(x, y, timestamps, len(df))
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(LOOKBACK, len(feature_cols))),
            layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="causal"),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(32),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="mae", metrics=["mse"])
    history = model.fit(
        x_train,
        y_train_scaled,
        validation_data=(x_test, y_test_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    pred_train = model.predict(x_train, verbose=0).reshape(-1) * y_std + y_mean
    pred_test = model.predict(x_test, verbose=0).reshape(-1) * y_std + y_mean
    pred_val = model.predict(x_val, verbose=0).reshape(-1) * y_std + y_mean

    metrics = {
        "model": MODEL_NAME,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "target_mean": y_mean,
        "target_std": y_std,
        "feature_columns": feature_cols,
        "train": regression_metrics(y_train, pred_train),
        "test": regression_metrics(y_test, pred_test),
        "validation": regression_metrics(y_val, pred_val),
        "loss_history": [float(v) for v in history.history.get("loss", [])],
        "validation_loss_history": [float(v) for v in history.history.get("val_loss", [])],
    }

    result_dir = RESULTS_DIR / MODEL_NAME
    save_metrics(result_dir, metrics)
    save_predictions(result_dir, ts_val, y_val, pred_val)
    save_basic_plots(result_dir, ts_val, y_val, pred_val, metrics["loss_history"])
    model.save(result_dir / "model.keras")
    update_model_comparison(MODEL_NAME, metrics["validation"])
    print(metrics)


if __name__ == "__main__":
    main()
