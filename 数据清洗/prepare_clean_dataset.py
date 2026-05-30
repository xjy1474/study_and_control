import csv
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
UTILS_DIR = ROOT_DIR / "公共工具"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


BASE_DIR = ROOT_DIR
SOURCE_DIR = BASE_DIR / "采集数据"
TARGET_KEYWORDS = ["机组实际负荷"]
MISSING_THRESHOLD = 0.40
CORR_THRESHOLD = 0.30
MIN_FEATURES = 20
MAX_FEATURES = 40
RESAMPLE_RULE = "1h"


def clean_column_name(path: Path):
    name = path.stem.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\u4e00-\u9fff#]+", "_", name)
    return name.strip("_")


def read_signal(path: Path):
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True, usecols=["Value", "Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Value"])
    df = df.sort_values("Timestamp")
    return df.groupby("Timestamp", as_index=True)["Value"].mean()


def cap_outliers_iqr(df, columns):
    capped = df.copy()
    caps = {}
    for col in columns:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lower = capped[col].quantile(0.01)
            upper = capped[col].quantile(0.99)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower, upper)
        caps[col] = {"lower": float(lower), "upper": float(upper)}
    return capped, caps


def standardize_features(df, feature_cols, train_end):
    result = df.copy()
    params = {}
    train = result.iloc[:train_end]
    for col in feature_cols:
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        result[col] = (result[col] - mean) / std
        params[col] = {"mean": float(mean), "std": float(std)}
    return result, params


def main():
    csv_files = sorted(SOURCE_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("采集数据文件夹中没有 CSV 文件。")

    series = {}
    target_col = None
    read_errors = []
    for path in csv_files:
        try:
            col = clean_column_name(path)
            signal = read_signal(path)
            if len(signal) == 0:
                continue
            if col in series:
                col = f"{col}_{len(series)}"
            series[col] = signal
            if any(keyword in path.name for keyword in TARGET_KEYWORDS):
                target_col = col
        except Exception as exc:
            read_errors.append({"file": path.name, "error": str(exc)})

    if target_col is None:
        raise RuntimeError("未找到机组实际负荷文件，无法确定 target。")

    raw = pd.DataFrame(series).sort_index()
    hourly = raw.resample(RESAMPLE_RULE).mean()
    hourly = hourly.dropna(subset=[target_col])

    missing_rate = hourly.isna().mean()
    keep_cols = [col for col in hourly.columns if col == target_col or missing_rate[col] <= MISSING_THRESHOLD]
    hourly = hourly[keep_cols]
    hourly = hourly.ffill().bfill()
    hourly = hourly.dropna(axis=1, how="any")

    numeric_cols = list(hourly.columns)
    cleaned, caps = cap_outliers_iqr(hourly, numeric_cols)
    cleaned = cleaned.rename(columns={target_col: "target"})

    feature_cols = [c for c in cleaned.columns if c != "target"]
    correlations = []
    for col in feature_cols:
        corr = cleaned[col].corr(cleaned["target"])
        if pd.isna(corr):
            corr = 0.0
        correlations.append({
            "feature": col,
            "pearson_correlation": float(corr),
            "abs_pearson_correlation": float(abs(corr)),
            "missing_rate_before_fill": float(missing_rate.get(col, 0.0)),
        })

    corr_df = pd.DataFrame(correlations).sort_values("abs_pearson_correlation", ascending=False)
    selected = corr_df[corr_df["abs_pearson_correlation"] >= CORR_THRESHOLD]["feature"].tolist()
    if len(selected) < MIN_FEATURES:
        selected = corr_df.head(min(MAX_FEATURES, max(MIN_FEATURES, len(corr_df))))["feature"].tolist()
    else:
        selected = selected[:MAX_FEATURES]

    model_df = cleaned[selected + ["target"]].copy()
    model_df.insert(0, "timestamp", model_df.index.strftime("%Y-%m-%d %H:%M:%S"))

    n = len(model_df)
    train_end = int(n * 0.8)
    test_end = int(n * 0.9)

    standardized_features, standardization_params = standardize_features(model_df, selected, train_end)

    model_df.to_csv(BASE_DIR / "cleaned_model_dataset_raw.csv", index=False, encoding="utf-8-sig")
    standardized_features.to_csv(BASE_DIR / "cleaned_model_dataset.csv", index=False, encoding="utf-8-sig")
    standardized_features.iloc[:train_end].to_csv(BASE_DIR / "train_dataset.csv", index=False, encoding="utf-8-sig")
    standardized_features.iloc[train_end:test_end].to_csv(BASE_DIR / "test_dataset.csv", index=False, encoding="utf-8-sig")
    standardized_features.iloc[test_end:].to_csv(BASE_DIR / "validation_dataset.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(BASE_DIR / "feature_correlation_report.csv", index=False, encoding="utf-8-sig")

    with (BASE_DIR / "standardization_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(standardization_params, f, ensure_ascii=False, indent=2)

    summary = {
        "source_file_count": len(csv_files),
        "loaded_signal_count": len(series),
        "read_error_count": len(read_errors),
        "target_source_column": target_col,
        "resample_rule": RESAMPLE_RULE,
        "missing_threshold": MISSING_THRESHOLD,
        "correlation_threshold": CORR_THRESHOLD,
        "row_count": int(n),
        "feature_count": len(selected),
        "train_rows": int(train_end),
        "test_rows": int(test_end - train_end),
        "validation_rows": int(n - test_end),
        "first_timestamp": str(model_df["timestamp"].iloc[0]) if n else "",
        "last_timestamp": str(model_df["timestamp"].iloc[-1]) if n else "",
        "selected_features": selected,
        "read_errors": read_errors[:20],
        "outlier_caps": caps,
    }
    with (BASE_DIR / "dataset_preparation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
