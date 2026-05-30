import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_split_csv(name):
    return pd.read_csv(BASE_DIR / name, encoding="utf-8-sig")


def feature_target_split(df):
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    return df[feature_cols].to_numpy(dtype=np.float32), df["target"].to_numpy(dtype=np.float32), feature_cols


def make_sequence_data(df, lookback=24, horizon=1):
    feature_cols = [c for c in df.columns if c not in ("timestamp", "target")]
    values = df[feature_cols].to_numpy(dtype=np.float32)
    target = df["target"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].astype(str).to_numpy()
    x, y, ts = [], [], []
    for i in range(lookback + horizon - 1, len(df)):
        start = i - horizon - lookback + 1
        end = i - horizon + 1
        x.append(values[start:end])
        y.append(target[i])
        ts.append(timestamps[i])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(ts), feature_cols


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    acc = float(1.0 - np.mean(np.abs(err) / denom))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot else 0.0
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "ACC": acc,
        "R2": r2,
    }


def save_metrics(result_dir, metrics):
    ensure_dir(result_dir)
    with open(Path(result_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_predictions(result_dir, timestamps, y_true, y_pred):
    ensure_dir(result_dir)
    with open(Path(result_dir) / "predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "actual", "predicted", "error"])
        for ts, actual, pred in zip(timestamps, y_true, y_pred):
            writer.writerow([ts, float(actual), float(pred), float(pred - actual)])


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def save_basic_plots(result_dir, timestamps, y_true, y_pred, loss_history=None):
    plt = try_import_matplotlib()
    if plt is None:
        save_svg_plots(result_dir, y_true, y_pred, loss_history)
        return

    result_dir = Path(result_dir)
    ensure_dir(result_dir)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual", linewidth=1.5)
    plt.plot(y_pred, label="Predicted", linewidth=1.2)
    plt.title("Prediction vs Actual")
    plt.xlabel("Sample")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "prediction_vs_actual.png", dpi=160)
    plt.close()

    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75, label="Prediction")
    plt.plot([low, high], [low, high], color="black", linewidth=1.2, label="Ideal fit")
    if len(y_true) >= 2:
        coef = np.polyfit(y_true, y_pred, deg=1)
        xs = np.linspace(low, high, 100)
        plt.plot(xs, coef[0] * xs + coef[1], color="#dc2626", linewidth=1.4, label="Fitted curve")
    plt.title("Curve Fitting Effect")
    plt.xlabel("Actual Load")
    plt.ylabel("Predicted Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "curve_fitting_effect.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(len(residual)), residual, s=10)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residual Plot")
    plt.xlabel("Sample")
    plt.ylabel("Prediction Error")
    plt.tight_layout()
    plt.savefig(result_dir / "residual_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(residual, bins=30)
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(result_dir / "error_distribution.png", dpi=160)
    plt.close()

    if loss_history:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history, label="Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / "loss_curve.png", dpi=160)
        plt.close()


def _points(values, width, height, pad):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return ""
    mn, mx = float(np.min(values)), float(np.max(values))
    if abs(mx - mn) < 1e-9:
        mx = mn + 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + (width - 2 * pad) * i / max(1, len(values) - 1)
        y = height - pad - (height - 2 * pad) * (float(v) - mn) / (mx - mn)
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)


def save_svg_line(path, series, labels, title):
    width, height, pad = 1000, 420, 45
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    lines = []
    for idx, values in enumerate(series):
        lines.append(
            f'<polyline fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2" points="{_points(values, width, height, pad)}" />'
        )
    legend = " ".join(
        f'<text x="{pad + i * 150}" y="28" font-size="14" fill="{colors[i % len(colors)]}">{labels[i]}</text>'
        for i in range(len(labels))
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{pad}" y="18" font-size="16" fill="#111827">{title}</text>
{legend}
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#111827"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#111827"/>
{''.join(lines)}
</svg>"""
    Path(path).write_text(svg, encoding="utf-8")


def save_svg_plots(result_dir, y_true, y_pred, loss_history=None):
    result_dir = Path(result_dir)
    ensure_dir(result_dir)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    save_svg_line(result_dir / "prediction_vs_actual.svg", [y_true, y_pred], ["Actual", "Predicted"], "Prediction vs Actual")
    save_svg_line(result_dir / "curve_fitting_effect.svg", [np.sort(y_true), y_pred[np.argsort(y_true)]], ["Fitted"], "Curve Fitting Effect")
    save_svg_line(result_dir / "residual_plot.svg", [residual], ["Residual"], "Residual Plot")
    if loss_history:
        save_svg_line(result_dir / "loss_curve.svg", [loss_history], ["Loss"], "Loss Curve")


def write_summary_report(comparison_rows):
    path = BASE_DIR / "model_analysis_report.md"
    rows = sorted(comparison_rows, key=lambda r: (r.get("RMSE", float("inf")), -r.get("ACC", -999)))
    best = rows[0] if rows else None
    lines = ["# 模型效果分析", ""]
    if best:
        lines.append(f"综合 RMSE、MAE、R2、ACC 指标，当前效果最好的模型是 **{best['model']}**。")
        lines.append("")
    lines.append("| 模型 | MAE | MSE | RMSE | ACC | R2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['MAE']:.6f} | {row['MSE']:.6f} | {row['RMSE']:.6f} | {row['ACC']:.6f} | {row['R2']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def update_model_comparison(model_name, metrics):
    path = BASE_DIR / "model_comparison.csv"
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [row for row in rows if row.get("model") != model_name]

    row = {"model": model_name}
    row.update({key: metrics.get(key, "") for key in ["MAE", "MSE", "RMSE", "ACC", "R2"]})
    rows.append(row)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "MAE", "MSE", "RMSE", "ACC", "R2"])
        writer.writeheader()
        writer.writerows(rows)

    numeric_rows = []
    for item in rows:
        try:
            numeric_rows.append({
                "model": item["model"],
                "MAE": float(item["MAE"]),
                "MSE": float(item["MSE"]),
                "RMSE": float(item["RMSE"]),
                "ACC": float(item["ACC"]),
                "R2": float(item["R2"]),
            })
        except (TypeError, ValueError):
            pass
    write_summary_report(numeric_rows)
