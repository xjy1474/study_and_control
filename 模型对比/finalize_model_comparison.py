import csv
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
UTILS_DIR = ROOT_DIR / "公共工具"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from common_model_utils import BASE_DIR, save_svg_line, try_import_matplotlib, write_summary_report


def read_comparison():
    path = BASE_DIR / "model_comparison.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({
                "model": row["model"],
                "MAE": float(row["MAE"]),
                "MSE": float(row["MSE"]),
                "RMSE": float(row["RMSE"]),
                "ACC": float(row["ACC"]),
                "R2": float(row["R2"]),
            })
    return rows


def save_comparison_plot(rows):
    plt = try_import_matplotlib()
    if plt is None:
        save_svg_line(
            BASE_DIR / "model_comparison.svg",
            [[row["RMSE"] for row in rows], [row["MAE"] for row in rows]],
            ["RMSE", "MAE"],
            "Model Comparison",
        )
        return

    models = [row["model"] for row in rows]
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, [row["MAE"] for row in rows], width, label="MAE")
    plt.bar(x + width / 2, [row["RMSE"] for row in rows], width, label="RMSE")
    plt.xticks(x, models)
    plt.ylabel("Error")
    plt.title("Model Error Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "model_comparison_error.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, [row["ACC"] for row in rows], width, label="ACC")
    plt.bar(x + width / 2, [row["R2"] for row in rows], width, label="R2")
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Model Score Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "model_comparison_score.png", dpi=160)
    plt.close()


def main():
    rows = read_comparison()
    rows = sorted(rows, key=lambda row: row["RMSE"])
    write_summary_report(rows)
    save_comparison_plot(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
