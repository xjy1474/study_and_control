import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT_DIR / "公共工具"
for path in (ROOT_DIR, UTILS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_model_utils import save_basic_plots


def main():
    for result_dir in sorted((ROOT_DIR / "results").iterdir()):
        prediction_path = result_dir / "predictions.csv"
        if not prediction_path.exists():
            continue
        df = pd.read_csv(prediction_path, encoding="utf-8-sig")
        loss_history = None
        metrics_path = result_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            loss_history = metrics.get("loss_history")
        save_basic_plots(
            result_dir,
            df["timestamp"].astype(str),
            df["actual"].to_numpy(),
            df["predicted"].to_numpy(),
            loss_history,
        )
        print(f"updated {result_dir.name}")


if __name__ == "__main__":
    main()
