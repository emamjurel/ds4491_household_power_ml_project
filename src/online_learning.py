"""Prequential River baseline: predict, score, then learn one record at a time."""
from pathlib import Path
import json
import pandas as pd
from river import metrics, tree
from preprocess import FEATURES, TARGET

def row_features(row):
    return {f: float(row[f]) for f in FEATURES if pd.notna(row[f])}

def main():
    # First stream historical training rows, one record at a time.
    # Then evaluate and keep adapting on future held-out rows.
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    model = tree.HoeffdingTreeRegressor(grace_period=200, delta=1e-5)
    for _, row in train.iterrows():
        model.learn_one(row_features(row), float(row[TARGET]))
    mae, rmse = metrics.MAE(), metrics.RMSE()
    predictions = 0
    for _, row in test.iterrows():
        x, y = row_features(row), float(row[TARGET])
        y_pred = model.predict_one(x)
        if y_pred is not None:
            mae.update(y, y_pred); rmse.update(y, y_pred); predictions += 1
        # learn_one updates immediately, which suits changing/streaming energy data.
        model.learn_one(x, y)
    result = {"model": "River HoeffdingTreeRegressor", "protocol": "chronological train stream, then prequential test (predict then learn_one)", "mae": float(mae.get()), "rmse": float(rmse.get()), "scored_rows": predictions}
    Path("results").mkdir(exist_ok=True); Path("results/river_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
