"""Evaluate the batch model and save metrics, predictions, and a visual check."""
from pathlib import Path
import json, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import FEATURES, TARGET

def main():
    test = pd.read_csv("data/processed/test.csv"); model = joblib.load("models/batch_model.joblib")
    pred = model.predict(test[FEATURES])
    metrics = {"task": "next-minute household active-power regression", "mae": float(mean_absolute_error(test[TARGET], pred)), "rmse": float(mean_squared_error(test[TARGET], pred) ** 0.5), "r2": float(r2_score(test[TARGET], pred)), "test_rows": int(len(test))}
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=2))
    out = test[["event_timestamp", TARGET]].copy(); out["prediction"] = pred; out.to_csv("results/predictions.csv", index=False)
    sample = out.head(500); plt.figure(figsize=(12,4)); plt.plot(sample[TARGET].to_numpy(), label="actual"); plt.plot(sample.prediction.to_numpy(), label="batch prediction"); plt.legend(); plt.xlabel("first 500 test minutes"); plt.ylabel("kW"); plt.tight_layout(); plt.savefig("results/plots/batch_predictions.png", dpi=150); plt.close()
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
