"""Create a small, viva-friendly comparison between batch and online results."""
from pathlib import Path
import json

def main():
    batch = json.loads(Path("results/metrics.json").read_text())
    river = json.loads(Path("results/river_metrics.json").read_text())
    comparison = {"batch_random_forest": {k: batch[k] for k in ("mae", "rmse")}, "river_incremental_hoeffding_tree": {k: river[k] for k in ("mae", "rmse")}, "interpretation": "Lower MAE/RMSE is better. Batch is periodically retrained; River updates after every arriving row."}
    Path("results/model_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
if __name__ == "__main__": main()
