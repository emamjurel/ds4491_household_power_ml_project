# DS-4491: Household Power ML Systems Design

This project predicts **next-minute `Global_active_power` (kW)** for one household. It uses DVC for reproducible batch ML, Feast for batch/online feature retrieval, and River for incremental learning.

## Why this task?

The input data is minute-level household electricity measurements. `Global_active_power` is a continuous, meaningful target. Predictors are electrical readings, calendar fields, and `active_power_lag_1` (the previous minute's known reading). The default `max_rows: 200000` makes it quick to demonstrate; set it to `null` in `params.yaml` for all rows.

## Leakage prevention

Rows are sorted by timestamp and split **chronologically**: the earliest 80% is training, the most recent 20% is test. The batch `SimpleImputer` and `StandardScaler` are fitted inside a scikit-learn pipeline on training data only. The lag feature is `shift(1)`, so it only uses the prior observation.

## Project structure

```text
data/raw/                         original CSV (DVC tracked)
data/processed/                   train.csv, test.csv, Feast parquet source
src/preprocess.py                 chronological split + features
src/train.py                      batch RandomForest pipeline
src/evaluate.py                   metrics, predictions and plot
src/online_learning.py            River incremental baseline
src/fetch_features.py             Feast retrieval examples
feature_repo/                     Feast definitions/configuration
models/batch_model.joblib         batch model output
results/                          metrics, comparisons and plots
dvc.yaml / dvc.lock / params.yaml reproducible pipeline
```

## Setup and run

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ds4491_power_project
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# first time only, if cloning a version without .dvc already initialised
dvc init

# retrieve raw data from the configured DVC remote (after you configure one)
dvc pull

# run the three-stage pipeline
dvc repro
dvc status
dvc dag

# run online learning and Feast demonstrations
python src/online_learning.py
cd feature_repo
feast apply
# This dataset's timestamps are in 2006–2007, so materialize that range:
feast materialize 2006-12-16T17:24:00 2007-05-05T00:00:00
cd ..
python src/fetch_features.py
python src/compare_results.py
```

### DVC commands for your viva

```bash
dvc init                              # initialise DVC in a Git project
dvc add data/raw/household_power_consumption.csv  # track raw data, creates .dvc pointer
git add data/raw/household_power_consumption.csv.dvc .gitignore
dvc repro                             # reproduce preprocessing → training → evaluation
dvc status                            # show what is out of date
dvc dag                               # show dependency graph
dvc remote add -d storage <REMOTE_URL> # e.g. an S3/Google Drive remote you own
dvc push                              # upload tracked data to remote
dvc pull                              # download tracked data from remote
```

Commit the code, `dvc.yaml`, `dvc.lock`, `params.yaml`, and the `.dvc` pointer to Git. Do **not** commit the large raw CSV itself.

## DVC pipeline

`raw CSV → preprocessing → batch training → evaluation → results`

`dvc.yaml` defines the stages, dependencies, parameters, and outputs. `dvc.lock` records exact hashes after `dvc repro`, so the same pipeline version can be reproduced later. Results are saved to `results/metrics.json`, `results/predictions.csv`, and `results/plots/batch_predictions.png`.

## Feast

`feature_repo/features.py` defines the `household_id` entity and `power_features` feature view. The offline file store reads `data/processed/feast_source.parquet`.

- **Offline / historical retrieval:** `get_historical_features()` joins features to dated training entity rows. Feast uses point-in-time correctness, preventing feature values from after the event timestamp being joined to training rows.
- **Online retrieval:** `feast materialize-incremental` writes the latest feature values for each `household_id` into the local SQLite online store. `get_online_features()` returns the latest values for `household_1`—the pattern used by a live prediction service.

`src/fetch_features.py` saves a historical retrieval sample to `results/feast_historical_features.csv` and one online example to `results/feast_online_example.json`.

## River online learning

`src/online_learning.py` first streams historical training records through a `HoeffdingTreeRegressor` using `learn_one()`. It then streams test-period records in time order using **predict → update MAE/RMSE → `learn_one()`**. This incremental tree is suited to changing, streaming energy data and writes `results/river_metrics.json`.

Run `python src/compare_results.py` after both pipelines. It writes `results/model_comparison.json`, comparing the batch Random Forest test MAE/RMSE against River's prequential MAE/RMSE. The comparison is useful in a viva: batch training generally provides a stronger fixed model, while River updates immediately without a full retrain.

## Expected output

Exact metrics depend on package versions and `max_rows`. After running, use the values in `results/metrics.json` and `results/river_metrics.json` in your report rather than claiming fixed values.
