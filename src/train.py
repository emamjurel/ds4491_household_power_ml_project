"""Train a batch Random Forest with all fitted transformations learned on train only."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib, pandas as pd, yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocess import FEATURES, TARGET

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args(); p = yaml.safe_load(open(args.params))["model"]
    train = pd.read_csv("data/processed/train.csv")
    # fit() is deliberately called ONLY on training features: no test leakage.
    prep = ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)])
    model = Pipeline([("prep", prep), ("regressor", RandomForestRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=p["random_state"], n_jobs=-1))])
    model.fit(train[FEATURES], train[TARGET])
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/batch_model.joblib")
    print("Saved models/batch_model.joblib")
if __name__ == "__main__": main()
