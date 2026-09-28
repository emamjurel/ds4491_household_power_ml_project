"""Show Feast point-in-time historical retrieval and one online lookup."""
from pathlib import Path
import pandas as pd
from feast import FeatureStore

FEATURE_REFS = ["power_features:Global_reactive_power", "power_features:Voltage", "power_features:Global_intensity", "power_features:Sub_metering_1", "power_features:Sub_metering_2", "power_features:Sub_metering_3", "power_features:hour", "power_features:day_of_week", "power_features:month", "power_features:active_power_lag_1"]

def main():
    store = FeatureStore(repo_path="feature_repo")
    train = pd.read_csv("data/processed/train.csv", parse_dates=["event_timestamp"])
    # Match the hourly Feast source cadence for a compact historical demo.
    entities = train.iloc[::60][["household_id", "event_timestamp"]].head(100)
    historical = store.get_historical_features(entity_df=entities, features=FEATURE_REFS).to_df()
    Path("results").mkdir(exist_ok=True); historical.to_csv("results/feast_historical_features.csv", index=False)
    # The online store contains each entity's most recently materialized feature values.
    online = store.get_online_features(features=FEATURE_REFS, entity_rows=[{"household_id": "household_1"}]).to_dict()
    Path("results/feast_online_example.json").write_text(pd.Series(online).to_json(indent=2))
    print("Saved historical point-in-time features and an online feature example.")
if __name__ == "__main__": main()
