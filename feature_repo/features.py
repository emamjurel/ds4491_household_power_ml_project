"""Feast definitions for household energy features."""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

household = Entity(name="household_id", join_keys=["household_id"], description="The metered household")
power_source = FileSource(
    name="power_source",
    path="../data/processed/feast_source.parquet",
    timestamp_field="event_timestamp",
)
power_features = FeatureView(
    name="power_features",
    entities=[household],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Global_reactive_power", dtype=Float32), Field(name="Voltage", dtype=Float32),
        Field(name="Global_intensity", dtype=Float32), Field(name="Sub_metering_1", dtype=Float32),
        Field(name="Sub_metering_2", dtype=Float32), Field(name="Sub_metering_3", dtype=Float32),
        Field(name="hour", dtype=Int64), Field(name="day_of_week", dtype=Int64),
        Field(name="month", dtype=Int64), Field(name="active_power_lag_1", dtype=Float32),
    ],
    source=power_source,
    online=True,
)
