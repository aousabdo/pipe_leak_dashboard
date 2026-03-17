"""Pandera schemas for data validation."""

import pandera as pa
from pandera import Column, Check


pipes_schema = pa.DataFrameSchema(
    {
        "pipe_id": Column(str),
        "length_m": Column(float, Check.greater_than(0)),
        "diameter_m": Column(float, Check.greater_than(0)),
        "mid_lat": Column(float, Check.in_range(-90, 90)),
        "mid_lon": Column(float, Check.in_range(-180, 180)),
        "pressure_avg_m": Column(float),
        "age": Column(int, Check.greater_than_or_equal_to(0)),
        "material": Column(str),
        "installation_year": Column(int, Check.in_range(1900, 2030)),
        "soil_type": Column(str),
        "soil_corrosivity": Column(float, Check.in_range(0, 10)),
    },
    coerce=True,
)


events_schema = pa.DataFrameSchema(
    {
        "pipe_id": Column(str),
        "date": Column("datetime64[ns]"),
        "severity": Column(str, Check.isin(["Minor", "Moderate", "Major", "Critical"])),
        "flow_rate_gpm": Column(float, Check.greater_than(0)),
        "repair_cost": Column(float, Check.greater_than(0)),
        "water_loss_gallons": Column(float, Check.greater_than_or_equal_to(0)),
    },
    coerce=True,
)
