import pandas as pd
from src.transformation.transform import transform


def test_transform_adds_expected_columns():
    df = pd.DataFrame({
        "Time": [0.0, 100.0],
        "Amount": [10.0, 20.0],
        "Class": [0, 1],
    })
    result = transform(df)
    assert "amount_scaled" in result.columns
    assert "time_of_day_seconds" in result.columns
    assert len(result) == 2


def test_transform_drops_duplicates():
    df = pd.DataFrame({
        "Time": [0.0, 0.0],
        "Amount": [10.0, 10.0],
        "Class": [0, 0],
    })
    result = transform(df)
    assert len(result) == 1