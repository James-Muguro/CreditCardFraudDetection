import pandas as pd
from pathlib import Path
from src.storage.load import load_to_warehouse
import duckdb


def test_load_to_warehouse(tmp_path):
    # Arrange: write a tiny parquet file
    df = pd.DataFrame({"Time": [0.0, 1.0], "Amount": [5.0, 10.0], "Class": [0, 1]})
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)

    db_path = tmp_path / "test.duckdb"

    # Act
    row_count = load_to_warehouse(parquet_path, db_path, "transactions")

    # Assert
    assert row_count == 2
    con = duckdb.connect(str(db_path))
    result = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    con.close()
    assert result == 2