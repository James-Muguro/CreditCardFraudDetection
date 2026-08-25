import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/warehouse/creditcard.duckdb")
TABLE_NAME = "transactions"


def get_latest_processed_file(processed_dir: Path) -> Path:
    parquet_files = sorted(processed_dir.glob("creditcard_processed_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No processed Parquet files found. Run transformation first.")
    return parquet_files[-1]


def load_to_warehouse(parquet_path: Path, db_path: Path, table_name: str) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        # Full refresh load: replace the table each run.
        # (Incremental/append strategies come later once orchestration exists.)
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_parquet('{parquet_path.as_posix()}')
        """)
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return row_count
    finally:
        con.close()


def main():
    parquet_file = get_latest_processed_file(PROCESSED_DIR)
    logger.info(f"Loading {parquet_file.name} into {DB_PATH}...")

    row_count = load_to_warehouse(parquet_file, DB_PATH, TABLE_NAME)
    logger.info(f"Loaded {row_count} rows into table '{TABLE_NAME}'.")


if __name__ == "__main__":
    main()