import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def get_latest_raw_file(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.glob("creditcard_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No raw CSV files found. Run ingestion first.")
    return csv_files[-1]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature transformations.

    Kept intentionally simple and explicit so each transformation
    step is traceable and testable in isolation.
    """
    df = df.copy()

    # Drop exact duplicate rows — common data quality issue in this dataset
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} duplicate rows.")

    # Normalize Amount into a scaled column, keep original for traceability
    df["amount_scaled"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # Convert Time (seconds since first transaction) into an hour-of-day-like cyclical feature
    seconds_in_day = 24 * 60 * 60
    df["time_of_day_seconds"] = df["Time"] % seconds_in_day

    return df


def write_processed(df: pd.DataFrame, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = dest_dir / f"creditcard_processed_{timestamp}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_manifest(out_path: Path, df: pd.DataFrame, dest_dir: Path) -> None:
    manifest = {
        "transformed_at": datetime.now(timezone.utc).isoformat(),
        "file": out_path.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "file_size_bytes": out_path.stat().st_size,
    }
    manifest_path = dest_dir / f"{out_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")


def main():
    raw_file = get_latest_raw_file(RAW_DIR)
    logger.info(f"Reading {raw_file.name}...")
    df = pd.read_csv(raw_file)

    transformed_df = transform(df)
    logger.info(f"Transformed to {len(transformed_df)} rows, {len(transformed_df.columns)} columns.")

    out_path = write_processed(transformed_df, PROCESSED_DIR)
    logger.info(f"Processed file written to {out_path}")

    write_manifest(out_path, transformed_df, PROCESSED_DIR)


if __name__ == "__main__":
    main()