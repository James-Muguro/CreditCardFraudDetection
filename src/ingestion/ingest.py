import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # must run BEFORE importing kaggle — it auto-authenticates on import

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET = "mlg-ulb/creditcardfraud"
RAW_DIR = Path("data/raw")

def download_dataset(dataset: str, dest_dir: Path) -> Path:
    """Download and unzip a Kaggle dataset into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    logger.info(f"Downloading dataset '{dataset}' from Kaggle...")
    api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)
    logger.info("Download complete.")

    csv_files = list(dest_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found after download.")
    return csv_files[0]


def stamp_and_move(csv_path: Path, dest_dir: Path) -> Path:
    """Rename the raw file with an ingestion timestamp for traceability."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    new_name = dest_dir / f"creditcard_{timestamp}.csv"
    csv_path.rename(new_name)
    return new_name


def write_manifest(csv_path: Path, df: pd.DataFrame, dest_dir: Path) -> None:
    manifest = {
        "source": DATASET,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "file": csv_path.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "file_size_bytes": csv_path.stat().st_size,
    }
    manifest_path = dest_dir / f"{csv_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")


def main():
    raw_csv = download_dataset(DATASET, RAW_DIR)
    final_csv = stamp_and_move(raw_csv, RAW_DIR)

    df = pd.read_csv(final_csv)
    logger.info(f"Ingested {len(df)} rows, {len(df.columns)} columns from {final_csv.name}")

    write_manifest(final_csv, df, RAW_DIR)


if __name__ == "__main__":
    main()