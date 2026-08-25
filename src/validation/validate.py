import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera as pa

from src.validation.schema import credit_card_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")


def get_latest_raw_file(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.glob("creditcard_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No raw CSV files found. Run ingestion first.")
    return csv_files[-1]


def validate_file(csv_path: Path) -> bool:
    df = pd.read_csv(csv_path)
    logger.info(f"Validating {csv_path.name} ({len(df)} rows)...")

    try:
        credit_card_schema.validate(df, lazy=True)
        logger.info("Validation PASSED.")
        return True
    except pa.errors.SchemaErrors as err:
        logger.error("Validation FAILED.")
        logger.error(err.failure_cases.to_string())
        report_path = csv_path.parent / f"{csv_path.stem}_validation_errors.json"
        err.failure_cases.to_json(report_path, orient="records", indent=2)
        logger.error(f"Failure detail written to {report_path}")
        return False


def main():
    latest_file = get_latest_raw_file(RAW_DIR)
    passed = validate_file(latest_file)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()