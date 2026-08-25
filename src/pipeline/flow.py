import logging
from prefect import flow, task

from src.ingestion.ingest import main as run_ingest
from src.validation.validate import get_latest_raw_file, validate_file, RAW_DIR
from src.transformation.transform import main as run_transform
from src.storage.load import main as run_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@task(name="ingest", retries=2, retry_delay_seconds=10)
def ingest_task():
    run_ingest()


@task(name="validate")
def validate_task():
    latest_file = get_latest_raw_file(RAW_DIR)
    passed = validate_file(latest_file)
    if not passed:
        raise ValueError(f"Validation failed for {latest_file.name}. Halting pipeline.")


@task(name="transform")
def transform_task():
    run_transform()


@task(name="load")
def load_task():
    run_load()


@flow(name="creditcard-fraud-pipeline")
def creditcard_pipeline():
    ingest_task()
    validate_task()
    transform_task()
    load_task()


if __name__ == "__main__":
    creditcard_pipeline()