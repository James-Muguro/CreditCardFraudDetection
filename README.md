# Credit Card Fraud Detection — Data Pipeline

A production-style data engineering pipeline that ingests, validates, transforms, and loads credit card transaction data into a queryable analytical warehouse. The pipeline is orchestrated end-to-end with Prefect, containerized with Docker for reproducible runs, and validated on every push via GitHub Actions.

**Dataset:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 31 features, ~0.17% fraud rate

---

## The Problem

Turning an externally hosted dataset into a reliable analytical asset requires more than downloading a CSV and loading it into a database. This project addresses the engineering challenges of:

- **External data dependency:** The source lives on Kaggle and must be fetched programmatically with API credentials
- **Schema reliability:** No guarantees that upstream data will remain consistent; explicit validation is required before downstream stages consume it
- **Data quality:** Class imbalance (~0.17% fraud), potential duplicates, and type/range violations must be caught early
- **Reproducibility:** The pipeline must run identically on any machine without manual intervention
- **Transformation consistency:** Feature engineering, scaling, and time transformations must be deterministic and versioned
- **Downstream data integrity:** Bad data must not propagate silently to storage; validation must act as a hard gate
- **Pipeline failure handling:** Orchestration must support retries, clear failure modes, and observable task execution

The fraud detection use case provides business context, but the core engineering problem is building a reliable, reproducible data pipeline that prepares transaction data for downstream analysis.

---

## Architecture

```text
┌─────────────────┐
│   Kaggle API    │
│  (external src) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │
│  (kaggle API)   │
│  → raw CSV      │
│  → manifest     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │
│  (pandera)      │
│  → schema gate  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformation  │
│  (pandas)       │
│  → Parquet      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Storage      │
│   (DuckDB)      │
│  → warehouse    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orchestration  │
│   (Prefect)     │
│  → DAG flow     │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│   Container     │  │       CI        │
│    (Docker)     │  │ (GitHub Actions)│
│  → repro runs   │  │  → pytest       │
└─────────────────┘  └─────────────────┘
```

### Pipeline Stages

| Stage | Responsibility | Technology | Output |
|-------|----------------|------------|--------|
| **Ingestion** | Pulls dataset from Kaggle API, timestamps raw file, writes manifest for traceability | `kaggle` API | Raw CSV + manifest JSON |
| **Validation** | Validates raw data against explicit schema (types, ranges, nullability) before downstream processing | `pandera` | Validation gate (pass/fail) |
| **Transformation** | Deduplicates records, engineers features (scaled amount, time-of-day), writes typed columnar output | `pandas`, `pyarrow` | Parquet files |
| **Storage** | Loads processed data into local analytical warehouse | `DuckDB` | DuckDB database with typed tables |
| **Orchestration** | Chains ingestion → validation → transformation → load into single flow with retries and hard validation gate | `Prefect` | Orchestrated DAG execution |
| **Containerization** | Packages full pipeline so it runs identically on any machine | `Docker`, `docker compose` | Reproducible container runtime |
| **CI** | Runs unit test suite and schema sanity check on every push/PR | `GitHub Actions`, `pytest` | Automated code validation |

---

## Pipeline Design

The pipeline is decomposed into independently runnable, independently tested modules. Each stage has a clear input, processing logic, validation, output, and failure behavior.

### Ingestion

**Input:** Kaggle API credentials (environment variables)

**Processing:**
- Authenticates to Kaggle using `KAGGLE_USERNAME` and `KAGGLE_KEY`
- Downloads the dataset to `data/raw/` with timestamped filename
- Generates a manifest JSON containing:
  - Source dataset identifier
  - Download timestamp
  - File path
  - Row count (if available)
  - Checksum (if computed)

**Output:** Raw CSV file + manifest JSON in `data/raw/`

**Failure behavior:** Pipeline halts if API authentication fails or download is incomplete. No downstream stages execute without a valid raw file and manifest.

**Traceability:** The manifest provides an audit trail linking the raw file to its source and download time.

### Validation

**Input:** Raw CSV from ingestion stage

**Processing:**
- Loads data into a Pandas DataFrame
- Applies a Pandera schema that enforces:
  - Required columns: `Time`, `V1`–`V28`, `Amount`, `Class`
  - Data types: All numeric (int/float)
  - Nullability: No nulls allowed in any column
  - Value constraints:
    - `Class` must be binary (0 or 1)
    - `Amount` must be non-negative
    - `Time` must be non-negative
  - Class distribution check: Validates that fraud rate is within expected bounds (minority class ~0.17%)

**Output:** Validation result (pass/fail)

**Failure behavior:** If validation fails, the pipeline halts immediately. No transformation or storage occurs. Bad data does not propagate downstream.

**Why before transformation:** Validation acts as a quality gate. Transforming invalid data would compound errors and make debugging harder.

### Transformation

**Input:** Validated raw DataFrame

**Processing:**
- **Deduplication:** Removes exact duplicate rows (if any exist)
- **Feature engineering:**
  - `amount_scaled`: StandardScaler normalization of `Amount`
  - `time_of_day`: Hour-of-day derived from `Time` (seconds since first transaction)
  - `is_night`: Boolean flag for transactions between 22:00–06:00
- **Data type conversions:** Ensures all columns are explicitly typed
- **Column ordering:** Standardizes column order for downstream consistency

**Output:** Parquet file in `data/processed/`

**Why Parquet:** Columnar format, efficient compression, native DuckDB support, and type safety.

### Storage

**Input:** Processed Parquet file

**Processing:**
- Creates or recreates DuckDB database at `data/warehouse/transactions.duckdb`
- Loads Parquet into a table named `transactions`
- Schema mirrors Parquet columns with explicit types
- Creates indexes on `Class` and `time_of_day` for common query patterns

**Output:** DuckDB database with `transactions` table

**Load method:** Full reload (not incremental) — appropriate for this dataset size and single-node architecture.

### Orchestration (Prefect)

**Input:** None (orchestrates all stages)

**Flow structure:**
- **Task 1:** `ingest_data()` — pulls from Kaggle
- **Task 2:** `validate_data()` — applies Pandera schema (hard gate)
- **Task 3:** `transform_data()` — feature engineering and Parquet output
- **Task 4:** `load_to_warehouse()` — DuckDB load

**Dependencies:** Sequential (ingest → validate → transform → load)

**Retries:** Configured with 2 retries and exponential backoff for ingestion and storage tasks

**Validation gate:** If `validate_data()` fails, downstream tasks are skipped entirely

**Failure behavior:** Prefect marks the flow as failed, logs the error, and does not proceed to transformation or load.

---

## Dataset

Sourced from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Machine Learning Group - ULB).

| Attribute | Value |
|-----------|-------|
| **Total transactions** | 284,807 |
| **Features** | 31 (28 PCA-transformed + `Time` + `Amount`) |
| **Target** | `Class` (0 = legitimate, 1 = fraud) |
| **Fraud cases** | 492 (~0.17% of total) |
| **Time span** | 2 days (September 2013) |
| **Data type** | All numeric (no categorical features) |

### Feature Categories

- **V1–V28:** Anonymized principal components from PCA transformation (confidentiality-preserving)
- **Time:** Seconds elapsed since first transaction in dataset
- **Amount:** Transaction amount in dollars
- **Class:** Binary label (1 = fraud, 0 = legitimate)

### Data Engineering Considerations

The extreme class imbalance (~0.17% fraud) is a data quality concern the pipeline validates explicitly. If the fraud rate deviates significantly from expected bounds, validation fails — this catches upstream data corruption or schema drift.

---

## Data Quality and Failure Handling

The pipeline enforces data quality at multiple stages:

| Check | Stage | Enforcement |
|-------|-------|-------------|
| **Schema validation** | Validation | Pandera schema enforces column names, types, nullability |
| **Type validation** | Validation | All columns must be numeric; non-numeric values fail |
| **Range validation** | Validation | `Amount` ≥ 0, `Time` ≥ 0, `Class` ∈ {0, 1} |
| **Nullability** | Validation | No nulls allowed in any column |
| **Class distribution** | Validation | Fraud rate must be within expected bounds (~0.17%) |
| **Deduplication** | Transformation | Exact duplicate rows removed |
| **Load validation** | Storage | Row count verified after DuckDB load |
| **Orchestration failure** | Prefect | Failed tasks halt downstream execution; retries configured |

**Validation as a hard gate:** The pipeline is designed so bad data stops the run rather than silently flowing through to storage. This is a deliberate design choice — warnings are insufficient when downstream consumers (analysts, models) depend on data integrity.

---

## Project Structure

```text
.
├── src/
│   ├── ingestion/          # Kaggle API pull, raw file + manifest
│   ├── validation/         # Pandera schema definition and validation gate
│   ├── transformation/     # Cleaning, feature engineering, Parquet output
│   ├── storage/            # DuckDB load
│   └── pipeline/           # Prefect flow tying all stages together
├── tests/
│   ├── test_ingestion.py   # Ingestion logic (mocked API)
│   ├── test_validation.py  # Pandera schema tests
│   ├── test_transformation.py  # Feature engineering tests
│   ├── test_storage.py     # DuckDB load tests
│   └── test_pipeline.py    # End-to-end flow tests (synthetic data)
├── data/                   # Gitignored: raw/, processed/, warehouse/ (generated at runtime)
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template (Kaggle credentials)
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI workflow
```

---

## Running the Pipeline

### Prerequisites

- Python 3.10+
- Docker Desktop (optional, for containerized runs)
- Free [Kaggle](https://www.kaggle.com/) account with API token

### 1. Clone and Configure

```bash
git clone <repository-url>
cd credit-card-fraud-pipeline
```

Create a `.env` file with your Kaggle credentials:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

**Do not commit `.env` to Git** — it is gitignored by default.

### 2. Run locally

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python -m src.pipeline.flow
```

The flow runs: ingestion → validation → transformation → load. If validation fails, the pipeline halts before transformation or storage.

### 3. Run containerized

```bash
docker compose up --build
```

Docker Compose builds the image and runs the pipeline in a container with the same environment configuration.

### 4. Run tests

```bash
pytest tests/ -v
```

Unit tests use synthetic data and do not require Kaggle access.

---

## Testing

The test suite is designed for speed, isolation, and reproducibility.

| Test Module | Coverage |
|-------------|----------|
| `test_ingestion.py` | Ingestion logic with mocked Kaggle API |
| `test_validation.py` | Pandera schema validation (pass/fail cases) |
| `test_transformation.py` | Feature engineering, scaling, time transformations |
| `test_storage.py` | DuckDB load and row count verification |
| `test_pipeline.py` | End-to-end flow with synthetic data |

**Key properties:**
- **Synthetic data:** Tests do not depend on the full dataset or Kaggle access
- **Isolation:** Each stage is tested independently
- **Speed:** Full test suite runs in seconds
- **Expected failures:** Tests include cases where validation should fail (e.g., invalid schema, wrong class distribution)

---

## Orchestration (Prefect)

The Prefect flow orchestrates the pipeline rather than performing transformations itself.

### Flow Entry Point

`src/pipeline/flow.py` — defines the `fraud_detection_pipeline()` flow

### Tasks

| Task | Function | Purpose |
|------|----------|---------|
| Ingest | `ingest_data()` | Pulls dataset from Kaggle API |
| Validate | `validate_data()` | Applies Pandera schema (hard gate) |
| Transform | `transform_data()` | Feature engineering and Parquet output |
| Load | `load_to_warehouse()` | DuckDB storage |

### Task Dependencies

Sequential: `ingest_data` → `validate_data` → `transform_data` → `load_to_warehouse`

### Retries

- Ingestion and storage tasks: 2 retries with exponential backoff
- Validation and transformation: No retries (failures indicate data issues, not transient errors)

### Failure Behavior

- If `validate_data()` fails, downstream tasks are skipped
- Prefect marks the flow as failed and logs the error
- No partial data is written to storage

---

## Containerization (Docker)

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

CMD ["python", "-m", "src.pipeline.flow"]
```

### docker-compose.yml

```yaml
services:
  pipeline:
    build: .
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

### Why Containerization Matters

- **Reproducibility:** Same Python version, dependencies, and environment on any machine
- **Isolation:** No conflicts with host system packages
- **Portability:** Runs identically on local machines, CI, or cloud infrastructure
- **Simplified onboarding:** New engineers can run the pipeline with a single `docker compose up` command

---

## CI/CD (GitHub Actions)

### Workflow: `.github/workflows/ci.yml`

**Triggers:** Push to `main`, pull requests

**Python version:** 3.11

**Steps:**
1. Checkout repository
2. Install Python dependencies
3. Run `pytest tests/ -v`
4. Schema sanity check (validates Pandera schema loads without errors)

**What CI does NOT do:**
- Does not run the live Kaggle pipeline (no API credentials in CI)
- Does not download the dataset
- Does not execute the full end-to-end flow against live data

**Rationale:** CI validates code correctness, not data availability. Full pipeline runs against live data are a separate, deliberate concern requiring API credentials.

---

## Engineering Decisions

### DuckDB over a Hosted Database

**Decision:** Use DuckDB for storage rather than PostgreSQL, MySQL, or a cloud warehouse.

**Reasons:**
- **Local analytical workload:** No server to manage; file-based database
- **Native Parquet support:** Reads Parquet directly without ETL
- **Single-node architecture:** Appropriate for this dataset size (~285K rows)
- **Reproducibility:** Database file is portable and versionable (though gitignored here)
- **Production-used:** DuckDB is genuinely used in production analytical workloads, not just a toy database

**Trade-off:** Not suitable for multi-user concurrent writes or cloud-scale workloads.

### Prefect over Airflow

**Decision:** Use Prefect for orchestration rather than Airflow.

**Reasons:**
- **Native cross-platform support:** Runs on Windows, macOS, Linux without WSL or Docker
- **Simpler setup:** No database backend or web server required for basic flows
- **Retries and observability:** Provides task-level retries, logging, and failure handling
- **Proper DAG structure:** Explicit task dependencies and execution order

**Trade-off:** Less mature ecosystem than Airflow for enterprise-scale deployments.

### Validation as a Hard Gate

**Decision:** Validation failures halt the pipeline entirely rather than logging warnings and continuing.

**Reasons:**
- **Downstream integrity:** Bad data in storage corrupts analytical results and model training
- **Debuggability:** Easier to trace errors when they occur at the source
- **Explicit quality contract:** Consumers can trust that data in the warehouse passed validation

**Trade-off:** Pipeline is more brittle — transient upstream issues cause full failures rather than partial runs.

### CI without Live Kaggle Execution

**Decision:** GitHub Actions runs unit tests only, not the full pipeline against live data.

**Reasons:**
- **Credential security:** API keys should not be stored in CI environment
- **External dependency:** CI should not depend on Kaggle API availability
- **Speed:** Unit tests run in seconds; full pipeline takes minutes

**Trade-off:** CI does not validate end-to-end data flow — that requires manual or scheduled runs.

---

## Reproducibility

To reproduce a pipeline run:

1. **Python version:** 3.10+ (tested on 3.11)
2. **Dependencies:** Install from `requirements.txt`
3. **Environment variables:** Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`
4. **Local directories:** `data/raw/`, `data/processed/`, `data/warehouse/` are created at runtime (gitignored)
5. **Docker (optional):** `docker compose up --build` ensures identical environment

### Generated Artifacts

| Artifact | Location | Git-tracked |
|----------|----------|-------------|
| Raw CSV | `data/raw/` | No |
| Manifest JSON | `data/raw/` | No |
| Processed Parquet | `data/processed/` | No |
| DuckDB database | `data/warehouse/` | No |

### Source Inputs

- `src/` — pipeline code
- `tests/` — test suite
- `requirements.txt` — dependencies
- `Dockerfile`, `docker-compose.yml` — containerization
- `.github/workflows/ci.yml` — CI configuration

---

## Data Flow

```text
Kaggle API
    ↓
Raw CSV (data/raw/)
    ↓
Pandera validation
    ↓
Pandas transformation
    ↓
Parquet (data/processed/)
    ↓
DuckDB (data/warehouse/transactions.duckdb)
    ↓
Downstream analytics / fraud modelling
```

---

## Downstream Use Cases

The resulting warehouse enables:

- **Fraud analysis:** Query transaction patterns by fraud label
- **Feature exploration:** Analyze distributions of PCA components, amount, time
- **Transaction profiling:** Segment transactions by time-of-day, amount ranges
- **Analytical queries:** SQL-based exploration of the full dataset
- **Model training:** Export data for fraud classification models

The pipeline prepares the data; downstream consumers (analysts, data scientists, models) use the warehouse as a trusted source.

---

## Limitations and Future Improvements

### Current Limitations

- **External Kaggle dependency:** Pipeline requires Kaggle API access and credentials
- **No incremental loading:** Full reload on each run (acceptable for this dataset size)
- **Local DuckDB storage:** Not suitable for multi-user concurrent access or cloud-scale workloads
- **No production scheduler:** Prefect flow runs manually or via Docker Compose, not deployed to a production orchestration platform
- **No data observability platform:** No monitoring, alerting, or drift detection beyond validation gate
- **No historical data versioning:** Raw and processed files are overwritten on each run (no versioned snapshots)
- **Limited CI integration testing:** CI runs unit tests only, not end-to-end flow against live data
- **Credential management:** API keys stored in `.env` file (acceptable for local development, not production)

### Potential Improvements

- **Cloud warehouse:** Migrate DuckDB to Snowflake, BigQuery, or Redshift for multi-user access
- **Incremental loading:** Implement watermark-based incremental updates if source data grows
- **Scheduled orchestration:** Deploy Prefect flow to Prefect Cloud or Airflow for automated runs
- **Data versioning:** Use DVC or similar to version raw and processed datasets
- **Observability:** Integrate with data quality monitoring tools (e.g., Great Expectations, Monte Carlo)
- **Secret management:** Use environment-specific secret management (e.g., GitHub Secrets, AWS Secrets Manager)

---

## Technical Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.11 | Pipeline implementation |
| **Data ingestion** | `kaggle` API | Download dataset from Kaggle |
| **Validation** | `pandera` | Schema enforcement and data quality gate |
| **Transformation** | `pandas`, `pyarrow` | Feature engineering and Parquet output |
| **Storage** | `DuckDB` | Local analytical warehouse |
| **Orchestration** | `Prefect` | DAG-based workflow orchestration |
| **Containerization** | `Docker`, `docker compose` | Reproducible runtime environment |
| **Testing** | `pytest` | Unit and integration tests |
| **CI** | `GitHub Actions` | Automated code validation on push/PR |

---

## Acknowledgments

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by Machine Learning Group - ULB, hosted on Kaggle.

This project uses the dataset for educational and portfolio purposes. The pipeline implementation is original; the dataset remains the property of its creators.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.