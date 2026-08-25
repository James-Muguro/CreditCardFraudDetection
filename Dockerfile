FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .

# data/ is mounted as a volume at runtime, not baked into the image
RUN mkdir -p data/raw data/processed data/warehouse

CMD ["python", "-m", "src.pipeline.flow"]