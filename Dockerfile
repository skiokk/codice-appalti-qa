FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only torch first, so sentence-transformers does not pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY sql ./sql
# Dataset and published results (the /valutazione page reads eval/results/latest.json).
COPY eval ./eval
RUN pip install .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface /app/data \
    && chown -R app:app /home/app /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "appalti_qa.web:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
