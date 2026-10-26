FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_LOCAL_LLM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cpu
COPY src ./src
COPY data/sources ./data/sources
RUN python -m pip install --no-cache-dir .

RUN useradd --system --create-home --uid 10001 agent \
    && mkdir -p data/state data/reports data/quarantine data/policies \
    && chown -R agent:agent /app
USER agent

CMD ["legal-agent", "run-workflow"]
