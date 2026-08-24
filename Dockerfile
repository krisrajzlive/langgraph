FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV CHECKPOINT_DB_PATH=/app/data/checkpoints.sqlite

ENTRYPOINT ["uv", "run", "main.py"]
CMD ["all"]
