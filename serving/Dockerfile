FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN pip install uv && uv sync --no-dev

COPY serving/app ./serving/app
COPY src ./src
COPY configs ./configs
COPY models ./models
COPY artifacts/checkpoints ./artifacts/checkpoints

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]