FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml readme.md ./

RUN uv sync --index https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --no-sources-package torch \
    --no-sources-package torchvision \
    --no-sources-package torchaudio
RUN uv pip install --python /opt/venv/bin/python torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8888

CMD ["uv", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=room-cv"]
