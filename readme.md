# Проект классификации типов помещений

Набор ноутбуков и экспериментов для классификации изображений помещений с использованием PyTorch и предобученных моделей.

## Зависимости

Зависимости описаны в `pyproject.toml` и совместимы с `uv` и `poetry`.

Базовая установка:

```bash
uv sync
```

CUDA-стек PyTorch 12.6 для Linux/Windows:

```bash
uv sync --extra cuda
```

Альтернатива через Poetry:

```bash
poetry install
```

CUDA-стек через Poetry:

```bash
poetry install -E cuda
```

## Запуск ноутбуков

Через `uv`:

```bash
uv run jupyter notebook
```

Через `poetry`:

```bash
poetry run jupyter notebook
```

## Docker

Минимальный Docker-сценарий в проекте рассчитан на локальную разработку и запуск Jupyter Notebook в CPU-контейнере.

### Установка Docker

1. Установите Docker Desktop для macOS/Windows или Docker Engine + Docker Compose Plugin для Linux.
2. Убедитесь, что команды доступны:

```bash
docker --version
docker compose version
```

### Сборка и запуск

Собрать и запустить контейнер:

```bash
docker compose up --build
```

После запуска Jupyter будет доступен по адресу:

```text
http://localhost:8888/tree?token=room-cv
```

Запуск в фоне:

```bash
docker compose up -d --build
```

Остановить проект:

```bash
docker compose down
```

Открыть shell внутри контейнера:

```bash
docker compose exec notebook sh
```

Чтобы выйти из shell внутри контейнера и вернуться в терминал хоста:

  - введите exit
  - или нажмите Ctrl+D

Примечания:

- В контейнер монтируется текущая директория проекта, поэтому изменения в ноутбуках и коде сразу видны внутри Docker.
- Dockerfile ставит CPU-версии `torch`, `torchvision` и `torchaudio`, чтобы проект запускался без CUDA-инфраструктуры.
- Для GPU-режима нужен отдельный Docker-стек с `nvidia-container-toolkit`.

### Docker + GPU (CUDA)

GPU-вариант запуска использует отдельные файлы `Dockerfile.gpu` и `docker-compose.gpu.yml`.

Требования к хосту:

1. Linux с NVIDIA GPU и установленным NVIDIA Driver + NVIDIA Container Toolkit.
2. Или Docker Desktop на Windows с backend WSL2 и включённой GPU support.
3. Для macOS этот NVIDIA-вариант не подходит.

Минимальная подготовка Linux-хоста:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Быстрая проверка, что Docker видит GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

Сборка и запуск проекта с GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Запуск в фоне:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Проверка CUDA внутри контейнера проекта:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm notebook \
  uv run python -c "import torch; print(torch.cuda.is_available())"
```

Примечания:

- В GPU-образе ставятся зависимости из `uv sync --extra cuda`, то есть используются CUDA-колёса PyTorch из `pyproject.toml`.
- В `docker-compose.gpu.yml` доступ к GPU описан через `devices` reservation, как рекомендует Docker Compose.
- Если нужен конкретный GPU, вместо `count: all` можно перейти на `device_ids`.

## Структура проекта

```text
.
├── artifacts/
│   ├── checkpoints/
│   └── logs/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
└── tests/
```

Коротко по папкам:

- `src/` — основной код проекта: загрузка данных, модели, обучение, инференс и утилиты.
- `src/datasets/` — отдельные части пайплайна обработки данных: reusable-модули для split, аудита и другой data-logic.
- `configs/` — конфигурации экспериментов и параметров запуска.
- `data/` — данные на разных стадиях подготовки: сырые, промежуточные и обработанные.
- `notebooks/` — исследовательские ноутбуки и быстрые эксперименты.
- `scripts/` — запуск отдельных модулей пайплайна; здесь лежат тонкие CLI-обёртки над кодом из `src/`.
- `artifacts/` — веса моделей, чекпоинты и логи обучения.
- `reports/` — графики, отчёты и итоговые визуализации.
- `tests/` — тесты для проверки ключевой логики проекта.
