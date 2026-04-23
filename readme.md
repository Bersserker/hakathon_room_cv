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
- `configs/` — конфигурации экспериментов и параметров запуска.
- `data/` — данные на разных стадиях подготовки: сырые, промежуточные и обработанные.
- `notebooks/` — исследовательские ноутбуки и быстрые эксперименты.
- `scripts/` — вспомогательные скрипты для запуска пайплайнов и подготовки данных.
- `artifacts/` — веса моделей, чекпоинты и логи обучения.
- `reports/` — графики, отчёты и итоговые визуализации.
- `tests/` — тесты для проверки ключевой логики проекта.
