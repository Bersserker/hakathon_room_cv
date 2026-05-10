## Serving API

REST API на FastAPI для инференса модели классификации помещений.

Что доступно:
- web UI для загрузки изображения: `http://localhost:8000/`;
- проверка состояния сервиса;
- информация о загруженной модели;
- предсказание для одного изображения по локальному пути;
- предсказание для загруженного файла;
- batch-предсказание для нескольких локальных путей.

Основные endpoint'ы:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI с загрузкой изображения |
| GET | `/health` | Проверка доступности сервиса |
| GET | `/model/info` | Информация о модели |
| POST | `/predict` | Предсказание для одного изображения по `image_path` |
| POST | `/predict_upload` | Предсказание для загруженного изображения (`multipart/form-data`) |
| POST | `/predict_batch` | Batch inference по списку `image_path` |

Запуск из корня репозитория после обучения release checkpoint:

```bash
docker compose -f serving/docker-compose.yml up --build
```

`configs/release/rc1_single.yaml` генерируется командой `make train-release`.
По умолчанию сервер читает этот файл, где checkpoint:

```text
artifacts/checkpoints/release_cv03_balanced_sampler_trainval_90_10.ckpt
```

Локальный запуск без Docker:

```bash
CONFIG_PATH=configs/release/rc1_single.yaml \
uv run uvicorn serving.app.main:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Отправка файла:

```bash
curl -X POST http://localhost:8000/predict_upload \
  -F "file=@/path/to/image.jpg"
```

Swagger:

```text
http://localhost:8000/docs
```
