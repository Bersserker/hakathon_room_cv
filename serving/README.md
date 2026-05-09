## Serving API

Для использования модели в production предусмотрен REST API на FastAPI.

API позволяет:
- проверить состояние сервиса;
- получить информацию о загруженной модели;
- выполнить предсказание для одного изображения;
- выполнить batch-предсказание для нескольких изображений.

Основные endpoint'ы:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Проверка доступности сервиса |
| GET | `/model/info` | Информация о модели |
| POST | `/predict` | Предсказание для одного изображения |
| POST | `/predict_batch` | Batch inference |

Сервис контейнеризуется через Docker и может быть запущен через docker-compose.

Команды запуска

docker compose -f serving/docker-compose.yml up --build

Проверка:

curl http://localhost:8000/health

Swagger:

http://localhost:8000/docs
