.PHONY: smoke-train train mlflow-ui

smoke-train:
	python src/training/train_image.py --config configs/model/image_baseline_v1.yaml --fold 0 --debug

train:
	python src/training/train_image.py --config configs/model/image_baseline_v1.yaml --fold 0

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///artifacts/logs/mlflow.db