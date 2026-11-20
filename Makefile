.PHONY: data features train mlflow-ui test

data:
	python -m src.data.clean

features:
	python -m src.features.build_features

train:
	python -m src.models.train

test:
	pytest -v

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000