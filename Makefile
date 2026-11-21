.PHONY: data features train test check monitor serve dashboard mlflow-ui docker-build docker-run

data:
	python -m src.data.clean

features:
	python -m src.features.build_features

train:
	python -m src.models.train

test:
	pytest -v

check:
	ruff check . && pytest

monitor:
	python -m src.monitoring.monitor

serve:
	uvicorn src.serving.app:app --reload --port 8000

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t fraud-api .

docker-run:
	docker run -d --name fraud-api -p 8000:8000 fraud-api