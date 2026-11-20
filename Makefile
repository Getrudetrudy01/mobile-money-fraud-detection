.PHONY: data features train test

data:
	python -m src.data.clean

features:
	python -m src.features.build_features

train:
	python -m src.models.train

test:
	pytest -v
