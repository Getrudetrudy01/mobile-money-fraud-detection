.PHONY: data features test

data:
	python -m src.data.clean

features:
	python -m src.features.build_features

test:
	pytest -v
