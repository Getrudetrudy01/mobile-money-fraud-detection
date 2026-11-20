.PHONY: data test

data:
	python -m src.data.clean

test:
	pytest -v