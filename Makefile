init:
	pip install -r requirements.txt

test:
	pytest -q tests

.PHONY: init test