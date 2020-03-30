init:
	pip install -r requirements.txt

# Run tests
test:
	pytest -q tests

# Run Verbose tests
vtest:
	pytest tests

.PHONY: init test