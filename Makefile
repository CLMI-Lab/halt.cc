.PHONY: help install test run clean lint

help:
	@echo "install  install the package and test deps"
	@echo "test     run the test suite (the propositions, numerically)"
	@echo "run      run the full reference pipeline into runs/reference/"
	@echo "clean    remove generated runs"

install:
	pip install -e ".[test]"

test:
	python3 -m pytest tests/ -q

run:
	bash scripts/run_all.sh reference

clean:
	rm -rf runs/ .pytest_cache **/__pycache__
