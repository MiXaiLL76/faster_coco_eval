all: clean format sdist wheel
	ls -lah dist

sdist:
	python3 -m build . --sdist

wheel:
	python3 -m build . --wheel

format:
	python3 -m black --config pyproject.toml .
	python3 -m isort --profile black .
	flake8

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y