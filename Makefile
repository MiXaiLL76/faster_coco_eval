all: clean format sdist wheel
	ls -lah dist

wheel:
	pipx run build --wheel .

sdist:
	pipx run build --sdist .

whl_file = $(shell ls dist/*.whl)

install: clean wheel
	pip3 install "$(whl_file)[tests]" --user

format:
	pre-commit run --all-files

linter:
	ruff check --force-exclude

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y
