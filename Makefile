all: clean format sdist wheel
	ls -lah dist

wheel:
	pipx run build --wheel .

sdist:
	pipx run build --sdist .

whl_file = $(shell ls dist/*.whl)

install: clean wheel
	pip3 install "$(whl_file)[tests]"

FORMAT_DIRS = ./faster_coco_eval ./tests setup.py
LINE_LENGTH = 80
BLACK_CONFIG = --preview --enable-unstable-feature string_processing

format:
	pre-commit run --all-files

linter:
	pre-commit check --all-files

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y
