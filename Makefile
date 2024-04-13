all: clean format sdist wheel
	ls -lah dist

wheel:
	pipx run build --wheel .

sdist:
	pipx run build --sdist .

install: clean wheel
	pip3 install dist/*.whl

FORMAT_DIRS = ./faster_coco_eval ./tests ./examples setup.py
LINE_LENGTH = 80
BLACK_CONFIG = --preview --enable-unstable-feature string_processing

format:
	python3 -m docformatter --in-place --recursive --blank $(FORMAT_DIRS)
	python3 -m black $(BLACK_CONFIG) --line-length $(LINE_LENGTH) $(FORMAT_DIRS)
	python3 -m isort --line-length $(LINE_LENGTH) --profile black $(FORMAT_DIRS)
	$(MAKE) linter

linter:
	python3 -m docformatter --check --recursive --blank $(FORMAT_DIRS)
	python3 -m black $(BLACK_CONFIG) --line-length $(LINE_LENGTH) $(FORMAT_DIRS) --check --diff
	python3 -m isort --line-length $(LINE_LENGTH) --profile black $(FORMAT_DIRS) --check --diff
	flake8 --max-line-length $(LINE_LENGTH) $(FORMAT_DIRS) --ignore=E203,W503 --exclude "*/model/*"

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y