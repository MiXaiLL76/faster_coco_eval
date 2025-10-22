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

sphinx:
	cp examples/*.ipynb docs/source/examples/
	cp history.md docs/source/history.md
	cp README.md docs/source/README.md

	$(MAKE) -C docs clean html
	rm -rf docs/source/examples/*.ipynb
	rm -rf docs/source/history.md
	rm -rf docs/source/README.md

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	rm -rf faster_coco_eval/*.so
	pip3 uninstall faster-coco-eval -y
