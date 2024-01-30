all: format docker-sdist docker-3.6 docker-3.7 docker-3.8 docker-3.9 docker-3.10  docker-3.11
	ls -lah dist

sdist:
	python3 -m build . --sdist

bdist: clean
	python3 setup.py bdist_wheel
	pip3 install dist/*.whl -U

wheel:
	python3 -m build . --wheel

docker-sdist:
	bash -e docker/auto_build.sh "cp38-cp38" sdist

docker-3.6:
	bash -e docker/auto_build.sh "cp36-cp36m" wheel

docker-3.7:
	bash -e docker/auto_build.sh "cp37-cp37m" wheel

docker-3.8:
	bash -e docker/auto_build.sh "cp38-cp38" wheel
	
docker-3.9:
	bash -e docker/auto_build.sh "cp39-cp39" wheel
	
docker-3.10:
	bash -e docker/auto_build.sh "cp310-cp310" wheel

docker-3.11:
	bash -e docker/auto_build.sh "cp311-cp311" wheel

pull:
	twine check dist/*
	twine upload --repository testpypi dist/*

pull-prod:
	twine check dist/*
	twine upload dist/*

format:
	python3 -m black --config pyproject.toml .
	python3 -m isort .
	flake8

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y