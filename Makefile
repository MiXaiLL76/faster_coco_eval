all: sdist wheel
	ls -lah dist

sdist:
	python3 -m build . --sdist

wheel:
	python3 -m build . --wheel

docker: docker-sdist docker-3.7 docker-3.8 docker-3.9 docker-3.10
	ls -lah dist

docker-sdist:
	bash docker/auto_build.sh "cp38-cp38" sdist

docker-3.7:
	bash docker/auto_build.sh "cp37-cp37m" wheel

docker-3.8:
	bash docker/auto_build.sh "cp38-cp38" wheel
	
docker-3.9:
	bash docker/auto_build.sh "cp39-cp39" wheel
	
docker-3.10:
	bash docker/auto_build.sh "cp310-cp310" wheel

pull:
	twine check dist/*
	twine upload --repository testpypi dist/*

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y