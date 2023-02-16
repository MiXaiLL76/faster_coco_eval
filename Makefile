all:
	pip wheel --no-deps -w dist .

install:
	pip3 install dist/faster_coco_eval* -U

test: clean all install clean
	cd tests && python3 basic.py 

docker: clean docker-3.6# docker-3.7 docker-3.8 docker-3.9 docker-3.10
	ls -lah dist

docker-3.6:
	bash auto_build.sh "cp36-cp36m"
	
docker-3.7:
	bash auto_build.sh "cp37-cp37m"

docker-3.8:
	bash auto_build.sh "cp38-cp38"
	
docker-3.9:
	bash auto_build.sh "cp39-cp39"
	
docker-3.10:
	bash auto_build.sh "cp310-cp310"

pull:
	twine check dist/*
	twine upload --repository testpypi dist/*

clean:
	rm -rf build
	rm -rf *.egg-info
	rm -rf dist
	pip3 uninstall faster-coco-eval -y