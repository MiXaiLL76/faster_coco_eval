#!/bin/bash -e

PYTHON3_VERSION=$1
MAKE_CONFIG=$2


docker build -f ./docker/Dockerfile \
    --build-arg ID=${UID} \
    --build-arg MAKE_CONFIG=${MAKE_CONFIG} \
    --build-arg PYTHON3_VERSION=${PYTHON3_VERSION} \
    --tag faster_coco_eval:${PYTHON3_VERSION}_${MAKE_CONFIG} .

docker_name="faster_coco_eval_${PYTHON3_VERSION//-/_}_${MAKE_CONFIG}"
docker run --name ${docker_name} -v $(pwd):/app/src faster_coco_eval:${PYTHON3_VERSION}_${MAKE_CONFIG}
docker rm ${docker_name} 
