#!/bin/bash -e

PYTHON3_VERSION=$1

docker build -f ./Dockerfile \
    --build-arg ID=${UID} \
    --build-arg PYTHON3_VERSION=${PYTHON3_VERSION} \
    --tag faster_coco_eval:${PYTHON3_VERSION} .

docker run -v $(pwd):/app/src faster_coco_eval:${PYTHON3_VERSION}