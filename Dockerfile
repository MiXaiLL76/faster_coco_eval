ARG PYTHON3_VERSION=3.9.10

FROM python:${PYTHON3_VERSION}-slim-buster

LABEL maintainer="MiXaiLL76 <mike.milos@yandex.ru>"

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install build-essential -y

### basic_config
WORKDIR /tmp

COPY ./requirements.txt ./
RUN python3 -m pip install --default-timeout=10 --no-cache-dir -r requirements.txt

# BUILD ONLY
RUN python3 -m pip install --default-timeout=10 --no-cache-dir setuptools>=42 wheel pybind11~=2.6.1 testresources
WORKDIR /app/src

ENTRYPOINT python3 setup.py bdist_wheel && rm -rf build *.egg-info