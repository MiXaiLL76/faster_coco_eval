FROM quay.io/pypa/manylinux_2_28_x86_64
LABEL maintainer="MiXaiLL76 <mike.milos@yandex.ru>"

USER root

ARG PYTHON3_VERSION="cp36-cp36m"
ENV PYTHON3_VERSION=${PYTHON3_VERSION}

ARG MAKE_CONFIG="wheel"
ENV MAKE_CONFIG=${MAKE_CONFIG}

ARG PLAT="manylinux_2_28_x86_64"
ENV PLAT=${PLAT}

ARG ID=1000
ENV EID=${ID}

ENV PATH="${PATH}:/opt/python/${PYTHON3_VERSION}/bin"

### basic_config
WORKDIR /tmp
COPY docker/build_pipeline.sh .

ENTRYPOINT bash build_pipeline.sh
