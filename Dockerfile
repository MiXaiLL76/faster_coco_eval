FROM quay.io/pypa/manylinux_2_28_x86_64
LABEL maintainer="MiXaiLL76 <mike.milos@yandex.ru>"

USER root

ARG PYTHON3_VERSION="cp36-cp36m"
ENV PYTHON3_VERSION=${PYTHON3_VERSION}

ARG PLAT="manylinux1_x86_64"
ENV PLAT=${PLAT}

ARG ID=1000
ENV EID=${ID}

ENV PATH="${PATH}:/opt/python/${PYTHON3_VERSION}/bin"

### basic_config
WORKDIR /tmp

COPY ./requirements.txt ./
RUN python3 -m pip install --default-timeout=10 --no-cache-dir -r requirements.txt

# BUILD ONLY
RUN python3 -m pip install --default-timeout=10 --no-cache-dir setuptools>=42 wheel pybind11~=2.6.1 cython testresources
WORKDIR /app/src

# ENTRYPOINT bash
ENTRYPOINT make && \
    rm -rf build && \
    rm -rf *.egg-info && \
    chown ${EID}:${EID} -R dist && \
    auditwheel repair "./dist/*${PYTHON3_VERSION}*.whl" --plat "$PLAT" -w ./wheelhouse
