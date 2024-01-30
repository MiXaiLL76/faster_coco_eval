#!/bin/bash -e
build_dir=$(mktemp -d)
cd ${build_dir}
cp -r /app/src/* ${build_dir}

make clean

make ${MAKE_CONFIG}

mkdir -p /app/src/dist

WHL="${build_dir}/dist/$(ls -lt ${build_dir}/dist/ | tail -1 | awk '{print $9}')"

echo "WHL: ${WHL}"

if [ "${WHL: -4}" == ".whl" ]; then
    auditwheel repair ${WHL} --plat "$PLAT" -w /app/src/dist
else
    cp dist/*.tar.gz /app/src/dist/
fi

python3 -m pip install /app/src/dist/*

python3 tests/basic.py

chown ${EID}:${EID} -R /app/src/dist