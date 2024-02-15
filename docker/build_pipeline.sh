#!/bin/bash -e
build_dir=$(mktemp -d)
cd ${build_dir}
cp -r /app/src/* ${build_dir}

make clean

make ${MAKE_CONFIG}

mkdir -p /app/src/dist

WHL="${build_dir}/dist/$(ls -lt ${build_dir}/dist/ | tail -1 | awk '{print $9}')"

echo "WHL: ${WHL}"
python3 -m pip install ${WHL}
python3 tests/basic.py

if [ "${WHL: -4}" == ".whl" ]; then
    auditwheel repair ${WHL} --plat "$PLAT" -w /app/src/dist
else
    cp dist/*.tar.gz /app/src/dist/
fi

chown ${EID}:${EID} -R /app/src/dist