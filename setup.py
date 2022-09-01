#!/usr/bin/python3

import setuptools
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


info_file = {}
with open("faster_coco_eval/info.py") as fp:
    exec(fp.read(), info_file)

__version__ = info_file['__version__']
__author__ = info_file['__author__']

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = []
with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split('\n')


def get_extensions():
    sources = [
        "csrc/coco_eval/cocoeval.cpp",
        "csrc/faster_eval_api.cpp",
    ]
    print(f"Sources: {sources}")

    ext_modules = [
        Pybind11Extension(
            name="faster_coco_eval.faster_eval_api_cpp",
            sources=sources,
        )
    ]
    return ext_modules


setup(
    name="faster-coco-eval",
    version=__version__,
    author=__author__,
    description="Faster interpretation of the original COCOEval",
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    packages=setuptools.find_packages(),
    cmdclass={"build_ext": build_ext},
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['csrc']},
    install_requires=install_requires,
)
