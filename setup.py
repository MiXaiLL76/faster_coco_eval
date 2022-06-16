#!/usr/bin/python3

import setuptools
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.1.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    author="MiXaiLL76",
    description="Faster interpretation of the original COCOEval",
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    packages=setuptools.find_packages(),
    cmdclass={"build_ext": build_ext},
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['csrc']},
    install_requires=[
        'pybind11>=2.9.2',
        'numpy>=1.18.0',
        'testresources==2.0.1',
        'pycocotools>=2.0.0'
    ],
)
