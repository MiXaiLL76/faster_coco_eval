#!/usr/bin/python3

import glob
import platform

import numpy as np
import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


version_file = "faster_coco_eval/version.py"


def get_version():
    with open(version_file) as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"], locals()["__author__"]


def parse_requirements(fname="requirements/runtime.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"

    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    yield from parse_line(line)

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_extensions(version_info):
    ext_modules = []

    sources = [
        "csrc/faster_eval_api/coco_eval/cocoeval.cpp",
        "csrc/faster_eval_api/faster_eval_api.cpp",
    ]
    print("Sources: {}".format(sources))

    ext_modules += [
        Pybind11Extension(
            name="faster_coco_eval.faster_eval_api_cpp",
            sources=sources,
            define_macros=[("VERSION_INFO", version_info)],
        )
    ]

    sources = [
        "csrc/mask/common/maskApi.c",
        "csrc/mask/pycocotools/_mask.pyx",
    ]
    include_dirs = [np.get_include(), "csrc/mask/common"]

    print("Sources: {}".format(sources))
    print("Include: {}".format(include_dirs))

    ext_modules += [
        Extension(
            "faster_coco_eval.mask_api_cpp",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=(
                []
                if platform.system() == "Windows"
                else [
                    "-Wno-cpp",
                    "-Wno-unused-function",
                    "-std=c99",
                    "-O3",
                    "-Wno-maybe-uninitialized",
                    "-Wno-misleading-indentation",
                ]
            ),
            extra_link_args=[],
        )
    ]

    return ext_modules


__version__, __author__ = get_version()

setup(
    name="faster-coco-eval",
    version=__version__,
    description="Faster interpretation of the original COCOEval",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MiXaiLL76/faster_coco_eval",
    author=__author__,
    author_email="mike.milos@yandex.ru",
    packages=setuptools.find_packages(),
    ext_modules=get_extensions(__version__),
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    data_files=glob.glob("requirements/*"),
    install_requires=parse_requirements("requirements/runtime.txt"),
)
