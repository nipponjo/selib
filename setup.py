#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py
from pathlib import Path

from setuptools import setup, find_packages


NAME = "speech-selib"
DESCRIPTION = "A simple speech enhancement library using ONNX models"
URL = "https://github.com/nipponjo/selib"
EMAIL = "nipponjo.git@gmail.com"
AUTHOR = "nipponjo"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.2.0"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,

    packages=find_packages(
        include=["selib", "selib.*"],
        exclude=["tmp", "tmp.*", "data", "data.*", "dist", "build", "*.egg-info"],
    ),
    install_requires=[
        "numpy",
        "librosa",
        "onnxruntime-gpu; sys_platform != 'darwin'",  # for Windows, Linux
        "onnxruntime; sys_platform == 'darwin'",  # for Mac
    ],

    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Telecommunications Industry",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ])
