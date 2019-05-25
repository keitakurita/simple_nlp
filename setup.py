#!/usr/bin/env python
import os
from setuptools import setup, find_packages

cd = os.path.dirname(__file__)

setup(
    name="simple_nlp",
    version=0.1,
    author="Keita Kurita",
    author_email="keita.kurita@gmail.com",
    description="A simple library for NLP preprocessing in PyTorch",
    license="MIT",
    url="https://github.com/keitakurita/simple_nlp",
    python_requires = ">=3.6",
    keywords = "PyTorch, deep learning, NLP",
    setup_requires=["pytest", ],
    install_requires=[
        "torch>=1.0.0",
    ],
    packages=find_packages(),
)
