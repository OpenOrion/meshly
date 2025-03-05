from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pymeshoptimizer",
    version="0.1.1",
    description="High-level abstractions and utilities for working with meshoptimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/pymeshoptimizer",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "meshoptimizer==0.2.20a4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
)