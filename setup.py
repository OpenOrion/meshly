from setuptools import setup, find_packages
import os

# Read the contents of the README file if it exists
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name="pymeshoptimizer",
    version="0.2.1",  # Updated version for new functionality
    description="High-level abstractions and utilities for working with meshoptimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/pymeshoptimizer",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,  # Include files specified in MANIFEST.in
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