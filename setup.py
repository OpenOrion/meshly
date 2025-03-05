from setuptools import setup, find_packages

setup(
    name="pymeshoptimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "meshoptimizer==0.2.20a4",
    ],
)