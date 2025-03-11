# meshly

A Python library for mesh optimization and encoding/decoding.

## Project Structure

This repository contains two main components:

1. **Python Library**: The main `meshly` package for mesh optimization and encoding/decoding.
2. **TypeScript Library**: A TypeScript implementation for decoding Python meshoptimizer zip files into THREE.js geometries.

### Python Library

The Python library is located in the `meshly` directory and provides:

- Mesh class as a Pydantic base class for representing and optimizing 3D meshes
- EncodedMesh class for storing encoded mesh data
- Functions for encoding and decoding meshes
- Utilities for compressing numpy arrays

### TypeScript Library

The TypeScript library is located in the `typescript` directory and provides:

- Functions to decode Python meshoptimizer zip files
- Conversion to THREE.js BufferGeometry
- Browser-compatible implementation

## Usage

See the examples directory for usage examples of both libraries.