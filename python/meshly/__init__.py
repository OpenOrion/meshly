"""
High-level export functionality for meshoptimizer.

This package provides high-level abstractions and utilities for working with
meshoptimizer, including:

1. Mesh class as a Pydantic base class for representing 3D meshes
2. MeshUtils class for mesh optimization and encoding/decoding operations
3. ArrayUtils class for array encoding/decoding operations
4. EncodedMesh class for storing encoded mesh data
5. I/O utilities for storing and loading meshes and arrays
6. Support for custom Mesh subclasses with automatic encoding/decoding of numpy arrays
"""

from .mesh import (
    Mesh,
    EncodedMesh,
    MeshSize,
    MeshMetadata,
    MeshUtils,
)

from .array import (
    EncodedArray,
    ArrayMetadata,
    EncodedArrayModel,
    ArrayResult,
    ArrayUtils,
)


__all__ = [
    # Mesh classes
    "Mesh",
    "EncodedMesh",
    "MeshSize",
    "MeshMetadata",
    "MeshUtils",
    # Array utilities
    "EncodedArray",
    "EncodedArrayModel",
    "ArrayMetadata",
    "ArrayResult",
    "ArrayUtils",
]
