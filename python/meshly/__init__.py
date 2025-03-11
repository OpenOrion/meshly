"""
High-level export functionality for meshoptimizer.

This package provides high-level abstractions and utilities for working with
meshoptimizer, including:

1. Mesh class as a Pydantic base class for representing and optimizing 3D meshes
2. EncodedMesh class for storing encoded mesh data
3. I/O utilities for storing and loading meshes and arrays
4. Support for custom Mesh subclasses with automatic encoding/decoding of numpy arrays
"""

from .mesh import (
    Mesh,
    EncodedMesh,
)

from .arrayutils import (
    EncodedArray,
    encode_array,
    decode_array,
)

from .models import (
    EncodedMeshModel,
    EncodedArrayModel,
    EncodedMeshData,
    ArrayMetadata,
    MeshMetadata,
    ModelData,
    MeshFileMetadata,
)


__all__ = [
    # Mesh classes
    'Mesh',
    'EncodedMesh',

    # Array utilities
    'EncodedArray',
    'encode_array',
    'decode_array',
    
    # Pydantic models
    'EncodedMeshModel',
    'EncodedArrayModel',
    'EncodedMeshData',
    'ArrayMetadata',
    'MeshMetadata',
    'ModelData',
    'MeshFileMetadata',
]
