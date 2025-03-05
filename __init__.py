"""
High-level export functionality for meshoptimizer.

This package provides high-level abstractions and utilities for working with
meshoptimizer, including:

1. Mesh class for representing and optimizing 3D meshes
2. EncodedMesh class for storing encoded mesh data
3. I/O utilities for storing and loading meshes and arrays
"""

from .mesh import (
    Mesh,
    EncodedMesh,
    encode_mesh,
    decode_mesh,
)

from .arrayutils import (
    EncodedArray,
    encode_array,
    decode_array,
)

from .io import (
    # Array file I/O
    save_encoded_array_to_file,
    load_encoded_array_from_file,
    save_array_to_file,
    load_array_from_file,
    
    # Mesh ZIP I/O
    save_encoded_mesh_to_zip,
    load_encoded_mesh_from_zip,
    save_mesh_to_zip,
    load_mesh_from_zip,
    
    # Array ZIP I/O
    save_encoded_array_to_zip,
    load_encoded_array_from_zip,
    save_array_to_zip,
    load_array_from_zip,
    save_arrays_to_zip,
    load_arrays_from_zip,
    
    # Combined data I/O
    save_combined_data_to_zip,
    load_combined_data_from_zip,
)

__all__ = [
    # Mesh classes
    'Mesh',
    'EncodedMesh',
    'encode_mesh',
    'decode_mesh',
    
    # Array utilities
    'EncodedArray',
    'encode_array',
    'decode_array',
    
    # Array file I/O
    'save_encoded_array_to_file',
    'load_encoded_array_from_file',
    'save_array_to_file',
    'load_array_from_file',
    
    # Mesh ZIP I/O
    'save_encoded_mesh_to_zip',
    'load_encoded_mesh_from_zip',
    'save_mesh_to_zip',
    'load_mesh_from_zip',
    
    # Array ZIP I/O
    'save_encoded_array_to_zip',
    'load_encoded_array_from_zip',
    'save_array_to_zip',
    'load_array_from_zip',
    'save_arrays_to_zip',
    'load_arrays_from_zip',
    
    # Combined data I/O
    'save_combined_data_to_zip',
    'load_combined_data_from_zip',
]