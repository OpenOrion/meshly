"""
High-level export functionality for meshoptimizer.

This package provides high-level abstractions and utilities for working with
meshoptimizer, including:

1. Packable base class for automatic array serialization
2. Mesh class as a Pydantic base class for representing 3D meshes
3. MeshUtils static class for mesh optimization operations
4. ArrayUtils class for array encoding/decoding operations
5. I/O utilities for storing and loading meshes and arrays
6. Support for custom subclasses with automatic encoding/decoding of numpy arrays
7. CellTypeUtils for VTK cell type conversions and edge topology extraction
"""

from meshly.array import (
    Array,
    ArrayRefInfo,
    ArrayType,
    ArrayUtils,
    ExtractedArray,
    ArrayEncoding,
    IndexSequence,
    VertexBuffer,
)
from meshly.cell_types import (
    CellType,
    CellTypeUtils,
    VTKCellType,
)
from meshly.common import (
    AssetProvider,
)
from meshly.constants import ExportConstants
from meshly.mesh import Mesh
from meshly.packable import ExtractedPackable, Packable, PackableMetadata, PackableRefInfo
from meshly.utils.dynamic_model import LazyModel
from meshly.resource import Resource
from meshly.asset_store import AssetStore
from meshly.utils import ElementUtils, MeshUtils

__all__ = [
    "PackableRefInfo",
    "PackableMetadata",

    # Packable base class
    "Packable",
    "ExtractedPackable",
    "LazyModel",
    "ArrayType",
    # Asset store
    "AssetStore",
    # Asset providers
    "AssetProvider",
    # Mesh classes
    "Mesh",
    # Array types and utilities
    "Array",
    "VertexBuffer",
    "IndexSequence",
    "ExtractedArray",
    "ArrayEncoding",
    "ArrayRefInfo",
    "ArrayUtils",

    # Cell type utilities
    "CellType",
    "VTKCellType",
    "CellTypeUtils",
    # Resource handling
    "Resource",
    # File format constants
    "ExportConstants",
    # Element utilities
    "ElementUtils",
    # Mesh operations
    "MeshUtils",
]
