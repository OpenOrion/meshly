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

from .array import (
    Array,
    ArrayMetadata,
    ArrayRefMetadata,
    ArrayType,
    ArrayUtils,
    EncodedArray,
    EncodingType,
    IndexSequence,
    PackableRefMetadata,
    VertexBuffer,

)
from .cell_types import (
    CellType,
    CellTypeUtils,
    VTKCellType,
)
from .data_handler import (
    AssetProvider,
    CachedAssetLoader,
    DataHandler,
)
from .mesh import (
    Mesh,
)
from .packable import ExtractedAssets, LazyModel, Packable, SerializedPackableData
from .resource import (
    Resource,
    ResourceRef,
    ResourceRefMetadata
)
from .constants import ExportConstants
from .utils import (
    ElementUtils,
    MeshUtils,
)

__all__ = [
    "PackableRefMetadata",
    "ResourceRefMetadata",

    # Packable base class
    "Packable",
    "SerializedPackableData",
    "ExtractedAssets",
    "LazyModel",
    "ArrayType",
    # Data handlers
    "AssetProvider",
    "CachedAssetLoader",
    "DataHandler",
    # Mesh classes
    "Mesh",
    # Array types and utilities
    "Array",
    "VertexBuffer",
    "IndexSequence",
    "ArrayRefMetadata",
    "EncodedArray",
    "EncodingType",
    "ArrayMetadata",
    "ArrayUtils",

    # Cell type utilities
    "CellType",
    "VTKCellType",
    "CellTypeUtils",
    # Resource handling
    "Resource",
    "ResourceRef",
    # File format constants
    "ExportConstants",
    # Element utilities
    "ElementUtils",
    # Mesh operations
    "MeshUtils",
]
