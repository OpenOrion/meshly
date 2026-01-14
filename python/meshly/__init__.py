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

from .packable import (
    Packable,
    PackableMetadata,
)

from .mesh import (
    Mesh,
)

from .array import (
    EncodedArray,
    ArrayMetadata,
    ArrayUtils,
    ArrayType,
    Array,
)

from .cell_types import (
    CellType,
    VTKCellType,
    CellTypeUtils,
)

from .utils import (
    ElementUtils,
    MeshUtils,
)

from .data_handler import (
    DataHandler,
)


__all__ = [
    # Packable base class
    "Packable",
    "PackableMetadata",
    "ArrayType",
    # Data handlers
    "DataHandler",
    # Mesh classes
    "Mesh",
    # Array types and utilities
    "Array",
    "EncodedArray",
    "ArrayMetadata",
    "ArrayUtils",
    # Cell type utilities
    "CellType",
    "VTKCellType",
    "CellTypeUtils",
    # Element utilities
    "ElementUtils",
    # Mesh operations
    "MeshUtils",
]
