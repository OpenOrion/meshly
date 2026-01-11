"""
High-level export functionality for meshoptimizer.

This package provides high-level abstractions and utilities for working with
meshoptimizer, including:

1. Packable base class for automatic array serialization
2. Mesh class as a Pydantic base class for representing 3D meshes
3. MeshUtils static class for mesh optimization operations
4. ArrayUtils class for array encoding/decoding operations
5. EncodedMesh class for storing encoded mesh data
6. I/O utilities for storing and loading meshes and arrays
7. Support for custom subclasses with automatic encoding/decoding of numpy arrays
8. CellTypeUtils for VTK cell type conversions and edge topology extraction
"""

from .packable import (
    Packable,
    PackableMetadata,
    EncodedData,
)

from .mesh import (
    Mesh,
    EncodedMesh,
    Array,
    HAS_JAX,
)

from .array import (
    EncodedArray,
    ArrayMetadata,
    EncodedArrayModel,
    ArrayUtils,
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


__all__ = [
    # Packable base class
    "Packable",
    "PackableMetadata",
    "EncodedData",
    # Mesh classes
    "Mesh",
    "EncodedMesh",
    # Array types and utilities
    "Array",
    "HAS_JAX",
    "EncodedArray",
    "EncodedArrayModel",
    "ArrayMetadata",
    "ArrayResult",
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
