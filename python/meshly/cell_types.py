"""
Cell type utilities for mesh processing.

This module provides standardized cell type definitions and conversion utilities
for working with different mesh element types.
"""

import numpy as np
from typing import Union, List



class CellType:
    """Standard cell type definitions."""
    POINT = 15
    LINE = 1
    TRIANGLE = 2
    QUADRILATERAL = 3
    TETRAHEDRON = 4
    HEXAHEDRON = 5
    PRISM = 6
    PYRAMID = 7


class VTKCellType:
    """VTK cell type definitions for compatibility."""
    VTK_VERTEX = 1
    VTK_LINE = 3
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    VTK_TETRA = 10
    VTK_HEXAHEDRON = 12
    VTK_WEDGE = 13
    VTK_PYRAMID = 14
    VTK_POLYGON = 7



class CellTypeUtils:
    """Utilities for working with cell types."""
    
    @staticmethod
    def size_to_vtk_cell_type(size: int) -> int:
        """
        Convert element size to VTK cell type.
        
        Args:
            size: Number of vertices in the element
            
        Returns:
            VTK cell type identifier
        """
        if size == 1:
            return VTKCellType.VTK_VERTEX
        elif size == 2:
            return VTKCellType.VTK_LINE
        elif size == 3:
            return VTKCellType.VTK_TRIANGLE
        elif size == 4:
            return VTKCellType.VTK_QUAD
        elif size == 5:
            return VTKCellType.VTK_PYRAMID
        elif size == 6:
            return VTKCellType.VTK_WEDGE
        elif size == 8:
            return VTKCellType.VTK_HEXAHEDRON
        else:
            return VTKCellType.VTK_POLYGON
    
    @staticmethod
    def vtk_cell_type_to_size(vtk_type: int) -> int:
        """
        Convert VTK cell type to element size.
        
        Args:
            vtk_type: VTK cell type identifier
            
        Returns:
            Number of vertices in the element
        """
        vtk_type_to_size = {
            VTKCellType.VTK_VERTEX: 1,
            VTKCellType.VTK_LINE: 2,
            VTKCellType.VTK_TRIANGLE: 3,
            VTKCellType.VTK_QUAD: 4,
            VTKCellType.VTK_TETRA: 4,
            VTKCellType.VTK_PYRAMID: 5,
            VTKCellType.VTK_WEDGE: 6,
            VTKCellType.VTK_HEXAHEDRON: 8,
        }
        
        return vtk_type_to_size.get(vtk_type, 0)
    
    @staticmethod
    def infer_vtk_cell_types_from_sizes(sizes: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Infer VTK cell types from element sizes.
        
        Args:
            sizes: Array or list of element sizes
            
        Returns:
            Array of VTK cell type identifiers
        """
        sizes_array = np.asarray(sizes)
        cell_types = np.zeros(len(sizes_array), dtype=np.uint32)
        
        for i, size in enumerate(sizes_array):
            cell_types[i] = CellTypeUtils.size_to_vtk_cell_type(size)
        
        return cell_types
    
    @staticmethod
    def infer_sizes_from_vtk_cell_types(vtk_cell_types: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Infer element sizes from VTK cell types.
        
        Args:
            vtk_cell_types: Array or list of VTK cell type identifiers
            
        Returns:
            Array of element sizes
        """
        vtk_cell_types = np.asarray(vtk_cell_types)
        sizes = np.zeros(len(vtk_cell_types), dtype=np.uint32)
        
        for i, vtk_type in enumerate(vtk_cell_types):
            size = CellTypeUtils.vtk_cell_type_to_size(vtk_type)
            if size == 0:
                raise ValueError(f"Unknown VTK cell type: {vtk_type}")
            sizes[i] = size
        
        return sizes
    
    @staticmethod
    def validate_sizes_and_vtk_types(sizes: np.ndarray, vtk_types: np.ndarray) -> bool:
        """
        Validate that element sizes match VTK cell types.
        
        Args:
            sizes: Array of element sizes
            vtk_types: Array of VTK cell type identifiers
            
        Returns:
            True if sizes and types are consistent
        """
        if len(sizes) != len(vtk_types):
            return False
        
        for size, vtk_type in zip(sizes, vtk_types):
            expected_size = CellTypeUtils.vtk_cell_type_to_size(vtk_type)
            if expected_size != size:
                return False
        
        return True