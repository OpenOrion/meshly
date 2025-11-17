"""
Standardized element processing utilities for meshes.

This module provides utilities for handling both indices and markers in a consistent way,
using sizes instead of offsets for element reconstruction.
"""

import numpy as np
from typing import Union, List, Dict, Tuple, Any, Optional
from .cell_types import CellTypeUtils, VTKCellType


class ElementUtils:
    """Utilities for processing mesh elements (indices, markers) in a standardized way."""
    
    @staticmethod
    def convert_list_to_flattened(elements: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert list of lists to flattened representation with sizes and types.
        
        Args:
            elements: List of element lists (e.g., [[0,1], [1,2,3], [0,2,3,4]])
            
        Returns:
            Tuple of (flattened_indices, sizes, vtk_types)
        """
        if not elements:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), np.array([], dtype=np.uint8)
        
        flattened_indices = []
        sizes = []
        vtk_types = []
        
        for element in elements:
            element_array = np.asarray(element, dtype=np.uint32)
            flattened_indices.extend(element_array.flatten())
            element_size = len(element_array.flatten())
            sizes.append(element_size)
            
            # Determine VTK cell type based on element size
            vtk_type = CellTypeUtils.size_to_vtk_cell_type(element_size)
            # Allow any polygon size - VTK_POLYGON can handle variable sizes
            vtk_types.append(vtk_type)
        
        return (
            np.array(flattened_indices, dtype=np.uint32),
            np.array(sizes, dtype=np.uint32),
            np.array(vtk_types, dtype=np.uint8)
        )
    
    @staticmethod
    def convert_flattened_to_list(
        flattened_indices: np.ndarray,
        sizes: np.ndarray,
        vtk_types: Optional[np.ndarray] = None
    ) -> List[List[int]]:
        """
        Convert flattened representation back to list of lists.
        
        Args:
            flattened_indices: Flattened array of indices
            sizes: Array of element sizes
            vtk_types: Optional array of VTK cell types (for validation)
            
        Returns:
            List of element lists
        """
        if len(flattened_indices) == 0:
            return []
        
        if len(sizes) == 0:
            return []
        
        # Validate that sizes sum matches total indices
        if np.sum(sizes) != len(flattened_indices):
            raise ValueError(
                f"Sum of sizes ({np.sum(sizes)}) does not match "
                f"total number of indices ({len(flattened_indices)})"
            )
        
        # Validate VTK types if provided
        if vtk_types is not None and len(vtk_types) != len(sizes):
            raise ValueError(
                f"Length of vtk_types ({len(vtk_types)}) does not match "
                f"length of sizes ({len(sizes)})"
            )
        
        elements = []
        offset = 0
        for i, size in enumerate(sizes):
            # Validate VTK type against size if provided
            if vtk_types is not None:
                expected_size = CellTypeUtils.vtk_cell_type_to_size(vtk_types[i])
                if expected_size > 0 and expected_size != size:
                    raise ValueError(
                        f"Element {i}: size {size} does not match VTK type {vtk_types[i]} "
                        f"(expected size: {expected_size})"
                    )
            
            element = flattened_indices[offset:offset + size].tolist()
            elements.append(element)
            offset += size
        
        return elements
    
    @staticmethod
    def convert_array_input(
        input_data: Union[np.ndarray, List[Any], None],
        explicit_sizes: Optional[Union[np.ndarray, List[int]]] = None,
        explicit_types: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert various input formats to standardized flattened representation.
        
        Handles:
        - 2D numpy arrays (uniform elements)
        - List of lists (mixed elements)
        - Flat arrays (requires explicit sizes)
        - None (returns None for all)
        
        Args:
            input_data: Input in various formats
            explicit_sizes: Explicitly provided sizes (optional)
            explicit_types: Explicitly provided VTK types (optional)
            
        Returns:
            Tuple of (flattened_indices, sizes, vtk_types) or (None, None, None)
        """
        if input_data is None:
            return None, None, None
        
        inferred_sizes = None
        
        # Convert various input formats to flattened indices
        inferred_vtk_types = None
        if isinstance(input_data, (list, tuple)):
            # Handle list of lists (mixed elements) or flat list
            if len(input_data) > 0 and isinstance(input_data[0], (list, tuple, np.ndarray)):
                # List of lists - mixed element format
                input_data, inferred_sizes, inferred_vtk_types = ElementUtils.convert_list_to_flattened(input_data)
            else:
                # Flat list
                input_data = np.asarray(input_data, dtype=np.uint32).flatten()
                inferred_sizes = None
        else:
            # Numpy array input
            original_shape = input_data.shape
            input_data = np.asarray(input_data, dtype=np.uint32).flatten()
            
            # If it was a 2D array, extract size information
            if len(original_shape) > 1:
                # Infer uniform elements from 2D array shape
                inferred_sizes = np.full(original_shape[0], original_shape[1], dtype=np.uint32)
            else:
                inferred_sizes = None
        
        # Handle sizes
        if explicit_sizes is not None:
            sizes = np.asarray(explicit_sizes, dtype=np.uint32)
            # Validate that explicit sizes matches inferred structure if both exist
            if inferred_sizes is not None:
                if not np.array_equal(sizes, inferred_sizes):
                    raise ValueError(
                        f"Explicit sizes {sizes.tolist()} does not match "
                        f"inferred structure {inferred_sizes.tolist()}"
                    )
        elif inferred_sizes is not None:
            sizes = inferred_sizes
        else:
            # Try to infer triangle structure for legacy flat arrays
            if len(input_data) % 3 == 0:
                # Assume triangles for legacy compatibility
                num_triangles = len(input_data) // 3
                sizes = np.full(num_triangles, 3, dtype=np.uint32)
            else:
                raise ValueError("Cannot determine element structure without explicit sizes")
        
        # Validate that sizes sum matches total indices
        if np.sum(sizes) != len(input_data):
            raise ValueError(
                f"Sum of sizes ({np.sum(sizes)}) does not match "
                f"total number of indices ({len(input_data)})"
            )
        
        # Handle cell types
        if explicit_types is not None:
            vtk_types = np.asarray(explicit_types, dtype=np.uint8)
            # Validate length matches sizes
            if len(vtk_types) != len(sizes):
                raise ValueError(
                    f"Length of vtk_types ({len(vtk_types)}) does not match "
                    f"length of sizes ({len(sizes)})"
                )
            # Validate that sizes and types are consistent
            if not CellTypeUtils.validate_sizes_and_vtk_types(sizes, vtk_types):
                raise ValueError("Element sizes do not match VTK cell types")
        elif inferred_vtk_types is not None:
            # Use inferred types from list conversion
            vtk_types = inferred_vtk_types
        else:
            # Infer VTK types from sizes
            vtk_types = CellTypeUtils.infer_vtk_cell_types_from_sizes(sizes)
        
        return input_data, sizes, vtk_types
    
    @staticmethod
    def get_element_structure(
        flattened_indices: np.ndarray,
        sizes: np.ndarray
    ) -> Union[np.ndarray, List[List[int]]]:
        """
        Get elements in their original structure.
        
        Returns:
            For uniform elements: 2D numpy array where each row is an element
            For mixed elements: List of lists where each sublist is an element
        """
        if len(sizes) == 0:
            return []
        
        # Check if all elements have the same size (uniform)
        if len(np.unique(sizes)) == 1:
            # Uniform case: return as 2D numpy array
            element_size = sizes[0]
            return flattened_indices.reshape(-1, element_size)
        else:
            # Mixed case: return as list of lists
            return ElementUtils.convert_flattened_to_list(flattened_indices, sizes)
    
    @staticmethod
    def is_uniform_elements(sizes: np.ndarray) -> bool:
        """Check if all elements have the same number of vertices."""
        if sizes is None or len(sizes) == 0:
            return True
        return len(np.unique(sizes)) == 1
    
    @staticmethod
    def validate_element_data(
        flattened_indices: np.ndarray,
        sizes: np.ndarray,
        vtk_types: np.ndarray
    ) -> bool:
        """
        Validate that element data is consistent.
        
        Args:
            flattened_indices: Flattened array of indices
            sizes: Array of element sizes
            vtk_types: Array of VTK cell types
            
        Returns:
            True if data is valid
        """
        # Check lengths match
        if len(sizes) != len(vtk_types):
            return False
        
        # Check sizes sum matches total indices
        if np.sum(sizes) != len(flattened_indices):
            return False
        
        # Check sizes and types are consistent
        return CellTypeUtils.validate_sizes_and_vtk_types(sizes, vtk_types)