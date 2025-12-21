"""
Cell type utilities for mesh processing.

This module provides standardized cell type definitions and conversion utilities
for working with different mesh element types, including edge topology for 3D
cell types used in wireframe visualization.
"""

import numpy as np
from typing import Union, List, Tuple, Sequence


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
    """Utilities for working with cell types, including edge topology."""
    
    # Edge topology for VTK cell types
    # Each entry maps local vertex indices to edges
    VTK_EDGE_TOPOLOGY = {
        VTKCellType.VTK_LINE: [(0, 1)],
        VTKCellType.VTK_TRIANGLE: [(0, 1), (1, 2), (2, 0)],
        VTKCellType.VTK_QUAD: [(0, 1), (1, 2), (2, 3), (3, 0)],
        VTKCellType.VTK_TETRA: [
            (0, 1), (0, 2), (0, 3),  # edges from vertex 0
            (1, 2), (1, 3),          # edges from vertex 1
            (2, 3),                   # edge from vertex 2
        ],
        VTKCellType.VTK_HEXAHEDRON: [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ],
        VTKCellType.VTK_WEDGE: [
            (0, 1), (1, 2), (2, 0),  # bottom triangle
            (3, 4), (4, 5), (5, 3),  # top triangle
            (0, 3), (1, 4), (2, 5),  # vertical edges
        ],
        VTKCellType.VTK_PYRAMID: [
            (0, 1), (1, 2), (2, 3), (3, 0),  # base quad
            (0, 4), (1, 4), (2, 4), (3, 4),  # edges to apex
        ],
    }
    
    @staticmethod
    def get_edge_topology(vtk_type: int) -> np.ndarray:
        """
        Get the edge topology (local vertex index pairs) for a VTK cell type.
        
        Args:
            vtk_type: VTK cell type identifier
            
        Returns:
            numpy array of shape (n_edges, 2) with local vertex indices.
            Returns empty array for unknown cell types.
        """
        topology = CellTypeUtils.VTK_EDGE_TOPOLOGY.get(vtk_type)
        if topology is None:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(topology, dtype=np.int32)
    
    @staticmethod
    def get_cell_edges(
        cell_vertices: Sequence[int], 
        cell_type: int
    ) -> np.ndarray:
        """
        Extract global vertex edges from a cell based on its VTK cell type.
        
        Args:
            cell_vertices: Sequence of global vertex indices for this cell
            cell_type: VTK cell type constant
            
        Returns:
            numpy array of shape (n_edges, 2) with global vertex indices, 
            where each row [u, v] has u < v
        """
        cell_verts = np.asarray(cell_vertices)
        edge_topology = CellTypeUtils.get_edge_topology(cell_type)
        
        if len(edge_topology) > 0:
            # Vectorized: index into cell_verts using edge topology
            edges = cell_verts[edge_topology]  # shape (n_edges, 2)
        else:
            # Unknown type - treat as polygon (connect consecutive vertices)
            n = len(cell_verts)
            indices = np.arange(n)
            edges = np.stack([cell_verts[indices], cell_verts[(indices + 1) % n]], axis=1)
        
        # Normalize edges so u < v (vectorized sort along axis 1)
        edges = np.sort(edges, axis=1)
        return edges
    
    @staticmethod
    def get_edges_from_element_size(
        element: Sequence[int]
    ) -> np.ndarray:
        """
        Extract edges from an element based on its vertex count.
        
        This is useful when cell_type is not available. The function infers
        the cell type from the number of vertices:
        - 2 vertices: line
        - 3 vertices: triangle
        - 4 vertices: quad (not tetrahedron - use get_cell_edges for 3D)
        - 5 vertices: pyramid
        - 6 vertices: wedge/prism
        - 8 vertices: hexahedron
        
        Args:
            element: Sequence of global vertex indices
            
        Returns:
            numpy array of shape (n_edges, 2) with global vertex indices,
            where each row [u, v] has u < v
        """
        element_arr = np.asarray(element)
        n = len(element_arr)
        
        # Map vertex count to likely VTK type
        size_to_type = {
            2: VTKCellType.VTK_LINE,
            3: VTKCellType.VTK_TRIANGLE,
            4: VTKCellType.VTK_QUAD,  # Assumes surface element
            5: VTKCellType.VTK_PYRAMID,
            6: VTKCellType.VTK_WEDGE,
            8: VTKCellType.VTK_HEXAHEDRON,
        }
        
        vtk_type = size_to_type.get(n)
        if vtk_type is not None:
            return CellTypeUtils.get_cell_edges(element_arr, vtk_type)
        
        # Unknown element size - treat as polygon (vectorized)
        indices = np.arange(n)
        edges = np.stack([element_arr[indices], element_arr[(indices + 1) % n]], axis=1)
        edges = np.sort(edges, axis=1)
        return edges
    
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

    @staticmethod
    def get_mesh_dimension(cell_types: np.ndarray) -> int:
        """
        Determine mesh dimension from cell types.

        Args:
            cell_types: Array of VTK cell types

        Returns:
            Dimension: 2 for surface meshes, 3 for volume meshes
        """
        # Map VTK cell types to dimensions
        # 2D elements: TRIANGLE (5), QUAD (9)
        # 3D elements: TETRA (10), HEXAHEDRON (12), WEDGE (13), PYRAMID (14)
        vtk_2d_types = {
            VTKCellType.VTK_TRIANGLE,
            VTKCellType.VTK_QUAD,
        }
        vtk_3d_types = {
            VTKCellType.VTK_TETRA,
            VTKCellType.VTK_HEXAHEDRON,
            VTKCellType.VTK_WEDGE,
            VTKCellType.VTK_PYRAMID,
        }

        unique_types = set(cell_types)

        # Check for 3D elements first
        if unique_types & vtk_3d_types:
            return 3
        # Check for 2D elements
        elif unique_types & vtk_2d_types:
            return 2
        else:
            # Default to 2D for other cases (lines, points, polygons)
            return 2