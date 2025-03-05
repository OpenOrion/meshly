"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class for representing and optimizing 3D meshes
2. EncodedMesh class for storing encoded mesh data
3. Functions for encoding and decoding meshes
"""

import numpy as np
from typing import Optional, Tuple

# Use meshoptimizer directly
from meshoptimizer import (
    # Encoder functions
    encode_vertex_buffer,
    encode_index_buffer,
    decode_vertex_buffer,
    decode_index_buffer,
    optimize_vertex_cache,
    optimize_overdraw,
    optimize_vertex_fetch,
    simplify,
)

class EncodedMesh:
    """
    Class representing an encoded mesh with its vertices and indices.
    """
    def __init__(self, vertices: bytes, indices: Optional[bytes], 
                vertex_count: int, vertex_size: int,
                index_count: Optional[int], index_size: int) -> None:
        """
        Initialize an EncodedMesh object.
        
        Args:
            vertices: Encoded vertex buffer
            indices: Encoded index buffer (optional)
            vertex_count: Number of vertices
            vertex_size: Size of each vertex in bytes
            index_count: Number of indices (optional)
            index_size: Size of each index in bytes
        """
        self.vertices = vertices
        self.indices = indices
        self.vertex_count = vertex_count
        self.vertex_size = vertex_size
        self.index_count = index_count
        self.index_size = index_size

class Mesh:
    """
    A class representing a 3D mesh with optimization capabilities.
    """
    
    def __init__(self, vertices: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        """
        Initialize a mesh with vertices and optional indices.
        
        Args:
            vertices: numpy array of vertex data
            indices: numpy array of indices (optional)
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.uint32) if indices is not None else None
        self.vertex_count = len(vertices)
        self.index_count = len(indices) if indices is not None else 0
    
    def optimize_vertex_cache(self) -> 'Mesh':
        """
        Optimize the mesh for vertex cache efficiency.
        
        Returns:
            self (for method chaining)
        """
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")
        
        optimized_indices = np.zeros_like(self.indices)
        optimize_vertex_cache(
            optimized_indices, 
            self.indices, 
            self.index_count, 
            self.vertex_count
        )
        
        self.indices = optimized_indices
        return self
    
    def optimize_overdraw(self, threshold: float = 1.05) -> 'Mesh':
        """
        Optimize the mesh for overdraw.
        
        Args:
            threshold: threshold for optimization (default: 1.05)
            
        Returns:
            self (for method chaining)
        """
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")
        
        optimized_indices = np.zeros_like(self.indices)
        optimize_overdraw(
            optimized_indices, 
            self.indices, 
            self.vertices, 
            self.index_count, 
            self.vertex_count, 
            self.vertices.itemsize * self.vertices.shape[1], 
            threshold
        )
        
        self.indices = optimized_indices
        return self
    
    def optimize_vertex_fetch(self) -> 'Mesh':
        """
        Optimize the mesh for vertex fetch efficiency.
        
        Returns:
            self (for method chaining)
        """
        if self.indices is None:
            raise ValueError("Mesh has no indices to optimize")
        
        optimized_vertices = np.zeros_like(self.vertices)
        unique_vertex_count = optimize_vertex_fetch(
            optimized_vertices, 
            self.indices, 
            self.vertices, 
            self.index_count, 
            self.vertex_count, 
            self.vertices.itemsize * self.vertices.shape[1]
        )
        
        self.vertices = optimized_vertices[:unique_vertex_count]
        self.vertex_count = unique_vertex_count
        return self
    
    def simplify(self, target_ratio: float = 0.25, target_error: float = 0.01, options: int = 0) -> 'Mesh':
        """
        Simplify the mesh.
        
        Args:
            target_ratio: ratio of triangles to keep (default: 0.25)
            target_error: target error (default: 0.01)
            options: simplification options (default: 0)
            
        Returns:
            self (for method chaining)
        """
        if self.indices is None:
            raise ValueError("Mesh has no indices to simplify")
        
        target_index_count = int(self.index_count * target_ratio)
        simplified_indices = np.zeros(self.index_count, dtype=np.uint32)
        
        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = simplify(
            simplified_indices, 
            self.indices, 
            self.vertices, 
            self.index_count, 
            self.vertex_count, 
            self.vertices.itemsize * self.vertices.shape[1], 
            target_index_count, 
            target_error, 
            options, 
            result_error
        )
        
        self.indices = simplified_indices[:new_index_count]
        self.index_count = new_index_count
        return self
    
    def encode(self) -> EncodedMesh:
        """
        Encode the mesh for efficient transmission.
        
        Returns:
            EncodedMesh object containing encoded buffers and size information
        """
        # Encode vertex buffer
        encoded_vertices = encode_vertex_buffer(
            self.vertices,
            self.vertex_count,
            self.vertices.itemsize * self.vertices.shape[1]
        )
        
        # Encode index buffer if present
        encoded_indices = None
        if self.indices is not None:
            encoded_indices = encode_index_buffer(
                self.indices,
                self.index_count,
                self.indices.itemsize
            )
        
        return EncodedMesh(
            vertices=encoded_vertices,
            indices=encoded_indices,
            vertex_count=self.vertex_count,
            vertex_size=self.vertices.itemsize * self.vertices.shape[1],
            index_count=self.index_count if self.indices is not None else None,
            index_size=self.indices.itemsize if self.indices is not None else 4
        )
    
    @classmethod
    def decode(cls, encoded_mesh: EncodedMesh) -> 'Mesh':
        """
        Decode an encoded mesh.
        
        Args:
            encoded_mesh: EncodedMesh object to decode
            
        Returns:
            Decoded Mesh object
        """
        # Decode vertex buffer
        vertices = decode_vertex_buffer(
            encoded_mesh.vertex_count,
            encoded_mesh.vertex_size,
            encoded_mesh.vertices
        )
        
        # Decode index buffer if present
        indices = None
        if encoded_mesh.indices is not None and encoded_mesh.index_count is not None:
            indices = decode_index_buffer(
                encoded_mesh.index_count,
                encoded_mesh.index_size,
                encoded_mesh.indices
            )
        
        return cls(vertices, indices)

def encode_mesh(vertices: np.ndarray,
              indices: Optional[np.ndarray] = None,
              vertex_count: Optional[int] = None,
              vertex_size: Optional[int] = None,
              index_count: Optional[int] = None) -> EncodedMesh:
    """
    Encode mesh data for efficient transmission.
    
    Args:
        vertices: Vertex buffer
        indices: Index buffer (optional)
        vertex_count: Number of vertices (optional, defaults to len(vertices))
        vertex_size: Size of each vertex in bytes (optional, defaults to vertices.itemsize * vertices.shape[1])
        index_count: Number of indices (optional, defaults to len(indices))
        
    Returns:
        EncodedMesh object containing encoded data
    """
    # Set default values
    if vertex_count is None:
        vertex_count = len(vertices)
    if vertex_size is None:
        vertex_size = vertices.itemsize * vertices.shape[1]
    if indices is not None and index_count is None:
        index_count = len(indices)
    
    # Create a mesh object
    mesh = Mesh(vertices, indices)
    
    # Return encoded mesh
    return mesh.encode()

def decode_mesh(encoded_mesh: EncodedMesh) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Decode mesh data.
    
    Args:
        encoded_mesh: EncodedMesh object containing encoded data
        
    Returns:
        Tuple of (vertices, indices), where indices may be None
    """
    # Create a mesh object from the encoded data
    mesh = Mesh.decode(encoded_mesh)
    
    # Return vertices and indices
    return mesh.vertices, mesh.indices