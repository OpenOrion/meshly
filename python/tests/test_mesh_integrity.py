"""
Tests for the meshoptimizer Python wrapper.

This file contains tests to verify that the mesh vertices indexed by the indices
are the same before and after encoding/decoding, ensuring that the mesh geometry
is preserved correctly.
"""
import numpy as np
import pytest
from io import BytesIO
from meshly import Mesh


class TestMeshIntegrity:
    """Test mesh integrity during encoding/decoding."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        # Create a simple mesh (a cube)
        self.vertices = np.array([
            # positions
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,  # front
            1, 5, 6, 6, 2, 1,  # right
            5, 4, 7, 7, 6, 5,  # back
            4, 0, 3, 3, 7, 4,  # left
            3, 2, 6, 6, 7, 3,  # top
            4, 5, 1, 1, 0, 4   # bottom
        ], dtype=np.uint32)

        self.mesh = Mesh(vertices=self.vertices, indices=self.indices)

    def get_triangles_set(self, vertices, indices):
        """
        Get a set of triangles from vertices and indices.
        Each triangle is represented as a frozenset of tuples of vertex coordinates.
        This makes the comparison invariant to vertex order within triangles.
        """
        triangles = set()
        for i in range(0, len(indices), 3):
            # Get the three vertices of the triangle
            v1 = tuple(vertices[indices[i]])
            v2 = tuple(vertices[indices[i+1]])
            v3 = tuple(vertices[indices[i+2]])
            # Create a frozenset of the vertices (order-invariant)
            triangle = frozenset([v1, v2, v3])
            triangles.add(triangle)
        return triangles

    def test_mesh_integrity_encode_decode(self):
        """Test that mesh vertices indexed by indices are preserved during encoding/decoding."""
        original_triangles = self.get_triangles_set(self.mesh.vertices, self.mesh.indices)

        buffer = BytesIO()
        self.mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)

        assert original_triangles == decoded_triangles

    def test_mesh_integrity_optimize_encode_decode(self):
        """Test that mesh vertices indexed by indices are preserved during optimization, encoding, and decoding."""
        mesh = Mesh(vertices=self.vertices.copy(), indices=self.indices.copy())

        optimized_mesh = mesh.optimize_vertex_cache()
        optimized_mesh = optimized_mesh.optimize_overdraw()
        optimized_mesh = optimized_mesh.optimize_vertex_fetch()

        optimized_triangles = self.get_triangles_set(optimized_mesh.vertices, optimized_mesh.indices)

        buffer = BytesIO()
        optimized_mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)

        assert optimized_triangles == decoded_triangles

    def test_mesh_integrity_simplify_encode_decode(self):
        """Test that mesh vertices indexed by indices are preserved during simplification, encoding, and decoding."""
        segments = 16
        rings = 16
        vertices = []
        indices = []

        for i in range(rings + 1):
            v = i / rings
            phi = v * np.pi

            for j in range(segments):
                u = j / segments
                theta = u * 2 * np.pi

                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)

                vertices.append([x, y, z])

        for i in range(rings):
            for j in range(segments):
                a = i * segments + j
                b = i * segments + (j + 1) % segments
                c = (i + 1) * segments + (j + 1) % segments
                d = (i + 1) * segments + j

                indices.extend([a, b, c])
                indices.extend([a, c, d])

        sphere_vertices = np.array(vertices, dtype=np.float32)
        sphere_indices = np.array(indices, dtype=np.uint32)
        sphere_mesh = Mesh(vertices=sphere_vertices, indices=sphere_indices)

        simplified_mesh = Mesh(vertices=sphere_vertices.copy(), indices=sphere_indices.copy())
        simplified_mesh = simplified_mesh.simplify(target_ratio=0.5)

        simplified_triangles = self.get_triangles_set(simplified_mesh.vertices, simplified_mesh.indices)

        buffer = BytesIO()
        simplified_mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)

        assert simplified_triangles == decoded_triangles

    def test_mesh_integrity_triangles(self):
        """Test that mesh triangles are preserved during encoding/decoding."""
        original_triangles = self.get_triangles_set(self.mesh.vertices, self.mesh.indices)

        buffer = BytesIO()
        self.mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = Mesh.load_from_zip(buffer)

        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)

        assert original_triangles == decoded_triangles
