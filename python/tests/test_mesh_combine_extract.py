"""
Tests for mesh combine and extract_by_marker methods.
"""

import unittest
import numpy as np
from meshly.mesh import Mesh


class TestMeshCombine(unittest.TestCase):
    """Test cases for Mesh.combine() method."""

    def test_combine_simple_meshes(self):
        """Test combining two simple meshes without markers."""
        # Create two triangle meshes
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        # Combine meshes
        combined = Mesh.combine([mesh1, mesh2])

        # Check vertices
        self.assertEqual(combined.vertex_count, 6)
        self.assertTrue(
            np.array_equal(combined.vertices[:3], mesh1.vertices),
            "First mesh vertices should be preserved",
        )
        self.assertTrue(
            np.array_equal(combined.vertices[3:], mesh2.vertices),
            "Second mesh vertices should be appended",
        )

        # Check indices (should be offset for second mesh)
        expected_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        self.assertTrue(
            np.array_equal(combined.indices, expected_indices),
            "Indices should be properly offset",
        )

    def test_combine_with_marker_names(self):
        """Test combining meshes with marker names."""
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        # Combine with marker names
        combined = Mesh.combine(
            [mesh1, mesh2], marker_names=["part1", "part2"])

        # Check markers exist
        self.assertIn("part1", combined.markers)
        self.assertIn("part2", combined.markers)

        # Check marker indices
        self.assertTrue(
            np.array_equal(
                combined.markers["part1"], np.array([0, 1, 2], dtype=np.uint32)
            )
        )
        self.assertTrue(
            np.array_equal(
                combined.markers["part2"], np.array([3, 4, 5], dtype=np.uint32)
            )
        )

    def test_combine_preserve_existing_markers(self):
        """Test combining meshes while preserving existing markers."""
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"edge": np.array([1, 2], dtype=np.uint32)},
            marker_sizes={"edge": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"edge": np.array([1, 1], dtype=np.uint32)},
        )

        # Combine with marker preservation
        combined = Mesh.combine([mesh1, mesh2], preserve_markers=True)

        # Check preserved markers exist with original names
        self.assertIn("boundary", combined.markers)
        self.assertIn("edge", combined.markers)

        # Check marker indices are properly offset
        self.assertTrue(
            np.array_equal(
                combined.markers["boundary"], np.array([0, 1], dtype=np.uint32)
            )
        )
        self.assertTrue(
            np.array_equal(
                combined.markers["edge"], np.array([4, 5], dtype=np.uint32)
            )
        )

    def test_combine_with_marker_names_and_preserve(self):
        """Test that marker_names takes precedence over existing markers."""
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"top": np.array([2], dtype=np.uint32)},
            marker_sizes={"top": np.array([1], dtype=np.uint32)},
            marker_cell_types={"top": np.array([1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        combined = Mesh.combine(
            [mesh1, mesh2], marker_names=["left", "right"], preserve_markers=True
        )

        # When marker_names is provided, only those markers should exist
        # preserve_markers is ignored
        self.assertIn("left", combined.markers)
        self.assertIn("right", combined.markers)
        self.assertNotIn("top", combined.markers)

        # Should have exactly 2 markers
        self.assertEqual(len(combined.markers), 2)

    def test_combine_same_marker_names(self):
        """Test combining meshes with same marker names - should merge them."""
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            markers={"boundary": np.array([1, 2], dtype=np.uint32)},
            marker_sizes={"boundary": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"boundary": np.array([1, 1], dtype=np.uint32)},
        )

        combined = Mesh.combine([mesh1, mesh2], preserve_markers=True)

        # Should have one "boundary" marker with combined elements
        self.assertIn("boundary", combined.markers)

        # Should have 4 elements total (2 from each mesh)
        self.assertEqual(len(combined.markers["boundary"]), 4)

        # Check that indices are properly offset
        # mesh1: [0, 1], mesh2: [1, 2] -> offset by 3 -> [4, 5]
        expected_indices = np.array([0, 1, 4, 5], dtype=np.uint32)
        self.assertTrue(np.array_equal(
            combined.markers["boundary"], expected_indices))

    def test_combine_empty_list(self):
        """Test that combining empty list raises error."""
        with self.assertRaisesRegex(ValueError, "Cannot combine empty list"):
            Mesh.combine([])

    def test_combine_marker_names_length_mismatch(self):
        """Test that mismatched marker_names length raises error."""
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        with self.assertRaisesRegex(ValueError, "marker_names length"):
            Mesh.combine([mesh1], marker_names=["part1", "part2"])


class TestMeshExtract(unittest.TestCase):
    """Test cases for Mesh.extract_by_marker() method."""

    def test_extract_by_marker(self):
        """Test extracting a submesh by marker name."""
        # Create a mesh with a marker
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
        )
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            markers={"edge": np.array([0, 1], dtype=np.uint32)},
            marker_sizes={"edge": np.array([1, 1], dtype=np.uint32)},
            marker_cell_types={"edge": np.array([1, 1], dtype=np.uint32)},
        )

        # Extract by marker
        extracted = mesh.extract_by_marker("edge")

        # Should only have 2 vertices (0 and 1 from original mesh)
        self.assertEqual(extracted.vertex_count, 2)
        self.assertTrue(np.array_equal(extracted.vertices[0], vertices[0]))
        self.assertTrue(np.array_equal(extracted.vertices[1], vertices[1]))

        # Indices should be remapped to 0, 1
        self.assertTrue(
            np.array_equal(extracted.indices, np.array(
                [0, 1], dtype=np.uint32))
        )

    def test_extract_by_marker_with_triangles(self):
        """Test extracting a submesh with triangle elements."""
        # Create a mesh with triangle markers
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32
        )

        mesh = Mesh(
            vertices=vertices,
            markers={
                "boundary": np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
            },  # Two triangles
            marker_sizes={"boundary": np.array([3, 3], dtype=np.uint32)},
            marker_cell_types={
                "boundary": np.array([5, 5], dtype=np.uint32)
            },  # Triangle type
        )

        extracted = mesh.extract_by_marker("boundary")

        # Should have 4 unique vertices (0, 1, 2, 3)
        self.assertEqual(extracted.vertex_count, 4)

        # Should have 6 indices (two triangles)
        self.assertEqual(extracted.index_count, 6)

        # Verify the triangles are properly remapped
        # Original indices: [0, 1, 2, 1, 3, 2]
        # Unique vertices: [0, 1, 2, 3] -> map to [0, 1, 2, 3]
        # Expected remapped: [0, 1, 2, 1, 3, 2]
        self.assertTrue(
            np.array_equal(
                extracted.indices, np.array(
                    [0, 1, 2, 1, 3, 2], dtype=np.uint32)
            )
        )

    def test_extract_by_marker_nonexistent(self):
        """Test that extracting by nonexistent marker raises error."""
        mesh = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        with self.assertRaisesRegex(ValueError, "Marker 'nonexistent' not found"):
            mesh.extract_by_marker("nonexistent")


class TestMeshCombineAndExtract(unittest.TestCase):
    """Test cases for combining and extracting meshes."""

    def test_combine_and_extract_roundtrip(self):
        """Test combining meshes and extracting them back."""
        # Create two distinct meshes
        mesh1 = Mesh(
            vertices=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        mesh2 = Mesh(
            vertices=np.array(
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        # Combine with markers
        combined = Mesh.combine(
            [mesh1, mesh2], marker_names=["part1", "part2"])

        # Extract part1 back
        extracted1 = combined.extract_by_marker("part1")

        # Should match original mesh1 vertices
        self.assertEqual(extracted1.vertex_count, mesh1.vertex_count)
        self.assertTrue(np.allclose(extracted1.vertices, mesh1.vertices))

        # Extract part2 back
        extracted2 = combined.extract_by_marker("part2")

        # Should match original mesh2 vertices
        self.assertEqual(extracted2.vertex_count, mesh2.vertex_count)
        self.assertTrue(np.allclose(extracted2.vertices, mesh2.vertices))


if __name__ == "__main__":
    unittest.main()
