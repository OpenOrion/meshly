"""Unit tests for cell_types module edge topology functions."""

import unittest
import numpy as np
from meshly import CellTypeUtils, VTKCellType


class TestEdgeTopology(unittest.TestCase):
    """Tests for VTK cell type edge topology."""

    def test_get_edge_topology_hexahedron(self):
        """Hexahedron should have 12 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_HEXAHEDRON)
        self.assertEqual(len(edges), 12)
        self.assertEqual(edges.shape, (12, 2))

    def test_get_edge_topology_tetrahedron(self):
        """Tetrahedron should have 6 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TETRA)
        self.assertEqual(len(edges), 6)
        self.assertEqual(edges.shape, (6, 2))

    def test_get_edge_topology_wedge(self):
        """Wedge/prism should have 9 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_WEDGE)
        self.assertEqual(len(edges), 9)

    def test_get_edge_topology_pyramid(self):
        """Pyramid should have 8 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_PYRAMID)
        self.assertEqual(len(edges), 8)

    def test_get_edge_topology_triangle(self):
        """Triangle should have 3 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TRIANGLE)
        self.assertEqual(len(edges), 3)

    def test_get_edge_topology_quad(self):
        """Quad should have 4 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_QUAD)
        self.assertEqual(len(edges), 4)

    def test_get_edge_topology_unknown(self):
        """Unknown cell type should return empty array."""
        edges = CellTypeUtils.get_edge_topology(999)
        self.assertEqual(len(edges), 0)
        self.assertEqual(edges.shape, (0, 2))


class TestGetCellEdges(unittest.TestCase):
    """Tests for get_cell_edges with global vertex indices."""

    def test_hexahedron_edges(self):
        """Test hex edges with global vertex indices."""
        vertices = [10, 20, 30, 40, 50, 60, 70, 80]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_HEXAHEDRON)
        
        self.assertEqual(len(edges), 12)
        self.assertEqual(edges.shape, (12, 2))
        # All edges should be (min, max) ordered
        for u, v in edges:
            self.assertLess(u, v)
        
        # Check specific edges exist (as sorted tuples)
        edge_set = {tuple(e) for e in edges}
        self.assertIn((10, 20), edge_set)
        self.assertIn((10, 50), edge_set)

    def test_triangle_edges(self):
        """Test triangle edges."""
        vertices = [5, 10, 15]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_TRIANGLE)
        
        self.assertEqual(len(edges), 3)
        edge_set = {tuple(e) for e in edges}
        self.assertIn((5, 10), edge_set)
        self.assertIn((10, 15), edge_set)
        self.assertIn((5, 15), edge_set)

    def test_tetrahedron_edges(self):
        """Test tetrahedron edges."""
        vertices = [0, 1, 2, 3]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_TETRA)
        
        self.assertEqual(len(edges), 6)
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        edge_set = {tuple(e) for e in edges}
        self.assertEqual(edge_set, expected)

    def test_unknown_type_as_polygon(self):
        """Unknown type should be treated as polygon."""
        vertices = [0, 1, 2, 3, 4]
        edges = CellTypeUtils.get_cell_edges(vertices, 999)
        
        self.assertEqual(len(edges), 5)
        edge_set = {tuple(e) for e in edges}
        self.assertIn((0, 1), edge_set)
        self.assertIn((1, 2), edge_set)
        self.assertIn((2, 3), edge_set)
        self.assertIn((3, 4), edge_set)
        self.assertIn((0, 4), edge_set)


class TestGetEdgesFromElementSize(unittest.TestCase):
    """Tests for inferring edges from element vertex count."""

    def test_8_vertices_is_hex(self):
        """8 vertices should be treated as hexahedron."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(8)))
        self.assertEqual(len(edges), 12)

    def test_6_vertices_is_wedge(self):
        """6 vertices should be treated as wedge."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(6)))
        self.assertEqual(len(edges), 9)

    def test_5_vertices_is_pyramid(self):
        """5 vertices should be treated as pyramid."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(5)))
        self.assertEqual(len(edges), 8)

    def test_4_vertices_is_quad(self):
        """4 vertices should be treated as quad (not tetra)."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(4)))
        self.assertEqual(len(edges), 4)

    def test_3_vertices_is_triangle(self):
        """3 vertices should be treated as triangle."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(3)))
        self.assertEqual(len(edges), 3)

    def test_2_vertices_is_line(self):
        """2 vertices should be treated as line."""
        edges = CellTypeUtils.get_edges_from_element_size([0, 1])
        self.assertEqual(len(edges), 1)
        self.assertTrue(np.array_equal(edges[0], [0, 1]))

    def test_unknown_size_as_polygon(self):
        """Unknown size (e.g., 7) should be treated as polygon."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(7)))
        self.assertEqual(len(edges), 7)


if __name__ == '__main__':
    unittest.main()
