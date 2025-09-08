"""
Tests for dictionary fields containing numpy arrays.

This test verifies that the library can handle mesh fields that contain
dictionaries of numpy arrays, extracting them for encoding while preserving
the dictionary structure for reconstruction.
"""
import os
import tempfile
import numpy as np
import unittest
from typing import Dict
from pydantic import Field

from meshly import Mesh, MeshUtils


class TexturedMesh(Mesh):
    """A custom mesh class with dictionary fields containing numpy arrays."""
    
    # Dictionary containing multiple texture arrays
    textures: Dict[str, np.ndarray] = Field(
        default_factory=dict, 
        description="Dictionary of texture arrays"
    )
    
    # Dictionary containing nested dictionaries with arrays
    material_data: Dict[str, Dict[str, np.ndarray]] = Field(
        default_factory=dict,
        description="Nested dictionary structure with arrays"
    )
    
    # Regular non-array field
    material_name: str = Field("default", description="Material name")


class TestDictArrays(unittest.TestCase):
    """Test dictionary fields containing numpy arrays."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple mesh (a triangle)
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=np.float32)

        self.indices = np.array([0, 1, 2], dtype=np.uint32)
        
        # Create texture arrays
        self.diffuse_texture = np.random.random((64, 64, 3)).astype(np.float32)
        self.normal_texture = np.random.random((64, 64, 3)).astype(np.float32)
        self.specular_texture = np.random.random((64, 64, 1)).astype(np.float32)
        
        # Create material property arrays
        self.roughness_map = np.random.random((32, 32)).astype(np.float32)
        self.metallic_map = np.random.random((32, 32)).astype(np.float32)
        self.emission_map = np.random.random((32, 32, 3)).astype(np.float32)
        
    def test_dict_array_detection(self):
        """Test that dictionary arrays are correctly detected."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            textures={
                "diffuse": self.diffuse_texture,
                "normal": self.normal_texture,
                "specular": self.specular_texture
            },
            material_data={
                "surface": {
                    "roughness": self.roughness_map,
                    "metallic": self.metallic_map
                },
                "lighting": {
                    "emission": self.emission_map
                }
            },
            material_name="test_material"
        )
        
        # Check that nested arrays are detected
        array_fields = mesh.array_fields
        expected_fields = {
            "vertices", "indices",
            "textures.diffuse", "textures.normal", "textures.specular",
            "material_data.surface.roughness", "material_data.surface.metallic",
            "material_data.lighting.emission"
        }
        
        self.assertEqual(array_fields, expected_fields)
        
        # Check dictionary structure metadata
        dict_structure = mesh.dict_structure_metadata
        self.assertIn("textures", dict_structure)
        self.assertIn("material_data", dict_structure)
        
        # Verify textures structure
        textures_structure = dict_structure["textures"]
        self.assertEqual(textures_structure["diffuse"]["type"], "array")
        self.assertEqual(textures_structure["normal"]["type"], "array")
        self.assertEqual(textures_structure["specular"]["type"], "array")
        
        # Verify nested material_data structure
        material_structure = dict_structure["material_data"]
        self.assertEqual(material_structure["surface"]["type"], "dict")
        self.assertEqual(material_structure["lighting"]["type"], "dict")
        
    def test_dict_array_encoding_decoding(self):
        """Test that dictionary arrays can be encoded and decoded."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            textures={
                "diffuse": self.diffuse_texture,
                "normal": self.normal_texture,
                "specular": self.specular_texture
            },
            material_data={
                "surface": {
                    "roughness": self.roughness_map,
                    "metallic": self.metallic_map
                },
                "lighting": {
                    "emission": self.emission_map
                }
            },
            material_name="test_material"
        )
        
        # Encode the mesh
        encoded_mesh = MeshUtils.encode(mesh)
        
        # Check that all arrays are encoded
        expected_array_names = {
            "textures.diffuse", "textures.normal", "textures.specular",
            "material_data.surface.roughness", "material_data.surface.metallic",
            "material_data.lighting.emission"
        }
        
        self.assertEqual(set(encoded_mesh.arrays.keys()), expected_array_names)
        
        # Decode the mesh
        decoded_mesh = MeshUtils.decode(TexturedMesh, encoded_mesh, mesh.dict_structure_metadata)
        
        # Manually set non-array fields since direct decode doesn't handle them
        # (This is normally done by load_from_zip via metadata.field_data)
        decoded_mesh.material_name = mesh.material_name
        
        # Verify basic mesh properties
        self.assertEqual(decoded_mesh.vertex_count, mesh.vertex_count)
        self.assertEqual(decoded_mesh.index_count, mesh.index_count)
        np.testing.assert_array_almost_equal(decoded_mesh.vertices, mesh.vertices)
        np.testing.assert_array_almost_equal(decoded_mesh.indices, mesh.indices)
        
        # Verify dictionary structure is preserved
        self.assertIsInstance(decoded_mesh.textures, dict)
        self.assertIsInstance(decoded_mesh.material_data, dict)
        
        # Verify texture arrays
        self.assertIn("diffuse", decoded_mesh.textures)
        self.assertIn("normal", decoded_mesh.textures)
        self.assertIn("specular", decoded_mesh.textures)
        
        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["diffuse"], self.diffuse_texture, decimal=5
        )
        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["normal"], self.normal_texture, decimal=5
        )
        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["specular"], self.specular_texture, decimal=5
        )
        
        # Verify nested material data
        self.assertIn("surface", decoded_mesh.material_data)
        self.assertIn("lighting", decoded_mesh.material_data)
        
        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["surface"]["roughness"], self.roughness_map, decimal=5
        )
        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["surface"]["metallic"], self.metallic_map, decimal=5
        )
        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["lighting"]["emission"], self.emission_map, decimal=5
        )
        
        # Verify non-array field is preserved
        self.assertEqual(decoded_mesh.material_name, "test_material")
        
    def test_dict_array_zip_serialization(self):
        """Test that dictionary arrays work with zip file serialization."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            textures={
                "diffuse": self.diffuse_texture,
                "normal": self.normal_texture
            },
            material_data={
                "surface": {
                    "roughness": self.roughness_map
                }
            },
            material_name="zip_test_material"
        )
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the mesh to a zip file
            MeshUtils.save_to_zip(mesh, temp_path)
            
            # Load the mesh from the zip file
            loaded_mesh = MeshUtils.load_from_zip(TexturedMesh, temp_path)
            
            # Verify all data is preserved
            self.assertEqual(loaded_mesh.vertex_count, mesh.vertex_count)
            self.assertEqual(loaded_mesh.index_count, mesh.index_count)
            np.testing.assert_array_almost_equal(loaded_mesh.vertices, mesh.vertices)
            np.testing.assert_array_almost_equal(loaded_mesh.indices, mesh.indices)
            
            # Verify dictionary structure
            self.assertIsInstance(loaded_mesh.textures, dict)
            self.assertIsInstance(loaded_mesh.material_data, dict)
            
            # Verify arrays in dictionaries
            np.testing.assert_array_almost_equal(
                loaded_mesh.textures["diffuse"], self.diffuse_texture, decimal=5
            )
            np.testing.assert_array_almost_equal(
                loaded_mesh.textures["normal"], self.normal_texture, decimal=5
            )
            np.testing.assert_array_almost_equal(
                loaded_mesh.material_data["surface"]["roughness"], self.roughness_map, decimal=5
            )
            
            # Verify non-array field
            self.assertEqual(loaded_mesh.material_name, "zip_test_material")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_empty_dict_fields(self):
        """Test handling of empty dictionary fields."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            material_name="empty_dict_test"
        )
        
        # Verify empty dictionaries don't cause issues
        self.assertEqual(len(mesh.array_fields), 2)  # Only vertices and indices
        self.assertEqual(len(mesh.dict_structure_metadata), 0)
        
        # Test encoding/decoding works with empty dictionaries
        encoded_mesh = MeshUtils.encode(mesh)
        decoded_mesh = MeshUtils.decode(TexturedMesh, encoded_mesh)
        
        self.assertIsInstance(decoded_mesh.textures, dict)
        self.assertIsInstance(decoded_mesh.material_data, dict)
        self.assertEqual(len(decoded_mesh.textures), 0)
        self.assertEqual(len(decoded_mesh.material_data), 0)


if __name__ == '__main__':
    unittest.main()