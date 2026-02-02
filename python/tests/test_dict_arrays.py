"""
Tests for dictionary fields containing numpy arrays.

This test verifies that the library can handle mesh fields that contain
dictionaries of numpy arrays, extracting them for encoding while preserving
the dictionary structure for reconstruction.
"""
import os
import tempfile
import numpy as np
import pytest
from io import BytesIO
from typing import Dict, Any
from pydantic import Field

from meshly import Mesh, Array


class TexturedMesh(Mesh):
    """A custom mesh class with dictionary fields containing numpy arrays."""

    textures: Dict[str, Array] = Field(
        default_factory=dict,
        description="Dictionary of texture arrays"
    )

    material_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Nested dictionary structure with arrays and other values"
    )

    material_name: str = Field("default", description="Material name")


class TestDictArrays:
    """Test dictionary fields containing numpy arrays."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=np.float32)

        self.indices = np.array([0, 1, 2], dtype=np.uint32)

        self.diffuse_texture = np.random.random((64, 64, 3)).astype(np.float32)
        self.normal_texture = np.random.random((64, 64, 3)).astype(np.float32)
        self.specular_texture = np.random.random((64, 64, 1)).astype(np.float32)

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

        encoded_mesh = mesh.encode()
        assert isinstance(encoded_mesh, bytes)
        assert len(encoded_mesh) > 0

        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = TexturedMesh.load_from_zip(buffer)

        assert decoded_mesh.vertex_count == mesh.vertex_count
        assert decoded_mesh.index_count == mesh.index_count
        np.testing.assert_array_almost_equal(decoded_mesh.vertices, mesh.vertices)
        np.testing.assert_array_almost_equal(decoded_mesh.indices, mesh.indices)

        assert isinstance(decoded_mesh.textures, dict)
        assert isinstance(decoded_mesh.material_data, dict)

        assert "diffuse" in decoded_mesh.textures
        assert "normal" in decoded_mesh.textures
        assert "specular" in decoded_mesh.textures

        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["diffuse"], self.diffuse_texture, decimal=5)
        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["normal"], self.normal_texture, decimal=5)
        np.testing.assert_array_almost_equal(
            decoded_mesh.textures["specular"], self.specular_texture, decimal=5)

        assert "surface" in decoded_mesh.material_data
        assert "lighting" in decoded_mesh.material_data

        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["surface"]["roughness"], self.roughness_map, decimal=5)
        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["surface"]["metallic"], self.metallic_map, decimal=5)
        np.testing.assert_array_almost_equal(
            decoded_mesh.material_data["lighting"]["emission"], self.emission_map, decimal=5)

        assert decoded_mesh.material_name == "test_material"

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

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = TexturedMesh.load_from_zip(temp_path)

            assert loaded_mesh.vertex_count == mesh.vertex_count
            assert loaded_mesh.index_count == mesh.index_count
            np.testing.assert_array_almost_equal(loaded_mesh.vertices, mesh.vertices)
            np.testing.assert_array_almost_equal(loaded_mesh.indices, mesh.indices)

            assert isinstance(loaded_mesh.textures, dict)
            assert isinstance(loaded_mesh.material_data, dict)

            np.testing.assert_array_almost_equal(
                loaded_mesh.textures["diffuse"], self.diffuse_texture, decimal=5)
            np.testing.assert_array_almost_equal(
                loaded_mesh.textures["normal"], self.normal_texture, decimal=5)
            np.testing.assert_array_almost_equal(
                loaded_mesh.material_data["surface"]["roughness"], self.roughness_map, decimal=5)

            assert loaded_mesh.material_name == "zip_test_material"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_empty_dict_fields(self):
        """Test handling of empty dictionary fields."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            material_name="empty_dict_test"
        )

        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_mesh = TexturedMesh.load_from_zip(buffer)

        assert isinstance(decoded_mesh.textures, dict)
        assert isinstance(decoded_mesh.material_data, dict)
        assert len(decoded_mesh.textures) == 0
        assert len(decoded_mesh.material_data) == 0

    def test_dict_with_non_array_values(self):
        """Test that dictionaries containing non-array values are preserved."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            textures={
                "diffuse": self.diffuse_texture,
                "normal": self.normal_texture
            },
            material_data={
                "surface": {
                    "roughness": self.roughness_map,
                    "metallic": self.metallic_map,
                    "name": "metal_surface",
                    "shininess": 0.8,
                    "enabled": True
                },
                "lighting": {
                    "emission": self.emission_map,
                    "intensity": 1.5,
                    "color": [1.0, 0.8, 0.6]
                },
                "metadata": {
                    "author": "test_user",
                    "version": 2,
                    "tags": ["metal", "shiny"]
                }
            },
            material_name="test_non_array_material"
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = TexturedMesh.load_from_zip(temp_path)

            np.testing.assert_array_almost_equal(
                loaded_mesh.material_data["surface"]["roughness"],
                self.roughness_map, decimal=5)
            np.testing.assert_array_almost_equal(
                loaded_mesh.material_data["lighting"]["emission"],
                self.emission_map, decimal=5)

            assert loaded_mesh.material_data["surface"]["name"] == "metal_surface"
            assert loaded_mesh.material_data["surface"]["shininess"] == 0.8
            assert loaded_mesh.material_data["surface"]["enabled"] is True
            assert loaded_mesh.material_data["lighting"]["intensity"] == 1.5
            assert loaded_mesh.material_data["lighting"]["color"] == [1.0, 0.8, 0.6]

            assert "metadata" in loaded_mesh.material_data
            assert loaded_mesh.material_data["metadata"]["author"] == "test_user"
            assert loaded_mesh.material_data["metadata"]["version"] == 2
            assert loaded_mesh.material_data["metadata"]["tags"] == ["metal", "shiny"]

            assert loaded_mesh.material_name == "test_non_array_material"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_dict_with_non_array_values_zip_round_trip(self):
        """Test that non-array dict values survive zip save/load."""
        mesh = TexturedMesh(
            vertices=self.vertices,
            indices=self.indices,
            material_data={
                "config": {
                    "name": "test_config",
                    "version": 3.14,
                    "settings": {
                        "quality": "high",
                        "enabled": True,
                        "options": [1, 2, 3]
                    }
                }
            },
            material_name="zip_non_array_test"
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = TexturedMesh.load_from_zip(temp_path)

            assert loaded_mesh.material_data["config"]["name"] == "test_config"
            assert loaded_mesh.material_data["config"]["version"] == 3.14
            assert loaded_mesh.material_data["config"]["settings"]["quality"] == "high"
            assert loaded_mesh.material_data["config"]["settings"]["enabled"] is True
            assert loaded_mesh.material_data["config"]["settings"]["options"] == [1, 2, 3]
            assert loaded_mesh.material_name == "zip_non_array_test"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
