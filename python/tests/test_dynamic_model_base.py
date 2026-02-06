"""Test that DynamicModelBuilder respects x-base hints for base class selection."""

import numpy as np
import pytest
from meshly import Mesh, Packable
from meshly.utils.dynamic_model import DynamicModelBuilder
from meshly.utils.json_schema import JsonSchema
from pydantic import BaseModel


def test_dynamic_model_base_mesh():
    """Test that x-base: 'Mesh' creates a model inheriting from Mesh."""
    schema_dict = {
        "title": "CustomMesh",
        "type": "object",
        "x-base": "Mesh",
        "properties": {
            "vertices": {
                "type": "vertex_buffer",
                "description": "Vertex positions"
            },
            "indices": {
                "type": "index_sequence",
                "description": "Triangle indices"
            },
            "custom_field": {
                "type": "string",
                "description": "Custom field"
            }
        },
        "required": ["vertices"]
    }
    
    schema = JsonSchema.model_validate(schema_dict)
    ModelClass = DynamicModelBuilder.build_model(schema, "CustomMesh")
    
    # Verify it inherits from Mesh
    assert issubclass(ModelClass, Mesh)
    assert issubclass(ModelClass, Packable)
    assert issubclass(ModelClass, BaseModel)
    
    # Verify it has Mesh methods
    assert hasattr(ModelClass, 'triangulate')
    assert hasattr(ModelClass, 'optimize_vertex_cache')
    assert hasattr(ModelClass, 'to_pyvista')
    
    # Create an instance and verify it works
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    instance = ModelClass(vertices=vertices, indices=indices, custom_field="test")
    
    assert isinstance(instance, Mesh)
    assert instance.vertex_count == 3
    assert instance.custom_field == "test"
    
    # Verify mesh operations work
    triangulated = instance.triangulate()
    assert isinstance(triangulated, Mesh)


def test_dynamic_model_base_packable():
    """Test that x-base: 'Packable' creates a model inheriting from Packable."""
    schema_dict = {
        "title": "CustomPackable",
        "type": "object",
        "x-base": "Packable",
        "properties": {
            "value": {
                "type": "number",
                "description": "A numerical value"
            },
            "name": {
                "type": "string",
                "description": "Name field"
            }
        },
        "required": ["name"]
    }
    
    schema = JsonSchema.model_validate(schema_dict)
    ModelClass = DynamicModelBuilder.build_model(schema, "CustomPackable")
    
    # Verify it inherits from Packable
    assert issubclass(ModelClass, Packable)
    assert issubclass(ModelClass, BaseModel)
    
    # Verify it has Packable methods
    assert hasattr(ModelClass, 'extract')
    assert hasattr(ModelClass, 'save_to_zip')
    assert hasattr(ModelClass, 'load_from_zip')
    
    # Create an instance and verify it works
    instance = ModelClass(value=42.0, name="test")
    
    assert isinstance(instance, Packable)
    assert instance.name == "test"
    assert instance.value == 42.0
    
    # Verify encode/decode works (zip format)
    encoded = instance.encode()
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    
    decoded = ModelClass.decode(encoded)
    assert decoded.name == "test"
    assert decoded.value == 42.0


def test_dynamic_model_base_default():
    """Test that no x-base or unrecognized x-base creates a BaseModel."""
    schema_dict = {
        "title": "PlainModel",
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
                "description": "A value"
            },
            "name": {
                "type": "string"
            }
        },
        "required": ["value"]
    }
    
    schema = JsonSchema.model_validate(schema_dict)
    ModelClass = DynamicModelBuilder.build_model(schema, "PlainModel")
    
    # Verify it inherits from BaseModel but not Packable or Mesh
    assert issubclass(ModelClass, BaseModel)
    assert not issubclass(ModelClass, Packable)
    assert not issubclass(ModelClass, Mesh)
    
    # Create an instance
    instance = ModelClass(value=42, name="test")
    assert instance.value == 42
    assert instance.name == "test"


def test_dynamic_model_instantiate_with_mesh_base():
    """Test instantiating a dynamic mesh model with data."""
    schema_dict = {
        "title": "DynamicMesh",
        "type": "object",
        "x-base": "Mesh",
        "properties": {
            "vertices": {"type": "vertex_buffer"},
            "indices": {"type": "index_sequence"},
            "label": {"type": "string"}
        },
        "required": ["vertices"]
    }
    
    schema = JsonSchema.model_validate(schema_dict)
    
    # Create test data
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    
    # Instantiate directly (without $refs, so no assets needed)
    instance = DynamicModelBuilder.instantiate(
        schema,
        {
            "vertices": vertices,
            "indices": indices,
            "label": "triangle"
        },
        assets={},
    )
    
    assert isinstance(instance, Mesh)
    assert instance.vertex_count == 3
    assert instance.label == "triangle"
    assert hasattr(instance, 'triangulate')


def test_dynamic_model_cache_respects_xbase():
    """Test that models with different x-base values are cached separately."""
    DynamicModelBuilder.clear_cache()
    
    # Create two schemas with same fields but different x-base
    schema_mesh = JsonSchema.model_validate({
        "title": "TestModel",
        "type": "object",
        "x-base": "Mesh",
        "properties": {"vertices": {"type": "vertex_buffer"}},
        "required": ["vertices"]
    })
    
    schema_packable = JsonSchema.model_validate({
        "title": "TestModel",
        "type": "object",
        "x-base": "Packable",
        "properties": {"vertices": {"type": "array"}},
        "required": ["vertices"]
    })
    
    ModelMesh = DynamicModelBuilder.build_model(schema_mesh)
    ModelPackable = DynamicModelBuilder.build_model(schema_packable)
    
    # They should be different classes
    assert ModelMesh is not ModelPackable
    assert issubclass(ModelMesh, Mesh)
    assert issubclass(ModelPackable, Packable)
    assert not issubclass(ModelPackable, Mesh)


def test_mesh_subclass_has_combine_method():
    """Test that dynamically created Mesh subclass inherits combine method."""
    schema_dict = {
        "title": "CustomMesh",
        "type": "object",
        "x-base": "Mesh",
        "properties": {
            "vertices": {"type": "vertex_buffer"},
            "indices": {"type": "index_sequence"},
            "temperature": {"type": "number"}
        },
        "required": ["vertices"]
    }
    
    schema = JsonSchema.model_validate(schema_dict)
    CustomMesh = DynamicModelBuilder.build_model(schema)
    
    # Create test meshes
    mesh1 = CustomMesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        temperature=20.0
    )
    
    mesh2 = CustomMesh(
        vertices=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
        indices=np.array([0, 1, 2], dtype=np.uint32),
        temperature=25.0
    )
    
    # Combine should work and preserve extra fields
    combined = CustomMesh.combine([mesh1, mesh2])
    
    assert isinstance(combined, CustomMesh)
    assert combined.vertex_count == 6
    assert combined.temperature == 20.0  # From first mesh
    assert hasattr(combined, 'triangulate')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
