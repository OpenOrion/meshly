"""Tests for ResourceRef and Resource field type."""

import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from meshly import Packable, ResourceRef


def test_resource_ref_from_path():
    """Test creating ResourceRef from file path."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Hello World")
        temp_path = f.name

    try:
        ref = ResourceRef(path=temp_path, ext=".txt")
        assert ref.path == temp_path
        assert ref.ext == ".txt"
        assert ref.read_bytes() == b"Hello World"
        assert ref.resolve_path() == Path(temp_path)
    finally:
        Path(temp_path).unlink()


def test_resource_ref_from_checksum():
    """Test creating ResourceRef from dict with $ref (simulating deserialization)."""
    # ResourceRef is typically created from $ref dict during deserialization
    ref = ResourceRef(**{"$ref": "abc123", "ext": ".stl"})
    assert ref.checksum == "abc123"
    assert ref.ext == ".stl"
    assert str(ref) == "abc123"
    assert "checksum='abc123'" in repr(ref)


def test_resource_field_validation():
    """Test Resource field type validation."""

    class TestModel(BaseModel):
        geometry: ResourceRef

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"STL data")
        temp_path = f.name

    try:
        # Create from string path
        model = TestModel(geometry=temp_path)
        assert isinstance(model.geometry, ResourceRef)
        assert model.geometry.path == temp_path
        assert model.geometry.ext == ".stl"
        assert model.geometry.read_bytes() == b"STL data"

        # Create from Path
        model2 = TestModel(geometry=Path(temp_path))
        assert isinstance(model2.geometry, ResourceRef)
        assert model2.geometry.read_bytes() == b"STL data"

        # Create from dict with $ref
        model3 = TestModel(geometry={"$ref": "xyz789", "ext": ".stl"})
        assert isinstance(model3.geometry, ResourceRef)
        assert model3.geometry.checksum == "xyz789"
        assert model3.geometry.ext == ".stl"
    finally:
        Path(temp_path).unlink()


def test_packable_extract_with_resource():
    """Test that Packable.extract() handles ResourceRef correctly."""
    import gzip

    class SimulationCase(BaseModel):
        name: str
        geometry: ResourceRef

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"STL geometry data")
        temp_path = f.name

    try:
        case = SimulationCase(name="test", geometry=temp_path)

        # Extract should convert ResourceRef to $ref with checksum
        extracted = Packable.extract(case)

        assert extracted.data["name"] == "test"
        assert "$ref" in extracted.data["geometry"]
        # name contains the filename (from which ext can be derived)
        assert "name" in extracted.data["geometry"]
        assert extracted.data["geometry"]["name"].endswith(".stl")

        # Should have the gzip-compressed file data in assets
        checksum = extracted.data["geometry"]["$ref"]
        assert checksum in extracted.assets
        # Data is gzip compressed
        assert gzip.decompress(extracted.assets[checksum]) == b"STL geometry data"
    finally:
        Path(temp_path).unlink()


def test_packable_reconstruct_with_resource():
    """Test that Packable.reconstruct() handles ResourceRef correctly."""

    class SimulationCase(BaseModel):
        name: str
        geometry: ResourceRef

    # Simulate serialized data - name contains the filename
    data = {"name": "test", "geometry": {"$ref": "abc123", "name": "model.stl"}}
    assets = {"abc123": b"STL geometry data"}

    # Reconstruct should create ResourceRef with checksum
    case = Packable.reconstruct(SimulationCase, data, assets)

    assert case.name == "test"
    assert isinstance(case.geometry, ResourceRef)
    assert case.geometry.checksum == "abc123"


def test_resource_round_trip():
    """Test full extract/reconstruct round trip with Resource fields."""

    class SimulationCase(BaseModel):
        name: str
        geometry: ResourceRef
        config: ResourceRef

    with (
        tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f1,
        tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f2,
    ):
        f1.write(b"STL data")
        f2.write(b'{"setting": "value"}')
        geom_path = f1.name
        config_path = f2.name

    try:
        # Original case
        case1 = SimulationCase(name="wind_tunnel", geometry=geom_path, config=config_path)

        # Extract
        extracted = Packable.extract(case1)

        # Verify extraction
        assert extracted.data["name"] == "wind_tunnel"
        assert "$ref" in extracted.data["geometry"]
        assert "$ref" in extracted.data["config"]
        assert len(extracted.assets) == 2

        # Reconstruct
        case2 = Packable.reconstruct(SimulationCase, extracted.data, extracted.assets)

        # Verify reconstruction
        assert case2.name == "wind_tunnel"
        assert isinstance(case2.geometry, ResourceRef)
        assert isinstance(case2.config, ResourceRef)
        assert case2.geometry.checksum is not None
        assert case2.config.checksum is not None
        # Extensions can be derived from the name field in the serialized data
    finally:
        Path(geom_path).unlink()
        Path(config_path).unlink()


def test_lazy_model_with_resource_ref():
    """Test LazyDynamicModel resolves ResourceRef fields on access."""
    from meshly.utils.dynamic_model import LazyDynamicModel

    class SimulationCase(BaseModel):
        name: str
        geometry: ResourceRef

    # Simulate serialized data with ResourceRef - use name instead of ext
    data = {"name": "test_case", "geometry": {"$ref": "abc123", "name": "model.stl"}}

    assets = {"abc123": b"STL geometry data"}

    # Track which assets are requested
    requested = []

    def tracking_loader(checksum: str) -> bytes:
        requested.append(checksum)
        return assets[checksum]

    # Reconstruct with is_lazy=True - returns LazyDynamicModel
    lazy = Packable.reconstruct(SimulationCase, data, tracking_loader, is_lazy=True)

    assert isinstance(lazy, LazyDynamicModel)
    assert len(requested) == 0, "No assets should be loaded yet"

    # Access primitive field - no loading
    assert lazy.name == "test_case"
    assert len(requested) == 0, "Primitive fields shouldn't trigger loading"

    # Access ResourceRef field - triggers asset fetch with schema-based resolution
    geometry = lazy.geometry
    assert isinstance(geometry, dict)  # Returns resolved dict with _bytes
    assert geometry["$ref"] == "abc123"
    assert len(requested) == 1, "ResourceRef access fetches asset"


def test_lazy_model_resolve_with_resource_ref():
    """Test model with ResourceRef fields using eager loading."""

    class SimulationCase(BaseModel):
        name: str
        geometry: ResourceRef
        config: ResourceRef

    data = {
        "name": "wind_tunnel",
        "geometry": {"$ref": "geo123", "name": "model.stl"},
        "config": {"$ref": "cfg456", "name": "config.json"},
    }

    assets = {"geo123": b"geometry data", "cfg456": b"config data"}

    # With dict assets (not callable), reconstruct returns the actual model, not LazyDynamicModel
    result = Packable.reconstruct(SimulationCase, data, assets)
    assert isinstance(result, SimulationCase)  # Not LazyDynamicModel when assets is a dict

    # Verify the model
    assert result.name == "wind_tunnel"
    assert isinstance(result.geometry, ResourceRef)
    assert isinstance(result.config, ResourceRef)
    assert result.geometry.checksum == "geo123"
    assert result.config.checksum == "cfg456"


def test_resource_ref_in_nested_dict_eager_loading():
    """Test ResourceRef in nested dictionary with eager loading."""

    class ExperimentCase(BaseModel):
        name: str
        files: dict[str, ResourceRef]

    data = {
        "name": "exp1",
        "files": {
            "mesh": {"$ref": "mesh123", "name": "mesh.msh"},
            "texture": {"$ref": "tex456", "name": "texture.png"},
        },
    }

    assets = {"mesh123": b"mesh data", "tex456": b"texture data"}

    # Use dict assets for eager loading
    result = Packable.reconstruct(ExperimentCase, data, assets)

    # Access nested dict with ResourceRefs
    files = result.files
    assert isinstance(files, dict)
    assert "mesh" in files
    assert "texture" in files

    # Both should be ResourceRef instances
    assert isinstance(files["mesh"], ResourceRef)
    assert isinstance(files["texture"], ResourceRef)
    assert files["mesh"].checksum == "mesh123"
    assert files["texture"].checksum == "tex456"


def test_resource_ref_in_list_eager_loading():
    """Test ResourceRef in list with eager loading."""

    class BatchJob(BaseModel):
        name: str
        input_files: list[ResourceRef]

    data = {
        "name": "batch1",
        "input_files": [{"$ref": "file1", "name": "input1.dat"}, {"$ref": "file2", "name": "input2.dat"}],
    }

    assets = {"file1": b"data1", "file2": b"data2"}

    # Use dict assets for eager loading
    result = Packable.reconstruct(BatchJob, data, assets)

    # Access list of ResourceRefs
    files = result.input_files
    assert isinstance(files, list)
    assert len(files) == 2

    assert isinstance(files[0], ResourceRef)
    assert isinstance(files[1], ResourceRef)
    assert files[0].checksum == "file1"
    assert files[1].checksum == "file2"


def test_mixed_resource_and_array_lazy_loading():
    """Test LazyDynamicModel with both ResourceRef and array fields."""
    import numpy as np
    from meshly.utils.dynamic_model import LazyDynamicModel
    from meshly import Array

    class Simulation(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        geometry: ResourceRef
        initial_conditions: Array

    # Create original data
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"geometry bytes")
        temp_path = f.name

    try:
        original = Simulation(
            name="sim1",
            geometry=temp_path,
            initial_conditions=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        # Extract
        extracted = Packable.extract(original)

        # Track asset requests
        requested = []

        def tracking_loader(checksum: str) -> bytes:
            requested.append(checksum)
            return extracted.assets[checksum]

        # Reconstruct with is_lazy=True
        lazy = Packable.reconstruct(Simulation, extracted.data, tracking_loader, is_lazy=True)

        assert isinstance(lazy, LazyDynamicModel)
        assert len(requested) == 0

        # Access array - loads array asset
        initial = lazy.initial_conditions
        assert isinstance(initial, np.ndarray)
        assert len(requested) == 1  # Array asset loaded
        np.testing.assert_array_equal(initial, [1.0, 2.0, 3.0])

        # Access ResourceRef - loads resource asset
        geometry = lazy.geometry
        assert isinstance(geometry, dict)
        assert geometry["$ref"] is not None
        assert len(requested) == 2  # Now resource asset also loaded
    finally:
        Path(temp_path).unlink()
