"""Tests for ResourceRef and Resource field type."""

import tempfile
from pathlib import Path

import pytest
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
        geometry: Resource

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

    class SimulationCase(BaseModel):
        name: str
        geometry: Resource

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"STL geometry data")
        temp_path = f.name

    try:
        case = SimulationCase(name="test", geometry=temp_path)

        # Extract should convert ResourceRef to $ref with checksum
        extracted = Packable.extract(case)

        assert extracted.data["name"] == "test"
        assert "$ref" in extracted.data["geometry"]
        assert "ext" in extracted.data["geometry"]
        assert extracted.data["geometry"]["ext"] == ".stl"

        # Should have the file data in assets
        checksum = extracted.data["geometry"]["$ref"]
        assert checksum in extracted.assets
        assert extracted.assets[checksum] == b"STL geometry data"
    finally:
        Path(temp_path).unlink()


def test_packable_reconstruct_with_resource():
    """Test that Packable.reconstruct() handles ResourceRef correctly."""

    class SimulationCase(BaseModel):
        name: str
        geometry: Resource

    # Simulate serialized data
    data = {"name": "test", "geometry": {"$ref": "abc123", "ext": ".stl"}}
    assets = {"abc123": b"STL geometry data"}

    # Reconstruct should create ResourceRef with checksum
    case = Packable.reconstruct(SimulationCase, data, assets)

    assert case.name == "test"
    assert isinstance(case.geometry, ResourceRef)
    assert case.geometry.checksum == "abc123"
    assert case.geometry.ext == ".stl"


def test_resource_round_trip():
    """Test full extract/reconstruct round trip with Resource fields."""

    class SimulationCase(BaseModel):
        name: str
        geometry: Resource
        config: Resource

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
        assert case2.geometry.ext == ".stl"
        assert case2.config.ext == ".json"
    finally:
        Path(geom_path).unlink()
        Path(config_path).unlink()


def test_lazy_model_with_resource_ref():
    """Test LazyModel defers loading ResourceRef assets."""
    from meshly.packable import LazyModel

    class SimulationCase(BaseModel):
        name: str
        geometry: Resource

    # Simulate serialized data with ResourceRef
    data = {"name": "test_case", "geometry": {"$ref": "abc123", "ext": ".stl"}}

    assets = {"abc123": b"STL geometry data"}

    # Track which assets are requested
    requested = []

    def tracking_loader(checksum: str) -> bytes:
        requested.append(checksum)
        return assets[checksum]

    # Reconstruct with callable - returns LazyModel
    lazy = Packable.reconstruct(SimulationCase, data, tracking_loader)

    assert isinstance(lazy, LazyModel)
    assert len(requested) == 0, "No assets should be loaded yet"

    # Access primitive field - no loading
    assert lazy.name == "test_case"
    assert len(requested) == 0, "Primitive fields shouldn't trigger loading"

    # Access ResourceRef field - LazyModel returns the dict directly
    # The dict will be passed to ResourceRef validator when the model is resolved
    geometry = lazy.geometry
    assert isinstance(geometry, dict)  # LazyModel returns dict, not ResourceRef instance
    assert geometry["$ref"] == "abc123"
    assert geometry["ext"] == ".stl"
    assert len(requested) == 0, "ResourceRef dict doesn't trigger loading"

    # Resolve to get actual model with ResourceRef instances
    resolved = lazy.resolve()
    assert isinstance(resolved.geometry, ResourceRef)
    assert resolved.geometry.checksum == "abc123"
    assert resolved.geometry.ext == ".stl"


def test_lazy_model_resolve_with_resource_ref():
    """Test model with ResourceRef fields using eager loading."""

    class SimulationCase(BaseModel):
        name: str
        geometry: Resource
        config: Resource

    data = {
        "name": "wind_tunnel",
        "geometry": {"$ref": "geo123", "ext": ".stl"},
        "config": {"$ref": "cfg456", "ext": ".json"},
    }

    assets = {"geo123": b"geometry data", "cfg456": b"config data"}

    # With dict assets (not callable), reconstruct returns the actual model, not LazyModel
    result = Packable.reconstruct(SimulationCase, data, assets)
    assert isinstance(result, SimulationCase)  # Not LazyModel when assets is a dict

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
        files: dict[str, Resource]

    data = {
        "name": "exp1",
        "files": {
            "mesh": {"$ref": "mesh123", "ext": ".msh"},
            "texture": {"$ref": "tex456", "ext": ".png"},
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
    assert files["mesh"].ext == ".msh"
    assert files["texture"].ext == ".png"


def test_resource_ref_in_list_eager_loading():
    """Test ResourceRef in list with eager loading."""

    class BatchJob(BaseModel):
        name: str
        input_files: list[Resource]

    data = {
        "name": "batch1",
        "input_files": [{"$ref": "file1", "ext": ".dat"}, {"$ref": "file2", "ext": ".dat"}],
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
    """Test LazyModel with both ResourceRef and array fields."""
    import numpy as np
    from meshly.packable import LazyModel

    class Simulation(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        geometry: Resource
        initial_conditions: np.ndarray

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

        # Reconstruct with lazy loading
        lazy = Packable.reconstruct(Simulation, extracted.data, tracking_loader)

        assert isinstance(lazy, LazyModel)
        assert len(requested) == 0

        # Access ResourceRef - LazyModel returns dict
        geometry = lazy.geometry
        assert isinstance(geometry, dict)
        assert geometry["$ref"] is not None
        assert len(requested) == 0

        # Access array - loads asset
        initial = lazy.initial_conditions
        assert isinstance(initial, np.ndarray)
        assert len(requested) == 1  # Only the array asset is loaded
        np.testing.assert_array_equal(initial, [1.0, 2.0, 3.0])

        # Resolve to get actual model with ResourceRef
        resolved = lazy.resolve()
        assert isinstance(resolved.geometry, ResourceRef)
    finally:
        Path(temp_path).unlink()


def test_cached_asset_loader_with_resource_ref():
    """Test CachedAssetLoader works with ResourceRef fields."""
    from meshly.data_handler import CachedAssetLoader, DataHandler
    from meshly.packable import LazyModel

    class DataPackage(BaseModel):
        name: str
        model_file: Resource

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f,
    ):
        f.write(b"3D model data")
        model_path = f.name

        try:
            # Create original
            pkg = DataPackage(name="package1", model_file=model_path)

            # Extract
            extracted = Packable.extract(pkg)

            # Setup cache
            cache_handler = DataHandler.create(Path(tmpdir) / "cache")

            fetch_count = [0]

            def fetch_with_counter(checksum: str) -> bytes:
                fetch_count[0] += 1
                return extracted.assets[checksum]

            loader = CachedAssetLoader(fetch_with_counter, cache_handler)

            # First reconstruction with CachedAssetLoader returns LazyModel
            lazy1 = Packable.reconstruct(DataPackage, extracted.data, loader)
            assert isinstance(lazy1, LazyModel)

            # Access the ResourceRef field - LazyModel returns dict
            model1 = lazy1.model_file
            assert isinstance(model1, dict)
            assert model1["$ref"] is not None
            assert fetch_count[0] == 0  # ResourceRef dict doesn't trigger fetch

            # Resolve to get actual model with ResourceRef
            resolved = lazy1.resolve()
            assert isinstance(resolved.model_file, ResourceRef)

            # Second reconstruction - same behavior
            lazy2 = Packable.reconstruct(DataPackage, extracted.data, loader)
            model2 = lazy2.model_file

            assert isinstance(model2, dict)
            assert fetch_count[0] == 0  # Still no fetch needed
        finally:
            Path(model_path).unlink()
