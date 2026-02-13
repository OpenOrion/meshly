"""Tests for ResourceRef and Resource field type."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from meshly import Array, Packable, Resource
from meshly.packable import ExtractedPackable
from meshly.utils.dynamic_model import LazyModel


def test_resource_ref_basic():
    """Test creating ResourceRef from data and ext."""
    ref = Resource(data=b"Hello World", ext=".txt")
    assert ref.data == b"Hello World"
    assert ref.ext == ".txt"
    assert ref.checksum is not None


def test_resource_ref_from_path():
    """Test creating ResourceRef from file path."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Hello World")
        temp_path = f.name

    try:
        ref = Resource.from_path(temp_path)
        assert ref.data == b"Hello World"
        assert ref.ext == ".txt"
        assert ref.checksum is not None
    finally:
        Path(temp_path).unlink()


def test_resource_ref_checksum():
    """Test that checksum is computed from data."""
    ref1 = Resource(data=b"same data", ext=".txt")
    ref2 = Resource(data=b"same data", ext=".bin")
    ref3 = Resource(data=b"different data", ext=".txt")
    
    # Same data = same checksum, regardless of ext
    assert ref1.checksum == ref2.checksum
    # Different data = different checksum
    assert ref1.checksum != ref3.checksum


def test_packable_extract_with_resource():
    """Test that extract() handles ResourceRef correctly."""

    class SimulationCase(Packable):
        name: str
        geometry: Resource

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"STL geometry data")
        temp_path = f.name

    try:
        case = SimulationCase(name="test", geometry=Resource.from_path(temp_path))

        # Extract should convert ResourceRef to $ref with checksum
        extracted = case.extract()

        assert extracted.data["name"] == "test"
        assert "$ref" in extracted.data["geometry"]
        assert "ext" in extracted.data["geometry"]
        assert extracted.data["geometry"]["ext"] == ".stl"

        # Should have the gzip-compressed file data in assets
        checksum = extracted.data["geometry"]["$ref"]
        assert checksum in extracted.assets
        # Data is gzip compressed
        assert gzip.decompress(extracted.assets[checksum]) == b"STL geometry data"
    finally:
        Path(temp_path).unlink()


def test_packable_reconstruct_with_resource():
    """Test that Packable.reconstruct() handles ResourceRef correctly."""

    class SimulationCase(Packable):
        name: str
        geometry: Resource

    # Simulate serialized data
    data = {"name": "test", "geometry": {"$ref": "abc123", "ext": ".stl"}}
    assets = {"abc123": gzip.compress(b"STL geometry data")}
    extracted = ExtractedPackable(data=data, assets=assets)

    # Reconstruct should create ResourceRef with data
    case = SimulationCase.reconstruct(extracted)

    assert case.name == "test"
    assert isinstance(case.geometry, Resource)
    assert case.geometry.data == b"STL geometry data"
    assert case.geometry.ext == ".stl"


def test_resource_round_trip():
    """Test full extract/reconstruct round trip with Resource fields."""

    class SimulationCase(Packable):
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
        case1 = SimulationCase(
            name="wind_tunnel", 
            geometry=Resource.from_path(geom_path), 
            config=Resource.from_path(config_path)
        )

        # Extract
        extracted = case1.extract()

        # Verify extraction
        assert extracted.data["name"] == "wind_tunnel"
        assert "$ref" in extracted.data["geometry"]
        assert "$ref" in extracted.data["config"]
        assert len(extracted.assets) == 2

        # Reconstruct
        case2 = SimulationCase.reconstruct(extracted)

        # Verify reconstruction
        assert case2.name == "wind_tunnel"
        assert isinstance(case2.geometry, Resource)
        assert isinstance(case2.config, Resource)
        assert case2.geometry.data == b"STL data"
        assert case2.config.data == b'{"setting": "value"}'
        assert case2.geometry.ext == ".stl"
        assert case2.config.ext == ".json"
    finally:
        Path(geom_path).unlink()
        Path(config_path).unlink()


def test_lazy_model_with_resource_ref():
    """Test LazyModel resolves ResourceRef fields on access."""

    class SimulationCase(Packable):
        name: str
        geometry: Resource

    # Simulate serialized data with ResourceRef
    data = {"name": "test_case", "geometry": {"$ref": "abc123", "ext": ".stl"}}
    assets = {"abc123": gzip.compress(b"STL geometry data")}
    extracted = ExtractedPackable(data=data, assets=assets)

    # Track which assets are requested
    requested = []

    def tracking_loader(checksum: str) -> bytes:
        requested.append(checksum)
        return assets[checksum]

    # Reconstruct with is_lazy=True - returns LazyModel
    lazy = SimulationCase.reconstruct(extracted, assets=tracking_loader, is_lazy=True)

    assert isinstance(lazy, LazyModel)
    assert len(requested) == 0, "No assets should be loaded yet"

    # Access primitive field - no loading
    assert lazy.name == "test_case"
    assert len(requested) == 0, "Primitive fields shouldn't trigger loading"

    # Access ResourceRef field - triggers asset fetch
    geometry = lazy.geometry
    assert isinstance(geometry, Resource)
    assert geometry.data == b"STL geometry data"
    assert geometry.ext == ".stl"
    assert len(requested) == 1, "ResourceRef access fetches asset"


def test_lazy_model_resolve_with_resource_ref():
    """Test model with ResourceRef fields using eager loading."""

    class SimulationCase(Packable):
        name: str
        geometry: Resource
        config: Resource

    data = {
        "name": "wind_tunnel",
        "geometry": {"$ref": "geo123", "ext": ".stl"},
        "config": {"$ref": "cfg456", "ext": ".json"},
    }

    assets = {
        "geo123": gzip.compress(b"geometry data"), 
        "cfg456": gzip.compress(b"config data")
    }
    extracted = ExtractedPackable(data=data, assets=assets)

    # With dict assets (not callable), reconstruct returns the actual model
    result = SimulationCase.reconstruct(extracted)
    assert isinstance(result, SimulationCase)

    # Verify the model
    assert result.name == "wind_tunnel"
    assert isinstance(result.geometry, Resource)
    assert isinstance(result.config, Resource)
    assert result.geometry.data == b"geometry data"
    assert result.config.data == b"config data"


def test_resource_ref_in_nested_dict_eager_loading():
    """Test ResourceRef in nested dictionary with eager loading."""

    class ExperimentCase(Packable):
        name: str
        files: dict[str, Resource]

    data = {
        "name": "exp1",
        "files": {
            "mesh": {"$ref": "mesh123", "ext": ".msh"},
            "texture": {"$ref": "tex456", "ext": ".png"},
        },
    }

    assets = {
        "mesh123": gzip.compress(b"mesh data"), 
        "tex456": gzip.compress(b"texture data")
    }
    extracted = ExtractedPackable(data=data, assets=assets)

    # Use dict assets for eager loading
    result = ExperimentCase.reconstruct(extracted)

    # Access nested dict with ResourceRefs
    files = result.files
    assert isinstance(files, dict)
    assert "mesh" in files
    assert "texture" in files

    # Both should be ResourceRef instances
    assert isinstance(files["mesh"], Resource)
    assert isinstance(files["texture"], Resource)
    assert files["mesh"].data == b"mesh data"
    assert files["texture"].data == b"texture data"


def test_resource_ref_in_list_eager_loading():
    """Test ResourceRef in list with eager loading."""

    class BatchJob(Packable):
        name: str
        input_files: list[Resource]

    data = {
        "name": "batch1",
        "input_files": [
            {"$ref": "file1", "ext": ".dat"}, 
            {"$ref": "file2", "ext": ".dat"}
        ],
    }

    assets = {
        "file1": gzip.compress(b"data1"), 
        "file2": gzip.compress(b"data2")
    }
    extracted = ExtractedPackable(data=data, assets=assets)

    # Use dict assets for eager loading
    result = BatchJob.reconstruct(extracted)

    # Access list of ResourceRefs
    files = result.input_files
    assert isinstance(files, list)
    assert len(files) == 2

    assert isinstance(files[0], Resource)
    assert isinstance(files[1], Resource)
    assert files[0].data == b"data1"
    assert files[1].data == b"data2"


def test_mixed_resource_and_array_lazy_loading():
    """Test LazyModel with both ResourceRef and array fields."""

    class Simulation(Packable):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        geometry: Resource
        initial_conditions: Array

    # Create original data
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        f.write(b"geometry bytes")
        temp_path = f.name

    try:
        original = Simulation(
            name="sim1",
            geometry=Resource.from_path(temp_path),
            initial_conditions=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        # Extract
        extracted = original.extract()

        # Track asset requests
        requested = []

        def tracking_loader(checksum: str) -> bytes:
            requested.append(checksum)
            return extracted.assets[checksum]

        # Reconstruct with is_lazy=True
        lazy = Simulation.reconstruct(extracted, assets=tracking_loader, is_lazy=True)

        assert isinstance(lazy, LazyModel)
        assert len(requested) == 0

        # Access array - loads array asset
        initial = lazy.initial_conditions
        assert isinstance(initial, np.ndarray)
        assert len(requested) == 1  # Array asset loaded
        np.testing.assert_array_equal(initial, [1.0, 2.0, 3.0])

        # Access ResourceRef - loads resource asset
        geometry = lazy.geometry
        assert isinstance(geometry, Resource)
        assert geometry.data == b"geometry bytes"
        assert len(requested) == 2  # Now resource asset also loaded
    finally:
        Path(temp_path).unlink()
