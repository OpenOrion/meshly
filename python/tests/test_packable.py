"""Tests for Packable base class."""

import pytest
import tempfile
import os
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from meshly.packable import Packable
from meshly.array import Array


class SimpleData(Packable):
    """Simple test data container."""
    name: str = Field(..., description="Name")
    values: Array = Field(..., description="Values array")


class SimulationResult(Packable):
    """Simulation result with multiple arrays."""
    time: float = Field(..., description="Time value")
    temperature: Array = Field(..., description="Temperature field")
    velocity: Array = Field(default=None, description="Velocity field")


class NestedData(Packable):
    """Data container with nested dictionary arrays."""
    label: str = Field(..., description="Label")
    fields: dict = Field(default_factory=dict, description="Named fields")


class FieldData(BaseModel):
    """Generic field data container for testing dict[str, BaseModel] edge case."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (scalar, vector, tensor)")
    data: Array = Field(..., description="Field data array")
    units: Optional[str] = Field(None, description="Field units")


class Snapshot(Packable):
    """Snapshot with dict of BaseModel containing arrays."""
    time: float = Field(..., description="Time value")
    fields: dict[str, FieldData] = Field(
        default_factory=dict, description="Field data")


class TestPackable:
    """Test Packable functionality."""

    def test_simple_container(self):
        """Test basic Packable creation."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        assert data.name == "test"
        np.testing.assert_array_equal(data.values, [1.0, 2.0, 3.0])

    def test_array_fields_detection(self):
        """Test automatic array field detection."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0])
        )
        assert "values" in data.array_fields
        assert "name" not in data.array_fields

    def test_encode_decode(self):
        """Test encoding and decoding via zip round-trip."""
        original = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

        encoded = original.encode()
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        buffer = BytesIO()
        original.save_to_zip(buffer)
        buffer.seek(0)
        decoded = SimpleData.load_from_zip(buffer)
        np.testing.assert_array_almost_equal(decoded.values, original.values)
        assert decoded.name == original.name

    def test_save_load_zip_file(self):
        """Test saving and loading from a zip file."""
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.zip")
            original.save_to_zip(path)

            loaded = SimulationResult.load_from_zip(path)

            assert loaded.time == pytest.approx(original.time)
            np.testing.assert_array_almost_equal(
                loaded.temperature, original.temperature)
            np.testing.assert_array_almost_equal(
                loaded.velocity, original.velocity)

    def test_save_load_bytesio(self):
        """Test saving and loading from BytesIO."""
        original = SimpleData(
            name="bytesio_test",
            values=np.array([10, 20, 30], dtype=np.int32)
        )

        buffer = BytesIO()
        original.save_to_zip(buffer)

        buffer.seek(0)
        loaded = SimpleData.load_from_zip(buffer)

        assert loaded.name == original.name
        np.testing.assert_array_equal(loaded.values, original.values)

    def test_nested_dict_arrays(self):
        """Test containers with nested dictionary arrays."""
        data = NestedData(
            label="nested_test",
            fields={
                "pressure": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "density": np.array([0.5, 0.6, 0.7], dtype=np.float32)
            }
        )

        array_fields = data.array_fields
        assert "fields.pressure" in array_fields
        assert "fields.density" in array_fields

        buffer = BytesIO()
        data.save_to_zip(buffer)

        buffer.seek(0)
        loaded = NestedData.load_from_zip(buffer)

        assert loaded.label == data.label
        np.testing.assert_array_almost_equal(
            loaded.fields["pressure"], data.fields["pressure"])
        np.testing.assert_array_almost_equal(
            loaded.fields["density"], data.fields["density"])

    def test_deterministic_encode(self):
        """Test that encode produces consistent output."""
        data = SimpleData(
            name="deterministic",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

        encoded1 = data.encode()
        encoded2 = data.encode()

        assert encoded1 == encoded2

    def test_class_mismatch_error(self):
        """Test error when loading with wrong class."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0])
        )

        buffer = BytesIO()
        data.save_to_zip(buffer)
        buffer.seek(0)

        # Loading wrong class should fail with Pydantic validation error
        # (missing required fields for SimulationResult)
        with pytest.raises(Exception):  # ValidationError from Pydantic
            SimulationResult.load_from_zip(buffer)

    def test_dict_of_basemodel_with_arrays(self):
        """Test containers with dict[str, BaseModel] where BaseModel contains arrays."""
        snapshot = Snapshot(
            time=0.5,
            fields={
                "temperature": FieldData(
                    name="temperature",
                    type="scalar",
                    data=np.array([300.0, 301.0, 302.0], dtype=np.float32),
                    units="K"
                ),
                "velocity": FieldData(
                    name="velocity",
                    type="vector",
                    data=np.array(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                    units="m/s"
                )
            }
        )

        array_fields = snapshot.array_fields
        assert "fields.temperature.data" in array_fields
        assert "fields.velocity.data" in array_fields

        buffer = BytesIO()
        snapshot.save_to_zip(buffer)

        buffer.seek(0)
        loaded = Snapshot.load_from_zip(buffer)

        assert loaded.time == pytest.approx(snapshot.time)

        assert isinstance(loaded.fields["temperature"], FieldData)
        assert isinstance(loaded.fields["velocity"], FieldData)

        assert loaded.fields["temperature"].name == "temperature"
        assert loaded.fields["temperature"].type == "scalar"
        assert loaded.fields["temperature"].units == "K"

        assert loaded.fields["velocity"].name == "velocity"
        assert loaded.fields["velocity"].type == "vector"
        assert loaded.fields["velocity"].units == "m/s"

        np.testing.assert_array_almost_equal(
            loaded.fields["temperature"].data, snapshot.fields["temperature"].data)
        np.testing.assert_array_almost_equal(
            loaded.fields["velocity"].data, snapshot.fields["velocity"].data)

    def test_dict_of_basemodel_with_optional_none_field(self):
        """Test that optional None fields in BaseModel are handled correctly."""
        snapshot = Snapshot(
            time=1.0,
            fields={
                "pressure": FieldData(
                    name="pressure",
                    type="scalar",
                    data=np.array([100.0, 101.0], dtype=np.float32),
                    units=None
                )
            }
        )

        buffer = BytesIO()
        snapshot.save_to_zip(buffer)

        buffer.seek(0)
        loaded = Snapshot.load_from_zip(buffer)

        assert isinstance(loaded.fields["pressure"], FieldData)
        assert loaded.fields["pressure"].units is None
        np.testing.assert_array_almost_equal(
            loaded.fields["pressure"].data, snapshot.fields["pressure"].data)

    def test_decode_without_class_raises_error(self):
        """Test that Packable.decode() raises TypeError - must use specific class."""
        # Create and encode a SimpleData instance
        original = SimpleData(
            name="dynamic_test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        encoded = original.encode()

        # Decode using base Packable class - should raise TypeError
        with pytest.raises(TypeError, match="Cannot decode on base Packable class"):
            Packable.decode(encoded)
        
        # Should work with the specific class
        decoded = SimpleData.decode(encoded)
        assert decoded.name == original.name
        np.testing.assert_array_almost_equal(decoded.values, original.values)

    def test_load_from_zip_without_class_raises_error(self):
        """Test that Packable.load_from_zip() raises TypeError - must use specific class."""
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "result.zip")
            original.save_to_zip(path)

            # Load using base Packable - should raise TypeError
            with pytest.raises(TypeError, match="Cannot decode on base Packable class"):
                Packable.load_from_zip(path)
            
            # Should work with the specific class
            loaded = SimulationResult.load_from_zip(path)
            assert loaded.time == pytest.approx(original.time)
            np.testing.assert_array_almost_equal(
                loaded.temperature, original.temperature)


class TestExtractReconstruct:
    """Test extract() and reconstruct() functionality."""

    def test_extract_simple(self):
        """Test extract() returns data dict with refs and assets."""
        original = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        
        # Data should have the primitive field
        assert extracted.data["name"] == "test"
        
        # Array should be replaced with ref (no $type - we use schema)
        assert "$ref" in extracted.data["values"]
        checksum = extracted.data["values"]["$ref"]
        
        # Assets should contain the encoded array
        assert checksum in extracted.assets
        assert isinstance(extracted.assets[checksum], bytes)

    def test_reconstruct_simple(self):
        """Test reconstruct() rebuilds the Packable from data and assets."""
        original = SimpleData(
            name="roundtrip",
            values=np.array([4.0, 5.0, 6.0], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        reconstructed = Packable.reconstruct(SimpleData, extracted.data, extracted.assets)
        
        assert reconstructed.name == original.name
        np.testing.assert_array_almost_equal(reconstructed.values, original.values)

    def test_extract_reconstruct_simulation_result(self):
        """Test extract/reconstruct with multiple arrays."""
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        
        # Should have 2 assets (2 arrays)
        assert len(extracted.assets) == 2
        
        # Primitive field should be preserved
        assert extracted.data["time"] == 0.5
        
        # Arrays should be refs
        assert "$ref" in extracted.data["temperature"]
        assert "$ref" in extracted.data["velocity"]
        
        # Reconstruct
        reconstructed = Packable.reconstruct(SimulationResult, extracted.data, extracted.assets)
        
        assert reconstructed.time == pytest.approx(original.time)
        np.testing.assert_array_almost_equal(reconstructed.temperature, original.temperature)
        np.testing.assert_array_almost_equal(reconstructed.velocity, original.velocity)

    def test_extract_data_is_json_serializable(self):
        """Test that extracted data can be JSON serialized."""
        original = SimulationResult(
            time=1.0,
            temperature=np.array([100.0], dtype=np.float32),
            velocity=np.array([[0.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        
        # Should be able to serialize to JSON
        json_str = json.dumps(extracted.data)
        assert isinstance(json_str, str)
        
        # And deserialize back
        loaded_data = json.loads(json_str)
        assert loaded_data["time"] == 1.0

    def test_reconstruct_missing_asset_raises(self):
        """Test that reconstruct raises KeyError when asset is missing."""
        data = {"name": "test", "values": {"$ref": "nonexistent_checksum"}}
        
        with pytest.raises(KeyError, match="Missing asset"):
            Packable.reconstruct(SimpleData, data, {})

    def test_extract_requires_basemodel(self):
        """Test extract() requires a Pydantic BaseModel, not plain dict."""
        data = {
            "name": "test",
            "positions": np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        }
        
        with pytest.raises(TypeError, match="requires a Pydantic BaseModel"):
            Packable.extract(data)

    def test_reconstruct_with_callable_returns_lazy_model(self):
        """Test that reconstruct() with callable returns LazyModel for lazy loading."""
        from meshly.packable import LazyModel
        
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        
        # Track which assets were requested
        requested_checksums = []
        
        def lazy_loader(checksum: str) -> bytes:
            """Simulate lazy loading from external storage."""
            requested_checksums.append(checksum)
            if checksum not in extracted.assets:
                raise KeyError(f"Missing asset with checksum '{checksum}'")
            return extracted.assets[checksum]
        
        # Reconstruct using callable - returns LazyModel
        lazy = Packable.reconstruct(
            SimulationResult, extracted.data, lazy_loader
        )
        
        # Should be a LazyModel, not loaded yet
        assert isinstance(lazy, LazyModel)
        assert len(requested_checksums) == 0
        
        # Access fields to trigger loading
        assert lazy.time == pytest.approx(original.time)
        np.testing.assert_array_almost_equal(lazy.temperature, original.temperature)
        np.testing.assert_array_almost_equal(lazy.velocity, original.velocity)
        
        # Now assets should be loaded
        assert len(requested_checksums) == 2

    def test_reconstruct_callable_missing_asset_raises_on_access(self):
        """Test that callable asset provider raises KeyError on field access."""
        data = {"name": "test", "values": {"$ref": "nonexistent"}}
        
        def failing_loader(checksum: str) -> bytes:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        
        # With callable, returns LazyModel immediately (no error)
        lazy = Packable.reconstruct(SimpleData, data, failing_loader)
        
        # Error raised when accessing the field
        with pytest.raises(KeyError, match="Missing asset"):
            _ = lazy.values

    def test_lazy_reconstruct_defers_loading(self):
        """Test that reconstruct() with callable doesn't load assets until accessed."""
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        requested_checksums = []
        
        def tracking_loader(checksum: str) -> bytes:
            requested_checksums.append(checksum)
            return extracted.assets[checksum]
        
        # Create lazy model with callable - NO assets should be loaded yet
        lazy = Packable.reconstruct(
            SimulationResult, extracted.data, tracking_loader
        )
        assert len(requested_checksums) == 0, "No assets should be loaded on creation"
        
        # Access primitive field - still no asset loading
        assert lazy.time == pytest.approx(0.5)
        assert len(requested_checksums) == 0, "Primitive access shouldn't load assets"
        
        # Access temperature - should load only temperature asset
        temp = lazy.temperature
        assert len(requested_checksums) == 1, "Should load exactly one asset"
        np.testing.assert_array_almost_equal(temp, original.temperature)
        
        # Access temperature again - should use cache, not reload
        temp2 = lazy.temperature
        assert len(requested_checksums) == 1, "Cached access shouldn't reload"
        
        # Access velocity - should load velocity asset
        vel = lazy.velocity
        assert len(requested_checksums) == 2, "Should now have loaded both assets"
        np.testing.assert_array_almost_equal(vel, original.velocity)

    def test_lazy_reconstruct_resolve(self):
        """Test that resolve() returns the full Pydantic model."""
        original = SimulationResult(
            time=1.0,
            temperature=np.array([100.0], dtype=np.float32),
            velocity=np.array([[0.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        
        # Use callable to get LazyModel
        lazy = Packable.reconstruct(
            SimulationResult, extracted.data, lambda c: extracted.assets[c]
        )
        
        # Resolve to get actual model
        resolved = lazy.resolve()
        
        # Should be actual SimulationResult instance
        assert isinstance(resolved, SimulationResult)
        assert resolved.time == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(resolved.temperature, original.temperature)
        
        # Resolve again should return same instance
        resolved2 = lazy.resolve()
        assert resolved is resolved2

    def test_lazy_model_repr(self):
        """Test LazyModel has informative repr."""
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0], dtype=np.float32),
            velocity=np.array([[1.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        lazy = Packable.reconstruct(
            SimulationResult, extracted.data, lambda c: extracted.assets[c]
        )
        
        repr_str = repr(lazy)
        assert "LazyModel" in repr_str
        assert "SimulationResult" in repr_str
        
        # After accessing one field, repr should reflect that
        _ = lazy.temperature
        repr_str = repr(lazy)
        assert "temperature" in repr_str

    def test_lazy_model_is_readonly(self):
        """Test that LazyModel doesn't allow attribute setting."""
        original = SimpleData(
            name="test",
            values=np.array([1.0], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        lazy = Packable.reconstruct(
            SimpleData, extracted.data, lambda c: extracted.assets[c]
        )
        
        with pytest.raises(AttributeError, match="read-only"):
            lazy.name = "modified"

    def test_reconstruct_with_cache_handler(self):
        """Test that CachedAssetLoader persists fetched assets to disk."""
        from meshly.data_handler import DataHandler
        from meshly.packable import CachedAssetLoader
        
        original = SimulationResult(
            time=0.5,
            temperature=np.array([300.0, 301.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        
        extracted = Packable.extract(original)
        fetch_count = [0]  # Use list to track calls in closure
        
        def counting_loader(checksum: str) -> bytes:
            fetch_count[0] += 1
            return extracted.assets[checksum]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache"
            cache_handler = DataHandler.create(cache_path)
            
            # First lazy model with CachedAssetLoader - should fetch from loader
            loader1 = CachedAssetLoader(counting_loader, cache_handler)
            lazy1 = Packable.reconstruct(
                SimulationResult, extracted.data, loader1
            )
            
            # Access temperature - should fetch and cache
            _ = lazy1.temperature
            assert fetch_count[0] == 1
            
            # Access velocity - should fetch and cache
            _ = lazy1.velocity
            assert fetch_count[0] == 2
            
            # Finalize to write cache
            cache_handler.finalize()
            
            # Create new cache handler pointing to same location
            cache_handler2 = DataHandler.create(cache_path)
            
            # Second lazy model with same cache - should read from cache
            loader2 = CachedAssetLoader(counting_loader, cache_handler2)
            lazy2 = Packable.reconstruct(
                SimulationResult, extracted.data, loader2
            )
            
            # Access both fields - should NOT call loader (reads from cache)
            temp2 = lazy2.temperature
            vel2 = lazy2.velocity
            assert fetch_count[0] == 2, "Should read from cache, not call loader"
            
            # Verify data integrity
            np.testing.assert_array_almost_equal(temp2, original.temperature)
            np.testing.assert_array_almost_equal(vel2, original.velocity)


class TestNestedPackableRejection:
    """Test nested Packable handling - now allowed with self_contained flag."""

    def test_direct_nested_packable_works(self):
        """Test that a Packable field containing another Packable works."""
        
        class InnerPackable(Packable):
            label: str
            data: Array
        
        class OuterPackable(Packable):
            name: str
            inner: Optional[InnerPackable] = None
        
        inner = InnerPackable(
            label="inner",
            data=np.array([1.0, 2.0], dtype=np.float32)
        )
        
        # Should now work - nested Packables are allowed
        outer = OuterPackable(name="outer", inner=inner)
        assert outer.name == "outer"
        assert outer.inner.label == "inner"
        np.testing.assert_array_equal(outer.inner.data, [1.0, 2.0])
        
        # Round-trip test
        buffer = BytesIO()
        outer.save_to_zip(buffer)
        buffer.seek(0)
        loaded = OuterPackable.load_from_zip(buffer)
        
        assert loaded.name == "outer"
        assert loaded.inner.label == "inner"
        np.testing.assert_array_almost_equal(loaded.inner.data, inner.data)

    def test_dict_of_packables_allowed(self):
        """Test that Dict[str, Packable] is allowed (Packable inside typed dict)."""
        
        class ContainerPackable(Packable):
            name: str
            items: Dict[str, SimpleData] = Field(default_factory=dict)
        
        inner = SimpleData(
            name="inner",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        
        # Should be allowed with typed dict
        container = ContainerPackable(name="container", items={"nested": inner})
        assert container.name == "container"
        assert isinstance(container.items["nested"], SimpleData)

    def test_extract_typed_dict_with_nested_packables(self):
        """Test that extract() handles typed dicts with nested Packables."""
        
        class ContainerPackable(Packable):
            name: str
            items: Dict[str, SimpleData] = Field(default_factory=dict)
        
        inner = SimpleData(
            name="inner",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        
        container = ContainerPackable(name="container", items={"nested": inner})
        
        # Extract expands non-self-contained Packables inline
        extracted = Packable.extract(container)
        
        # The nested Packable should be expanded (not a single $ref)
        # with its arrays as $refs
        nested_data = extracted.data["items"]["nested"]
        assert nested_data["name"] == "inner"
        assert "$ref" in nested_data["values"]  # Array is a ref
        assert "$module" in nested_data  # Module info preserved
        
        # Should have asset for the array
        assert len(extracted.assets) >= 1
        
    def test_reconstruct_typed_dict_with_nested_packables(self):
        """Test that reconstruct() handles typed dicts with nested Packables."""
        
        class ContainerPackable(Packable):
            name: str
            items: Dict[str, SimpleData] = Field(default_factory=dict)
        
        inner = SimpleData(
            name="inner",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        
        container = ContainerPackable(name="container", items={"nested": inner})
        
        # Extract and reconstruct
        extracted = Packable.extract(container)
        reconstructed = Packable.reconstruct(ContainerPackable, extracted.data, extracted.assets)
        
        assert reconstructed.name == "container"
        assert isinstance(reconstructed.items["nested"], SimpleData)
        assert reconstructed.items["nested"].name == "inner"
        np.testing.assert_array_almost_equal(
            reconstructed.items["nested"].values, inner.values
        )

    def test_none_nested_packable_allowed(self):
        """Test that Optional[Packable] = None is allowed."""
        
        class InnerPackable(Packable):
            label: str
            data: Array
        
        class OuterPackable(Packable):
            name: str
            inner: Optional[InnerPackable] = None
        
        # Should work with None
        outer = OuterPackable(name="outer", inner=None)
        assert outer.name == "outer"
        assert outer.inner is None


class TestDataHandler:
    """Test DataHandler functionality."""

    def test_context_manager_file_handler(self):
        """Test DataHandler can be used as context manager with FileHandler."""
        from meshly.data_handler import DataHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            with DataHandler.create(tmpdir) as handler:
                handler.write_text("test.txt", "hello world")
                assert handler.exists("test.txt")

            # File should still exist after context exit
            assert os.path.exists(os.path.join(tmpdir, "test.txt"))

    def test_context_manager_zip_handler(self):
        """Test DataHandler can be used as context manager with ZipHandler."""
        from meshly.data_handler import DataHandler

        buffer = BytesIO()
        with DataHandler.create(buffer) as handler:
            handler.write_text("metadata.json", '{"test": true}')
            handler.write_binary("data.bin", b"binary content")

        # After context exit, zip should be finalized and readable
        buffer.seek(0)
        with DataHandler.create(BytesIO(buffer.read())) as reader:
            content = reader.read_text("metadata.json")
            assert content == '{"test": true}'
            assert reader.read_binary("data.bin") == b"binary content"

    def test_remove_file(self):
        """Test remove_file functionality for FileHandler."""
        from meshly.data_handler import DataHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            handler = DataHandler.create(tmpdir)
            handler.write_text("to_delete.txt", "temporary")
            assert handler.exists("to_delete.txt")

            handler.remove_file("to_delete.txt")
            assert not handler.exists("to_delete.txt")

    def test_remove_file_zip_raises(self):
        """Test remove_file raises NotImplementedError for ZipHandler."""
        from meshly.data_handler import DataHandler

        buffer = BytesIO()
        with DataHandler.create(buffer) as handler:
            handler.write_text("test.txt", "content")
            with pytest.raises(NotImplementedError):
                handler.remove_file("test.txt")
