"""Tests for Packable base class."""

import pytest
import tempfile
import os
from io import BytesIO
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from meshly.packable import Packable


class SimpleData(Packable):
    """Simple test data container."""
    name: str = Field(..., description="Name")
    values: np.ndarray = Field(..., description="Values array")


class SimulationResult(Packable):
    """Simulation result with multiple arrays."""
    time: float = Field(..., description="Time value")
    temperature: np.ndarray = Field(..., description="Temperature field")
    velocity: np.ndarray = Field(default=None, description="Velocity field")


class NestedData(Packable):
    """Data container with nested dictionary arrays."""
    label: str = Field(..., description="Label")
    fields: dict = Field(default_factory=dict, description="Named fields")


class FieldData(BaseModel):
    """Generic field data container for testing dict[str, BaseModel] edge case."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (scalar, vector, tensor)")
    data: np.ndarray = Field(..., description="Field data array")
    units: Optional[str] = Field(None, description="Field units")


class Snapshot(Packable):
    """Snapshot with dict of BaseModel containing arrays."""
    time: float = Field(..., description="Time value")
    fields: dict[str, FieldData] = Field(default_factory=dict, description="Field data")


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
            np.testing.assert_array_almost_equal(loaded.temperature, original.temperature)
            np.testing.assert_array_almost_equal(loaded.velocity, original.velocity)

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
        np.testing.assert_array_almost_equal(loaded.fields["pressure"], data.fields["pressure"])
        np.testing.assert_array_almost_equal(loaded.fields["density"], data.fields["density"])

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

        with pytest.raises(ValueError, match="Class mismatch"):
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
                    data=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
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


class InnerPackable(Packable):
    """Inner packable for testing nested support."""
    label: str = Field(..., description="Label")
    data: np.ndarray = Field(..., description="Data array")


class OuterPackable(Packable):
    """Outer packable containing a nested packable."""
    name: str = Field(..., description="Name")
    inner: Optional[InnerPackable] = Field(None, description="Nested packable")


class TestNestedPackableCache:
    """Test nested Packable with cache support."""

    def test_nested_packable_without_cache(self):
        """Test nested packable save/load without cache."""
        inner = InnerPackable(
            label="inner",
            data=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        outer = OuterPackable(name="outer", inner=inner)

        buffer = BytesIO()
        outer.save_to_zip(buffer)

        buffer.seek(0)
        loaded = OuterPackable.load_from_zip(buffer)

        assert loaded.name == "outer"
        assert loaded.inner is not None
        assert loaded.inner.label == "inner"
        np.testing.assert_array_almost_equal(loaded.inner.data, inner.data)

    def test_nested_packable_with_cache(self):
        """Test nested packable save/load with cache."""
        from meshly.data_handler import ReadHandler, WriteHandler

        inner = InnerPackable(
            label="cached_inner",
            data=np.array([4.0, 5.0, 6.0], dtype=np.float32)
        )
        outer = OuterPackable(name="cached_outer", inner=inner)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache")
            zip_path = os.path.join(tmpdir, "outer.zip")

            cache_saver = WriteHandler.create_cache_saver(cache_path)
            outer.save_to_zip(zip_path, cache_saver=cache_saver)

            cache_files = os.listdir(cache_path)
            assert len(cache_files) == 1
            assert cache_files[0].endswith(".zip")

            cache_loader = ReadHandler.create_cache_loader(cache_path)
            loaded = OuterPackable.load_from_zip(zip_path, cache_loader=cache_loader)

            assert loaded.name == "cached_outer"
            assert loaded.inner is not None
            assert loaded.inner.label == "cached_inner"
            np.testing.assert_array_almost_equal(loaded.inner.data, inner.data)

    def test_cache_deduplication(self):
        """Test that identical nested packables share the same cache file."""
        from meshly.data_handler import ReadHandler, WriteHandler

        inner1 = InnerPackable(
            label="same",
            data=np.array([1.0, 2.0], dtype=np.float32)
        )
        inner2 = InnerPackable(
            label="same",
            data=np.array([1.0, 2.0], dtype=np.float32)
        )
        outer1 = OuterPackable(name="outer1", inner=inner1)
        outer2 = OuterPackable(name="outer2", inner=inner2)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache")
            zip1_path = os.path.join(tmpdir, "outer1.zip")
            zip2_path = os.path.join(tmpdir, "outer2.zip")

            cache_saver = WriteHandler.create_cache_saver(cache_path)
            outer1.save_to_zip(zip1_path, cache_saver=cache_saver)
            outer2.save_to_zip(zip2_path, cache_saver=cache_saver)

            cache_files = os.listdir(cache_path)
            assert len(cache_files) == 1

            cache_loader = ReadHandler.create_cache_loader(cache_path)
            loaded1 = OuterPackable.load_from_zip(zip1_path, cache_loader=cache_loader)
            loaded2 = OuterPackable.load_from_zip(zip2_path, cache_loader=cache_loader)

            assert loaded1.inner.label == "same"
            assert loaded2.inner.label == "same"

    def test_cache_missing_falls_back_to_embedded(self):
        """Test loading works when cache file is missing but data is embedded."""
        from meshly.data_handler import ReadHandler

        inner = InnerPackable(
            label="fallback",
            data=np.array([7.0, 8.0], dtype=np.float32)
        )
        outer = OuterPackable(name="fallback_outer", inner=inner)

        buffer = BytesIO()
        outer.save_to_zip(buffer)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache")
            os.makedirs(cache_path)
            cache_loader = ReadHandler.create_cache_loader(cache_path)
            buffer.seek(0)
            loaded = OuterPackable.load_from_zip(buffer, cache_loader=cache_loader)

            assert loaded.name == "fallback_outer"
            assert loaded.inner.label == "fallback"
