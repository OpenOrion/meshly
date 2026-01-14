"""Tests for Packable base class."""

import unittest
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
    fields: dict[str, FieldData] = Field(
        default_factory=dict, description="Field data")


class TestPackable(unittest.TestCase):
    """Test Packable functionality."""

    def test_simple_container(self):
        """Test basic Packable creation."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        self.assertEqual(data.name, "test")
        np.testing.assert_array_equal(data.values, [1.0, 2.0, 3.0])

    def test_array_fields_detection(self):
        """Test automatic array field detection."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0])
        )
        self.assertIn("values", data.array_fields)
        self.assertNotIn("name", data.array_fields)

    def test_encode_decode(self):
        """Test encoding and decoding via zip round-trip."""
        original = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

        # Test that encode produces bytes
        encoded = original.encode()
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)

        # Test full round-trip via zip
        buffer = BytesIO()
        original.save_to_zip(buffer)
        buffer.seek(0)
        decoded = SimpleData.load_from_zip(buffer)
        np.testing.assert_array_almost_equal(decoded.values, original.values)
        self.assertEqual(decoded.name, original.name)

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

            self.assertAlmostEqual(loaded.time, original.time)
            np.testing.assert_array_almost_equal(
                loaded.temperature, original.temperature
            )
            np.testing.assert_array_almost_equal(
                loaded.velocity, original.velocity
            )

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

        self.assertEqual(loaded.name, original.name)
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

        # Check nested array detection
        array_fields = data.array_fields
        self.assertIn("fields.pressure", array_fields)
        self.assertIn("fields.density", array_fields)

        # Test round-trip
        buffer = BytesIO()
        data.save_to_zip(buffer)

        buffer.seek(0)
        loaded = NestedData.load_from_zip(buffer)

        self.assertEqual(loaded.label, data.label)
        np.testing.assert_array_almost_equal(
            loaded.fields["pressure"], data.fields["pressure"]
        )
        np.testing.assert_array_almost_equal(
            loaded.fields["density"], data.fields["density"]
        )

    def test_deterministic_encode(self):
        """Test that encode produces consistent output."""
        data = SimpleData(
            name="deterministic",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

        # Multiple encodes should produce the same bytes
        encoded1 = data.encode()
        encoded2 = data.encode()

        self.assertEqual(encoded1, encoded2)

    def test_class_mismatch_error(self):
        """Test error when loading with wrong class."""
        data = SimpleData(
            name="test",
            values=np.array([1.0, 2.0, 3.0])
        )

        buffer = BytesIO()
        data.save_to_zip(buffer)
        buffer.seek(0)

        with self.assertRaises(ValueError) as ctx:
            SimulationResult.load_from_zip(buffer)

        self.assertIn("Class mismatch", str(ctx.exception))

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

        # Check nested array detection
        array_fields = snapshot.array_fields
        self.assertIn("fields.temperature.data", array_fields)
        self.assertIn("fields.velocity.data", array_fields)

        # Test round-trip
        buffer = BytesIO()
        snapshot.save_to_zip(buffer)

        buffer.seek(0)
        loaded = Snapshot.load_from_zip(buffer)

        self.assertAlmostEqual(loaded.time, snapshot.time)

        # Check that FieldData instances are reconstructed
        self.assertIsInstance(loaded.fields["temperature"], FieldData)
        self.assertIsInstance(loaded.fields["velocity"], FieldData)

        # Check non-array fields
        self.assertEqual(loaded.fields["temperature"].name, "temperature")
        self.assertEqual(loaded.fields["temperature"].type, "scalar")
        self.assertEqual(loaded.fields["temperature"].units, "K")

        self.assertEqual(loaded.fields["velocity"].name, "velocity")
        self.assertEqual(loaded.fields["velocity"].type, "vector")
        self.assertEqual(loaded.fields["velocity"].units, "m/s")

        # Check array data
        np.testing.assert_array_almost_equal(
            loaded.fields["temperature"].data, snapshot.fields["temperature"].data
        )
        np.testing.assert_array_almost_equal(
            loaded.fields["velocity"].data, snapshot.fields["velocity"].data
        )

    def test_dict_of_basemodel_with_optional_none_field(self):
        """Test that optional None fields in BaseModel are handled correctly."""
        snapshot = Snapshot(
            time=1.0,
            fields={
                "pressure": FieldData(
                    name="pressure",
                    type="scalar",
                    data=np.array([100.0, 101.0], dtype=np.float32),
                    units=None  # Optional field set to None
                )
            }
        )

        buffer = BytesIO()
        snapshot.save_to_zip(buffer)

        buffer.seek(0)
        loaded = Snapshot.load_from_zip(buffer)

        self.assertIsInstance(loaded.fields["pressure"], FieldData)
        self.assertIsNone(loaded.fields["pressure"].units)
        np.testing.assert_array_almost_equal(
            loaded.fields["pressure"].data, snapshot.fields["pressure"].data
        )


class InnerPackable(Packable):
    """Inner packable for testing nested support."""
    label: str = Field(..., description="Label")
    data: np.ndarray = Field(..., description="Data array")


class OuterPackable(Packable):
    """Outer packable containing a nested packable."""
    name: str = Field(..., description="Name")
    inner: Optional[InnerPackable] = Field(None, description="Nested packable")


class TestNestedPackableCache(unittest.TestCase):
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

        self.assertEqual(loaded.name, "outer")
        self.assertIsNotNone(loaded.inner)
        self.assertEqual(loaded.inner.label, "inner")
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

            # Create cache saver and save with cache
            cache_saver = WriteHandler.create_cache_saver(cache_path)
            outer.save_to_zip(zip_path, cache_saver=cache_saver)

            # Verify cache file was created
            cache_files = os.listdir(cache_path)
            self.assertEqual(len(cache_files), 1)
            self.assertTrue(cache_files[0].endswith(".zip"))

            # Create cache loader and load with cache
            cache_loader = ReadHandler.create_cache_loader(cache_path)
            loaded = OuterPackable.load_from_zip(zip_path, cache_loader=cache_loader)

            self.assertEqual(loaded.name, "cached_outer")
            self.assertIsNotNone(loaded.inner)
            self.assertEqual(loaded.inner.label, "cached_inner")
            np.testing.assert_array_almost_equal(loaded.inner.data, inner.data)

    def test_cache_deduplication(self):
        """Test that identical nested packables share the same cache file."""
        from meshly.data_handler import ReadHandler, WriteHandler

        # Create identical inner packables
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

            # Save both with same cache
            cache_saver = WriteHandler.create_cache_saver(cache_path)
            outer1.save_to_zip(zip1_path, cache_saver=cache_saver)
            outer2.save_to_zip(zip2_path, cache_saver=cache_saver)

            # Both should use the same cache file (SHA256 deduplication)
            cache_files = os.listdir(cache_path)
            self.assertEqual(len(cache_files), 1)

            # Both should load correctly
            cache_loader = ReadHandler.create_cache_loader(cache_path)
            loaded1 = OuterPackable.load_from_zip(zip1_path, cache_loader=cache_loader)
            loaded2 = OuterPackable.load_from_zip(zip2_path, cache_loader=cache_loader)

            self.assertEqual(loaded1.inner.label, "same")
            self.assertEqual(loaded2.inner.label, "same")

    def test_cache_missing_falls_back_to_embedded(self):
        """Test loading works when cache file is missing but data is embedded."""
        from meshly.data_handler import ReadHandler

        inner = InnerPackable(
            label="fallback",
            data=np.array([7.0, 8.0], dtype=np.float32)
        )
        outer = OuterPackable(name="fallback_outer", inner=inner)

        # Save without cache (embedded)
        buffer = BytesIO()
        outer.save_to_zip(buffer)

        # Load with a cache loader that won't find anything (should still work from embedded data)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache")
            os.makedirs(cache_path)  # Create empty cache dir
            cache_loader = ReadHandler.create_cache_loader(cache_path)
            buffer.seek(0)
            loaded = OuterPackable.load_from_zip(buffer, cache_loader=cache_loader)

            self.assertEqual(loaded.name, "fallback_outer")
            self.assertEqual(loaded.inner.label, "fallback")


if __name__ == "__main__":
    unittest.main()
