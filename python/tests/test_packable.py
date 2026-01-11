"""Tests for Packable base class."""

import unittest
import tempfile
import os
from io import BytesIO
import numpy as np
from pydantic import Field

from meshly.packable import Packable, EncodedData


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

        # Test that encode produces arrays
        encoded = original.encode()
        self.assertIn("values", encoded.arrays)

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

    def test_deterministic_zip(self):
        """Test deterministic zip output with date_time parameter."""
        data = SimpleData(
            name="deterministic",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

        date_time = (2020, 1, 1, 0, 0, 0)

        buffer1 = BytesIO()
        data.save_to_zip(buffer1, date_time=date_time)

        buffer2 = BytesIO()
        data.save_to_zip(buffer2, date_time=date_time)

        self.assertEqual(buffer1.getvalue(), buffer2.getvalue())

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


if __name__ == "__main__":
    unittest.main()
