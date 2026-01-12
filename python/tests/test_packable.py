"""Tests for Packable base class."""

import unittest
import tempfile
import os
from io import BytesIO
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

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


if __name__ == "__main__":
    unittest.main()
