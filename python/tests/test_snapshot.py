"""Tests for snapshot module."""
import unittest
import numpy as np
from io import BytesIO
from pathlib import Path
import tempfile
import zipfile
import json

from meshly import Snapshot, FieldData, FieldMetadata, SnapshotMetadata, SnapshotUtils


class TestFieldData(unittest.TestCase):
    """Tests for FieldData class."""

    def test_field_data_creation(self):
        """Test creating a FieldData instance."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        field = FieldData(name="test", type="scalar", data=data, units="m/s")

        self.assertEqual(field.name, "test")
        self.assertEqual(field.type, "scalar")
        self.assertEqual(field.units, "m/s")
        self.assertTrue(np.array_equal(field.data, data))

    def test_field_data_no_units(self):
        """Test creating a FieldData without units."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        field = FieldData(name="test", type="scalar", data=data)

        self.assertIsNone(field.units)


class TestSnapshot(unittest.TestCase):
    """Tests for Snapshot class."""

    def test_snapshot_creation(self):
        """Test creating a Snapshot instance."""
        snapshot = Snapshot(time=0.5)
        self.assertEqual(snapshot.time, 0.5)
        self.assertEqual(snapshot.fields, {})

    def test_add_field(self):
        """Test adding fields to a snapshot."""
        snapshot = Snapshot(time=1.0)
        data = np.random.rand(100, 3).astype(np.float32)
        snapshot.add_field("velocity", data, "vector", "m/s")

        self.assertIn("velocity", snapshot.fields)
        self.assertEqual(snapshot.fields["velocity"].name, "velocity")
        self.assertEqual(snapshot.fields["velocity"].type, "vector")
        self.assertEqual(snapshot.fields["velocity"].units, "m/s")
        self.assertTrue(np.array_equal(snapshot.fields["velocity"].data, data))

    def test_get_field(self):
        """Test getting a field by name."""
        snapshot = Snapshot(time=1.0)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        snapshot.add_field("pressure", data, "scalar", "Pa")

        field = snapshot.get_field("pressure")
        self.assertIsNotNone(field)
        self.assertEqual(field.name, "pressure")

        # Non-existent field
        self.assertIsNone(snapshot.get_field("nonexistent"))

    def test_field_names(self):
        """Test getting field names."""
        snapshot = Snapshot(time=1.0)
        snapshot.add_field("velocity", np.zeros(
            10, dtype=np.float32), "vector")
        snapshot.add_field("pressure", np.zeros(
            10, dtype=np.float32), "scalar")

        names = snapshot.field_names
        self.assertIn("velocity", names)
        self.assertIn("pressure", names)
        self.assertEqual(len(names), 2)


class TestSnapshotIO(unittest.TestCase):
    """Tests for Snapshot I/O operations."""

    def test_save_and_load_bytesio(self):
        """Test saving and loading from BytesIO."""
        # Create snapshot with fields
        snapshot = Snapshot(time=0.5)
        velocity = np.random.rand(100, 3).astype(np.float32)
        pressure = np.random.rand(100).astype(np.float32)
        snapshot.add_field("velocity", velocity, "vector", "m/s")
        snapshot.add_field("pressure", pressure, "scalar", "Pa")

        # Save to BytesIO
        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        # Load back
        loaded = SnapshotUtils.load_from_zip(buffer)

        self.assertEqual(loaded.time, snapshot.time)
        self.assertEqual(set(loaded.field_names), set(snapshot.field_names))
        self.assertTrue(np.allclose(loaded.fields["velocity"].data, velocity))
        self.assertTrue(np.allclose(loaded.fields["pressure"].data, pressure))
        self.assertEqual(loaded.fields["velocity"].units, "m/s")
        self.assertEqual(loaded.fields["pressure"].units, "Pa")

    def test_save_and_load_file(self):
        """Test saving and loading from a file."""
        snapshot = Snapshot(time=1.5)
        data = np.linspace(0, 100, 1000).astype(np.float32)
        snapshot.add_field("temperature", data, "scalar", "K")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "snapshot.zip"
            SnapshotUtils.save_to_zip(snapshot, filepath)

            # Verify file exists and is valid zip
            self.assertTrue(filepath.exists())
            with zipfile.ZipFile(filepath, "r") as zf:
                self.assertIn("metadata.json", zf.namelist())
                self.assertIn("fields/temperature.bin", zf.namelist())

            # Load back
            loaded = SnapshotUtils.load_from_zip(filepath)

            self.assertEqual(loaded.time, 1.5)
            self.assertIn("temperature", loaded.field_names)
            self.assertTrue(np.allclose(
                loaded.fields["temperature"].data, data))

    def test_compression_ratio(self):
        """Test that data is actually compressed."""
        snapshot = Snapshot(time=0.0)
        # Sequential data compresses well
        data = np.arange(10000, dtype=np.float32)
        snapshot.add_field("indices", data, "scalar")

        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        original_size = data.nbytes
        compressed_size = len(buffer.getvalue())

        # Should be significantly smaller (meshopt + zip deflate)
        self.assertLess(compressed_size, original_size * 0.5,
                        f"Compression ratio too high: {compressed_size}/{original_size}")

    def test_deterministic_output(self):
        """Test that output is deterministic with date_time."""
        snapshot = Snapshot(time=1.0)
        snapshot.add_field("data", np.array(
            [1.0, 2.0, 3.0], dtype=np.float32), "scalar")

        date_time = (2024, 1, 1, 0, 0, 0)

        buffer1 = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer1, date_time=date_time)

        buffer2 = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer2, date_time=date_time)

        self.assertEqual(buffer1.getvalue(), buffer2.getvalue())

    def test_multidimensional_arrays(self):
        """Test saving/loading multidimensional arrays."""
        snapshot = Snapshot(time=0.0)
        # 3D tensor field
        tensor = np.random.rand(100, 3, 3).astype(np.float32)
        snapshot.add_field("stress", tensor, "tensor", "Pa")

        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        loaded = SnapshotUtils.load_from_zip(buffer)
        self.assertEqual(loaded.fields["stress"].data.shape, (100, 3, 3))
        self.assertTrue(np.allclose(loaded.fields["stress"].data, tensor))

    def test_empty_snapshot(self):
        """Test saving/loading a snapshot with no fields."""
        snapshot = Snapshot(time=42.0)

        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        loaded = SnapshotUtils.load_from_zip(buffer)
        self.assertEqual(loaded.time, 42.0)
        self.assertEqual(loaded.fields, {})


class TestSnapshotUtils(unittest.TestCase):
    """Tests for SnapshotUtils class."""

    def test_save_via_utils(self):
        """Test saving via SnapshotUtils."""
        snapshot = Snapshot(time=1.0)
        snapshot.add_field("data", np.zeros(10, dtype=np.float32), "scalar")

        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        loaded = SnapshotUtils.load_from_zip(buffer)
        self.assertEqual(loaded.time, 1.0)
        self.assertIn("data", loaded.field_names)

    def test_load_via_utils(self):
        """Test loading via SnapshotUtils."""
        snapshot = Snapshot(time=2.0)
        snapshot.add_field("velocity", np.ones(
            (50, 3), dtype=np.float32), "vector")

        buffer = BytesIO()
        SnapshotUtils.save_to_zip(snapshot, buffer)
        buffer.seek(0)

        loaded = SnapshotUtils.load_from_zip(buffer)
        self.assertEqual(loaded.time, 2.0)
        self.assertEqual(loaded.fields["velocity"].data.shape, (50, 3))


if __name__ == "__main__":
    unittest.main()
