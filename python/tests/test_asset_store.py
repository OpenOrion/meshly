"""Tests for AssetStore and Packable save/load."""

import pytest
import tempfile
import os
import json
from pathlib import Path
import numpy as np
from pydantic import Field

from meshly.packable import Packable
from meshly.asset_store import AssetStore
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


class TestAssetStore:
    """Test AssetStore functionality."""

    def test_create_store(self):
        """Test creating an AssetStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            assert store.assets_path == Path(tmpdir)
            assert store.metadata_path == Path(tmpdir)

    def test_create_store_separate_paths(self):
        """Test creating an AssetStore with separate asset and metadata paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = os.path.join(tmpdir, "assets")
            metadata_path = os.path.join(tmpdir, "metadata")
            store = AssetStore(assets_path, metadata_path)
            assert store.assets_path == Path(assets_path)
            assert store.metadata_path == Path(metadata_path)
            # Directories should be created
            assert store.assets_path.exists()
            assert store.metadata_path.exists()

    def test_save_load_asset(self):
        """Test saving and loading binary assets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            data = b"test binary data"
            checksum = "abc123"
            
            # Save asset
            path = store.save_asset(data, checksum)
            assert path.exists()
            assert store.asset_exists(checksum)
            
            # Load asset
            loaded = store.load_asset(checksum)
            assert loaded == data

    def test_load_nonexistent_asset(self):
        """Test loading a non-existent asset raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            with pytest.raises(FileNotFoundError):
                store.load_asset("nonexistent")

    def test_asset_deduplication(self):
        """Test that saving the same checksum twice doesn't duplicate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            data1 = b"original data"
            data2 = b"different data"
            checksum = "same_checksum"
            
            # Save first
            store.save_asset(data1, checksum)
            
            # Save again with different data (should be ignored)
            store.save_asset(data2, checksum)
            
            # Should still have original data
            loaded = store.load_asset(checksum)
            assert loaded == data1


class TestPackableSaveToStore:
    """Test Packable.save and load."""

    def test_save_auto_path(self):
        """Test saving to store with auto-generated checksum path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            path = data.save(store)
            
            # Path should be the checksum
            assert path == data.checksum
            assert store.exists(path)

    def test_save_explicit_path(self):
        """Test saving to store with explicit path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            path = data.save(store, "my/custom/path")
            
            assert path == "my/custom/path"
            assert store.exists("my/custom/path")

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            original = SimulationResult(
                time=0.5,
                temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
                velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            )
            
            path = original.save(store)
            loaded = SimulationResult.load(store, path)
            
            assert loaded.time == pytest.approx(original.time)
            np.testing.assert_array_almost_equal(
                loaded.temperature, original.temperature)
            np.testing.assert_array_almost_equal(
                loaded.velocity, original.velocity)

    def test_save_load_separate_paths(self):
        """Test save/load with separate asset and metadata paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = os.path.join(tmpdir, "assets")
            metadata_path = os.path.join(tmpdir, "runs")
            store = AssetStore(assets_path, metadata_path)
            
            original = SimpleData(
                name="experiment",
                values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            )
            
            original.save(store, "exp001/result")
            
            # Check directory structure
            assert (Path(metadata_path) / "exp001" / "result" / "data.json").exists()
            assert (Path(metadata_path) / "exp001" / "result" / "schema.json").exists()
            
            # Load and verify
            loaded = SimpleData.load(store, "exp001/result")
            assert loaded.name == original.name
            np.testing.assert_array_almost_equal(loaded.values, original.values)

    def test_load_nonexistent_path(self):
        """Test loading from non-existent path returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            result = SimpleData.load(store, "nonexistent/path")
            assert result is None

    def test_lazy_loading(self):
        """Test lazy loading from store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            original = SimulationResult(
                time=0.5,
                temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
                velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            )
            
            path = original.save(store)
            lazy = SimulationResult.load(store, path, is_lazy=True)
            
            # Check lazy model
            from meshly.lazy_model import LazyModel
            assert isinstance(lazy, LazyModel)
            
            # Access primitive field (no fetch needed)
            assert lazy.time == pytest.approx(original.time)
            
            # Access array field (triggers fetch)
            np.testing.assert_array_almost_equal(
                lazy.temperature, original.temperature)

    def test_deduplication_across_saves(self):
        """Test that identical arrays are deduplicated across saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            shared_temp = np.array([300.0, 301.0, 302.0], dtype=np.float32)
            
            result1 = SimulationResult(
                time=0.0,
                temperature=shared_temp,
                velocity=np.array([[1.0, 0.0]], dtype=np.float32)
            )
            
            result2 = SimulationResult(
                time=1.0,
                temperature=shared_temp,
                velocity=np.array([[0.0, 1.0]], dtype=np.float32)
            )
            
            result1.save(store, "result1")
            result2.save(store, "result2")
            
            # Count asset files (should have 3: shared temp, vel1, vel2)
            asset_files = list(store.assets_path.glob("*.bin"))
            assert len(asset_files) == 3  # temp is shared

    def test_metadata_files_content(self):
        """Test that metadata files have correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AssetStore(tmpdir)
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            data.save(store, "test_path")
            
            # Check data.json
            data_dict = store.load_data("test_path")
            assert data_dict is not None
            assert data_dict["name"] == "test"
            assert "$ref" in data_dict["values"]
            
            # Check schema.json
            schema = store.load_schema("test_path")
            assert schema is not None
            assert "properties" in schema
