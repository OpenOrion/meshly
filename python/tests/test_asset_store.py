"""Tests for PackableStore and Packable save/load."""

import pytest
import tempfile
import os
import json
from pathlib import Path
import numpy as np
from pydantic import Field

from meshly.packable import Packable, PackableStore
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


class TestPackableStore:
    """Test PackableStore functionality."""

    def test_create_store(self):
        """Test creating a PackableStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            assert store.assets_path == Path(tmpdir)
            assert store.extracted_path is None

    def test_create_store_separate_paths(self):
        """Test creating a PackableStore with separate asset and extracted paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "assets"
            extracted_path = Path(tmpdir) / "extracted"
            store = PackableStore(assets_path=assets_path, extracted_path=extracted_path)
            assert store.assets_path == assets_path
            assert store.extracted_path == extracted_path

    def test_asset_file_path(self):
        """Test asset_file returns correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            checksum = "abc123"
            path = store.asset_file(checksum)
            assert path == Path(tmpdir) / "abc123.bin"

    def test_extracted_file_path(self):
        """Test extracted_file returns correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            key = "my/custom/path"
            path = store.extracted_file(key)
            assert path == Path(tmpdir) / "my" / "custom" / "path.json"


class TestPackableSaveToStore:
    """Test Packable.save and load."""

    def test_save_auto_key(self):
        """Test saving to store with auto-generated checksum key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            key = data.save(store)
            
            # Key should be the checksum
            assert key == data.checksum
            assert store.extracted_file(key).exists()

    def test_save_explicit_key(self):
        """Test saving to store with explicit key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            key = data.save(store, "my/custom/path")
            
            assert key == "my/custom/path"
            assert store.extracted_file("my/custom/path").exists()

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            original = SimulationResult(
                time=0.5,
                temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
                velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            )
            
            key = original.save(store)
            loaded = SimulationResult.load(store, key)
            
            assert loaded.time == pytest.approx(original.time)
            np.testing.assert_array_almost_equal(
                loaded.temperature, original.temperature)
            np.testing.assert_array_almost_equal(
                loaded.velocity, original.velocity)

    def test_save_load_separate_paths(self):
        """Test save/load with separate asset and extracted paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_path = Path(tmpdir) / "assets"
            extracted_path = Path(tmpdir) / "runs"
            store = PackableStore(assets_path=assets_path, extracted_path=extracted_path)
            
            original = SimpleData(
                name="experiment",
                values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            )
            
            original.save(store, "exp001/result")
            
            # Check directory structure - single json file
            assert (extracted_path / "exp001" / "result.json").exists()
            
            # Load and verify
            loaded = SimpleData.load(store, "exp001/result")
            assert loaded.name == original.name
            np.testing.assert_array_almost_equal(loaded.values, original.values)

    def test_load_nonexistent_key(self):
        """Test loading from non-existent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            result = SimpleData.load(store, "nonexistent/path")
            assert result is None

    def test_lazy_loading(self):
        """Test lazy loading from store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            original = SimulationResult(
                time=0.5,
                temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
                velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            )
            
            key = original.save(store)
            lazy = SimulationResult.load(store, key, is_lazy=True)
            
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
            store = PackableStore(assets_path=Path(tmpdir))
            
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

    def test_extracted_files_content(self):
        """Test that extracted files have correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(assets_path=Path(tmpdir))
            
            data = SimpleData(
                name="test",
                values=np.array([1.0, 2.0, 3.0], dtype=np.float32)
            )
            
            data.save(store, "test_key")
            
            # Check extracted file directly
            import json
            file_path = store.extracted_file("test_key")
            assert file_path.exists()
            extracted = json.loads(file_path.read_text())
            assert "data" in extracted
            assert "json_schema" in extracted
            assert extracted["data"]["name"] == "test"
            assert "$ref" in extracted["data"]["values"]
            assert "properties" in extracted["json_schema"]
