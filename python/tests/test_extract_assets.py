"""Tests for Packable.extract_assets() static method and ExtractedAssets."""

import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from meshly import Packable, ResourceRef
from meshly.packable import ExtractedAssets


class TestExtractAssets:
    """Test Packable.extract_assets() functionality."""

    def test_extract_assets_from_single_array(self):
        """Test extracting assets from a single numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        extracted = Packable.extract_assets(arr)

        assert isinstance(extracted, ExtractedAssets)
        assert len(extracted.assets) == 1
        assert len(extracted.extensions) == 0  # Arrays don't have extensions

        # Check that the asset is stored
        checksum = list(extracted.assets.keys())[0]
        assert isinstance(checksum, str)
        assert len(checksum) == 16  # Checksum is 16 characters
        assert isinstance(extracted.assets[checksum], bytes)

    def test_extract_assets_from_multiple_arrays(self):
        """Test extracting assets from multiple arrays."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        arr3 = np.array([5.0, 6.0], dtype=np.float64)

        extracted = Packable.extract_assets(arr1, arr2, arr3)

        assert isinstance(extracted, ExtractedAssets)
        assert len(extracted.assets) == 3
        assert len(extracted.extensions) == 0

    def test_extract_assets_with_deduplication(self):
        """Test that identical arrays are deduplicated."""
        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Same as arr1
        arr3 = np.array([4.0, 5.0], dtype=np.float32)  # Different

        extracted = Packable.extract_assets(arr1, arr2, arr3)

        # Should only have 2 assets since arr1 and arr2 are identical
        assert len(extracted.assets) == 2

    def test_extract_assets_from_packable(self):
        """Test extracting assets from a Packable instance."""

        class SimpleData(Packable):
            name: str
            values: np.ndarray

        data = SimpleData(name="test", values=np.array([1.0, 2.0, 3.0], dtype=np.float32))

        extracted = Packable.extract_assets(data)

        # Should have the encoded Packable as an asset
        assert len(extracted.assets) >= 1
        assert len(extracted.extensions) == 0

    def test_extract_assets_from_resource_ref(self):
        """Test extracting assets from ResourceRef."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"STL geometry data")
            temp_path = f.name

        try:
            ref = ResourceRef(path=temp_path)

            extracted = Packable.extract_assets(ref)

            # Should have the file data as an asset
            assert len(extracted.assets) == 1
            # Should have the extension mapping
            assert len(extracted.extensions) == 1

            checksum = list(extracted.assets.keys())[0]
            assert extracted.assets[checksum] == b"STL geometry data"
            assert extracted.extensions[checksum] == ".stl"
        finally:
            Path(temp_path).unlink()

    def test_extract_assets_from_basemodel_with_arrays(self):
        """Test extracting assets from a BaseModel containing arrays."""

        class DataContainer(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            name: str
            temperature: np.ndarray
            velocity: np.ndarray

        container = DataContainer(
            name="simulation",
            temperature=np.array([300.0, 301.0, 302.0], dtype=np.float32),
            velocity=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        extracted = Packable.extract_assets(container)

        # Should have both arrays as assets
        assert len(extracted.assets) == 2

    def test_extract_assets_from_dict_with_arrays(self):
        """Test extracting assets from a dictionary containing arrays."""
        data = {
            "temp": np.array([100.0, 200.0], dtype=np.float32),
            "pressure": np.array([1.0, 2.0], dtype=np.float32),
            "name": "test",  # Non-array field
        }

        extracted = Packable.extract_assets(data)

        # Should have both arrays
        assert len(extracted.assets) == 2

    def test_extract_assets_from_list_with_arrays(self):
        """Test extracting assets from a list containing arrays."""
        data = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
            "string value",  # Non-array
        ]

        extracted = Packable.extract_assets(data)

        # Should have both arrays
        assert len(extracted.assets) == 2

    def test_extract_assets_from_nested_structure(self):
        """Test extracting assets from deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "array": np.array([1.0, 2.0], dtype=np.float32),
                    "list": [
                        np.array([3.0, 4.0], dtype=np.float32),
                        {"nested_array": np.array([5.0], dtype=np.float32)},
                    ],
                }
            }
        }

        extracted = Packable.extract_assets(data)

        # Should extract all 3 arrays
        assert len(extracted.assets) == 3

    def test_extract_assets_from_multiple_basemodels(self):
        """Test extracting assets from multiple BaseModel instances."""

        class Result(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            time: float
            data: np.ndarray

        result1 = Result(time=0.0, data=np.array([1.0, 2.0], dtype=np.float32))
        result2 = Result(time=1.0, data=np.array([3.0, 4.0], dtype=np.float32))

        extracted = Packable.extract_assets(result1, result2)

        # Should have both arrays
        assert len(extracted.assets) == 2

    def test_extract_assets_mixed_types(self):
        """Test extracting assets from mixed argument types."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"file content")
            temp_path = f.name

        try:

            class SimpleData(Packable):
                values: np.ndarray

            arr = np.array([1.0, 2.0], dtype=np.float32)
            packable = SimpleData(values=np.array([3.0, 4.0], dtype=np.float32))
            resource = ResourceRef(path=temp_path)
            base_dict = {"array": np.array([5.0], dtype=np.float32)}

            extracted = Packable.extract_assets(arr, packable, resource, base_dict)

            # Should have all assets
            assert len(extracted.assets) >= 4
            # Should have extension for the resource
            assert len(extracted.extensions) >= 1
            assert ".txt" in extracted.extensions.values()
        finally:
            Path(temp_path).unlink()

    def test_extract_assets_from_empty_values(self):
        """Test extracting assets with no values."""
        extracted = Packable.extract_assets()

        assert isinstance(extracted, ExtractedAssets)
        assert len(extracted.assets) == 0
        assert len(extracted.extensions) == 0

    def test_extract_assets_with_none_values(self):
        """Test extracting assets with None values."""
        extracted = Packable.extract_assets(None, None)

        assert isinstance(extracted, ExtractedAssets)
        assert len(extracted.assets) == 0
        assert len(extracted.extensions) == 0

    def test_extract_assets_from_tuple(self):
        """Test extracting assets from tuple."""
        data = (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))

        extracted = Packable.extract_assets(data)

        assert len(extracted.assets) == 2

    def test_extract_assets_preserves_extensions(self):
        """Test that extensions are correctly mapped to checksums."""
        with (
            tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f1,
            tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f2,
        ):
            f1.write(b"stl data")
            f2.write(b"json data")
            path1 = f1.name
            path2 = f2.name

        try:
            ref1 = ResourceRef(path=path1)
            ref2 = ResourceRef(path=path2)

            extracted = Packable.extract_assets(ref1, ref2)

            # Should have both files
            assert len(extracted.assets) == 2
            assert len(extracted.extensions) == 2

            # Check that extensions are correct
            checksums = list(extracted.assets.keys())
            extensions = [extracted.extensions[cs] for cs in checksums]
            assert ".stl" in extensions
            assert ".json" in extensions
        finally:
            Path(path1).unlink()
            Path(path2).unlink()

    def test_extract_assets_function_args_use_case(self):
        """Test the typical use case: extracting assets from function args.

        This simulates extracting assets from function arguments before remote execution.
        """
        with tempfile.NamedTemporaryFile(suffix=".mesh", delete=False) as f:
            f.write(b"mesh data")
            temp_path = f.name

        try:
            # Simulate function arguments
            def simulate(geometry: ResourceRef, initial_temp: np.ndarray, config: dict):
                pass

            # Create args
            geometry = ResourceRef(path=temp_path)
            initial_temp = np.array([300.0, 301.0, 302.0], dtype=np.float32)
            config = {"solver": "cfd", "timesteps": 100}

            # Extract all assets from args
            extracted = Packable.extract_assets(geometry, initial_temp, config)

            # Should have the geometry file and the temperature array
            assert len(extracted.assets) == 2
            # Should have the extension for geometry
            assert len(extracted.extensions) == 1
            assert extracted.extensions[list(extracted.extensions.keys())[0]] == ".mesh"
        finally:
            Path(temp_path).unlink()

    def test_extracted_assets_dataclass(self):
        """Test ExtractedAssets dataclass structure."""
        assets = {"abc123": b"data1", "def456": b"data2"}
        extensions = {"abc123": ".stl", "def456": ".json"}

        extracted = ExtractedAssets(assets=assets, extensions=extensions)

        assert extracted.assets == assets
        assert extracted.extensions == extensions
        assert isinstance(extracted.assets, dict)
        assert isinstance(extracted.extensions, dict)
