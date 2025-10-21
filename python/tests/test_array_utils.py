"""
Tests for the array utility functions.
"""
import unittest
import tempfile
import os
from io import BytesIO
import numpy as np
from meshly import ArrayUtils, ArrayResult
from pydantic import BaseModel

class TestArrayUtils(unittest.TestCase):
    """Test cases for the array utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create test arrays
        self.array_1d = np.linspace(0, 10, 100, dtype=np.float32)
        self.array_2d = np.random.random((50, 3)).astype(np.float32)
        self.array_3d = np.random.random((10, 10, 10)).astype(np.float32)
        self.array_int = np.random.randint(0, 100, (20, 20), dtype=np.int32)
    
    def test_encode_decode_array_1d(self):
        """Test encoding and decoding a 1D array."""
        encoded = ArrayUtils.encode_array(self.array_1d)
        decoded = ArrayUtils.decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_1d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_1d.nbytes)
        
        # Print compression ratio
        print(f"1D array compression ratio: {len(encoded.data) / self.array_1d.nbytes:.2f}")
    
    def test_encode_decode_array_2d(self):
        """Test encoding and decoding a 2D array."""
        encoded = ArrayUtils.encode_array(self.array_2d)
        decoded = ArrayUtils.decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_2d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_2d.nbytes)
        
    
    def test_encode_decode_array_3d(self):
        """Test encoding and decoding a 3D array."""
        encoded = ArrayUtils.encode_array(self.array_3d)
        decoded = ArrayUtils.decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_3d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_3d.nbytes)
        
        # Print compression ratio
        print(f"3D array compression ratio: {len(encoded.data) / self.array_3d.nbytes:.2f}")
    
    def test_encode_decode_array_int(self):
        """Test encoding and decoding an integer array."""
        encoded = ArrayUtils.encode_array(self.array_int)
        decoded = ArrayUtils.decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_int, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_int.nbytes)
        
        # Print compression ratio
        print(f"Integer array compression ratio: {len(encoded.data) / self.array_int.nbytes:.2f}")

    def test_save_load_array_to_zip_file(self):
        """Test saving and loading an array to/from a zip file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save array to zip file
            ArrayUtils.save_to_zip(self.array_2d, temp_path)
            
            # Load array from zip file
            result = ArrayUtils.load_from_zip(temp_path)
            
            # Check that the loaded array matches the original
            np.testing.assert_allclose(result.array, self.array_2d, rtol=1e-5)
            self.assertEqual(result.array.shape, self.array_2d.shape)
            self.assertEqual(result.array.dtype, self.array_2d.dtype)
            
            print(f"Array successfully saved and loaded from zip file: {temp_path}")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_load_array_to_zip_bytesio(self):
        """Test saving and loading an array to/from a zip file using BytesIO."""
        # Create a BytesIO buffer
        buffer = BytesIO()
        
        # Save array to zip buffer
        ArrayUtils.save_to_zip(self.array_3d, buffer)
        
        # Reset buffer position for reading
        buffer.seek(0)
        
        # Load array from zip buffer
        result = ArrayUtils.load_from_zip(buffer)
        
        # Check that the loaded array matches the original
        np.testing.assert_allclose(result.array, self.array_3d, rtol=1e-5)
        self.assertEqual(result.array.shape, self.array_3d.shape)
        self.assertEqual(result.array.dtype, self.array_3d.dtype)
        
        print(f"Array successfully saved and loaded from zip BytesIO buffer")

    def test_save_load_array_different_dtypes(self):
        """Test saving and loading arrays with different data types."""
        test_arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.random.random((5, 5)).astype(np.float32),
        ]
        
        for i, test_array in enumerate(test_arrays):
            with self.subTest(array_index=i, dtype=test_array.dtype):
                buffer = BytesIO()
                
                # Save and load the array
                ArrayUtils.save_to_zip(test_array, buffer)
                buffer.seek(0)
                result = ArrayUtils.load_from_zip(buffer)
                
                # Check that the loaded array matches the original
                np.testing.assert_allclose(result.array, test_array, rtol=1e-5)
                self.assertEqual(result.array.shape, test_array.shape)
                self.assertEqual(result.array.dtype, test_array.dtype)

    def test_custom_metadata(self):
        """Test saving and loading arrays with custom metadata."""
        # Define a custom metadata class
        class SensorMetadata(BaseModel):
            sensor_id: str
            timestamp: float
            location: tuple
            calibration_data: dict
        
        # Create test array and metadata
        test_array = np.random.random((10, 3)).astype(np.float32)
        custom_metadata = SensorMetadata(
            sensor_id="sensor_001",
            timestamp=1234567890.123,
            location=(37.7749, -122.4194, 100.0),
            calibration_data={"offset": [0.1, 0.2, 0.3], "scale": [1.0, 1.1, 0.9]}
        )
        
        # Test with BytesIO
        buffer = BytesIO()
        ArrayUtils.save_to_zip(test_array, buffer, custom_metadata=custom_metadata)
        buffer.seek(0)
        
        # Load with metadata
        result = ArrayUtils.load_from_zip(buffer, SensorMetadata)
        
        # Verify result is ArrayResult
        self.assertIsInstance(result, ArrayResult)
        
        # Verify array
        np.testing.assert_allclose(result.array, test_array, rtol=1e-5)
        self.assertEqual(result.array.shape, test_array.shape)
        self.assertEqual(result.array.dtype, test_array.dtype)
        
        # Verify metadata
        self.assertIsInstance(result.custom_metadata, SensorMetadata)
        self.assertEqual(result.custom_metadata.sensor_id, "sensor_001")
        self.assertEqual(result.custom_metadata.timestamp, 1234567890.123)
        self.assertEqual(result.custom_metadata.location, (37.7749, -122.4194, 100.0))
        self.assertEqual(result.custom_metadata.calibration_data, {"offset": [0.1, 0.2, 0.3], "scale": [1.0, 1.1, 0.9]})

    def test_custom_metadata_with_class(self):
        """Test loading custom metadata with the specified class."""
        # Define a custom metadata class
        class DeviceInfo(BaseModel):
            device_name: str
            firmware_version: str
            settings: dict
        
        # Create test array and metadata
        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        custom_metadata = DeviceInfo(
            device_name="Test Device",
            firmware_version="1.2.3",
            settings={"mode": "auto", "sensitivity": 0.8}
        )
        
        # Save with custom metadata
        buffer = BytesIO()
        ArrayUtils.save_to_zip(test_array, buffer, custom_metadata=custom_metadata)
        buffer.seek(0)
        
        # Load with metadata class to get both array and metadata
        result = ArrayUtils.load_from_zip(buffer, DeviceInfo)
        
        # Verify result is ArrayResult
        self.assertIsInstance(result, ArrayResult)
        
        # Verify array
        np.testing.assert_allclose(result.array, test_array, rtol=1e-5)
        
        # Verify metadata
        self.assertIsInstance(result.custom_metadata, DeviceInfo)
        self.assertEqual(result.custom_metadata.device_name, "Test Device")
        self.assertEqual(result.custom_metadata.firmware_version, "1.2.3")
        self.assertEqual(result.custom_metadata.settings, {"mode": "auto", "sensitivity": 0.8})

    def test_array_without_custom_metadata(self):
        """Test that arrays saved without custom metadata still work with the new load function."""
        test_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        
        # Save without custom metadata (using the regular method)
        buffer = BytesIO()
        ArrayUtils.save_to_zip(test_array, buffer)
        buffer.seek(0)
        
        # Load with the regular function (no metadata class)
        result = ArrayUtils.load_from_zip(buffer)
        
        # Verify result is ArrayResult with no custom metadata
        self.assertIsInstance(result, ArrayResult)
        self.assertIsNone(result.custom_metadata)
        
        # Verify array
        np.testing.assert_allclose(result.array, test_array, rtol=1e-5)

    def test_custom_metadata_file_operations(self):
        """Test custom metadata with actual file operations."""
        # Define a custom metadata class
        class ExperimentMetadata(BaseModel):
            experiment_id: str
            researcher: str
            date: str
            parameters: dict
        
        # Create test data
        test_array = np.linspace(0, 1, 100).astype(np.float32)
        custom_metadata = ExperimentMetadata(
            experiment_id="EXP-2024-001",
            researcher="Dr. Smith",
            date="2024-01-15",
            parameters={"temperature": 25.0, "pressure": 1013.25, "humidity": 45}
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save with custom metadata
            ArrayUtils.save_to_zip(test_array, temp_path, custom_metadata=custom_metadata)
            
            # Load back
            result = ArrayUtils.load_from_zip(temp_path, ExperimentMetadata)
            
            # Verify result is ArrayResult
            self.assertIsInstance(result, ArrayResult)
            
            # Verify array
            np.testing.assert_allclose(result.array, test_array, rtol=1e-5)
            
            # Verify metadata
            self.assertIsInstance(result.custom_metadata, ExperimentMetadata)
            self.assertEqual(result.custom_metadata.experiment_id, "EXP-2024-001")
            self.assertEqual(result.custom_metadata.researcher, "Dr. Smith")
            self.assertEqual(result.custom_metadata.date, "2024-01-15")
            self.assertEqual(result.custom_metadata.parameters["temperature"], 25.0)
            
            print(f"Successfully saved and loaded array with custom metadata from {temp_path}")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    unittest.main()