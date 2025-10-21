"""
Tests for the array utility functions.
"""
import unittest
import tempfile
import os
from io import BytesIO
import numpy as np
from meshly import ArrayUtils

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
            loaded_array = ArrayUtils.load_from_zip(temp_path)
            
            # Check that the loaded array matches the original
            np.testing.assert_allclose(loaded_array, self.array_2d, rtol=1e-5)
            self.assertEqual(loaded_array.shape, self.array_2d.shape)
            self.assertEqual(loaded_array.dtype, self.array_2d.dtype)
            
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
        loaded_array = ArrayUtils.load_from_zip(buffer)
        
        # Check that the loaded array matches the original
        np.testing.assert_allclose(loaded_array, self.array_3d, rtol=1e-5)
        self.assertEqual(loaded_array.shape, self.array_3d.shape)
        self.assertEqual(loaded_array.dtype, self.array_3d.dtype)
        
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
                loaded_array = ArrayUtils.load_from_zip(buffer)
                
                # Check that the loaded array matches the original
                np.testing.assert_allclose(loaded_array, test_array, rtol=1e-5)
                self.assertEqual(loaded_array.shape, test_array.shape)
                self.assertEqual(loaded_array.dtype, test_array.dtype)

if __name__ == "__main__":
    unittest.main()