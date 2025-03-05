"""
Tests for the arrayutils module.
"""
import unittest
import numpy as np
from pymeshoptimizer.arrayutils import encode_array, decode_array

class TestArrayUtils(unittest.TestCase):
    """Test cases for the arrayutils module."""
    
    def setUp(self):
        """Set up test data."""
        # Create test arrays
        self.array_1d = np.linspace(0, 10, 100, dtype=np.float32)
        self.array_2d = np.random.random((50, 3)).astype(np.float32)
        self.array_3d = np.random.random((10, 10, 10)).astype(np.float32)
        self.array_int = np.random.randint(0, 100, (20, 20), dtype=np.int32)
    
    def test_encode_decode_array_1d(self):
        """Test encoding and decoding a 1D array."""
        encoded = encode_array(self.array_1d)
        decoded = decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_1d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_1d.nbytes)
        
        # Print compression ratio
        print(f"1D array compression ratio: {len(encoded.data) / self.array_1d.nbytes:.2f}")
    
    def test_encode_decode_array_2d(self):
        """Test encoding and decoding a 2D array."""
        encoded = encode_array(self.array_2d)
        decoded = decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_2d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_2d.nbytes)
        
        # Print compression ratio
        print(f"2D array compression ratio: {len(encoded.data) / self.array_2d.nbytes:.2f}")
    
    def test_encode_decode_array_3d(self):
        """Test encoding and decoding a 3D array."""
        encoded = encode_array(self.array_3d)
        decoded = decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_3d, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_3d.nbytes)
        
        # Print compression ratio
        print(f"3D array compression ratio: {len(encoded.data) / self.array_3d.nbytes:.2f}")
    
    def test_encode_decode_array_int(self):
        """Test encoding and decoding an integer array."""
        encoded = encode_array(self.array_int)
        decoded = decode_array(encoded)
        
        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_int, rtol=1e-5)
        
        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_int.nbytes)
        
        # Print compression ratio
        print(f"Integer array compression ratio: {len(encoded.data) / self.array_int.nbytes:.2f}")

if __name__ == "__main__":
    unittest.main()