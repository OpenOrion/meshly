"""
Tests for the array utility functions.
"""
import unittest
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
        print(
            f"1D array compression ratio: {len(encoded.data) / self.array_1d.nbytes:.2f}")

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
        print(
            f"3D array compression ratio: {len(encoded.data) / self.array_3d.nbytes:.2f}")

    def test_encode_decode_array_int(self):
        """Test encoding and decoding an integer array."""
        encoded = ArrayUtils.encode_array(self.array_int)
        decoded = ArrayUtils.decode_array(encoded)

        # Check that the decoded array matches the original
        np.testing.assert_allclose(decoded, self.array_int, rtol=1e-5)

        # Check that the encoded data is smaller than the original
        self.assertLess(len(encoded.data), self.array_int.nbytes)

        # Print compression ratio
        print(
            f"Integer array compression ratio: {len(encoded.data) / self.array_int.nbytes:.2f}")

    def test_encode_decode_preserves_shape(self):
        """Test that encoding and decoding preserves array shape."""
        test_shapes = [(10,), (5, 5), (2, 3, 4), (2, 2, 2, 2)]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                array = np.random.random(shape).astype(np.float32)
                encoded = ArrayUtils.encode_array(array)
                decoded = ArrayUtils.decode_array(encoded)

                self.assertEqual(decoded.shape, shape)
                np.testing.assert_allclose(decoded, array, rtol=1e-5)

    def test_encode_decode_different_dtypes(self):
        """Test encoding and decoding arrays with different data types."""
        test_arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.random.random((5, 5)).astype(np.float32),
        ]

        for i, test_array in enumerate(test_arrays):
            with self.subTest(array_index=i, dtype=test_array.dtype):
                encoded = ArrayUtils.encode_array(test_array)
                decoded = ArrayUtils.decode_array(encoded)

                np.testing.assert_allclose(decoded, test_array, rtol=1e-5)
                self.assertEqual(decoded.shape, test_array.shape)
                self.assertEqual(decoded.dtype, test_array.dtype)


if __name__ == "__main__":
    unittest.main()
