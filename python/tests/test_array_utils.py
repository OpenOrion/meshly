"""
Tests for the array utility functions.
"""
import numpy as np
import pytest
from meshly import ArrayUtils


class TestArrayUtils:
    """Test cases for the array utility functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.array_1d = np.linspace(0, 10, 100, dtype=np.float32)
        self.array_2d = np.random.random((50, 3)).astype(np.float32)
        self.array_3d = np.random.random((10, 10, 10)).astype(np.float32)
        self.array_int = np.random.randint(0, 100, (20, 20), dtype=np.int32)

    def test_encode_decode_array_1d(self):
        """Test encoding and decoding a 1D array."""
        encoded = ArrayUtils.encode_array(self.array_1d)
        decoded = ArrayUtils.decode_array(encoded)

        np.testing.assert_allclose(decoded, self.array_1d, rtol=1e-5)
        assert len(encoded.data) < self.array_1d.nbytes

    def test_encode_decode_array_2d(self):
        """Test encoding and decoding a 2D array."""
        encoded = ArrayUtils.encode_array(self.array_2d)
        decoded = ArrayUtils.decode_array(encoded)

        np.testing.assert_allclose(decoded, self.array_2d, rtol=1e-5)
        assert len(encoded.data) < self.array_2d.nbytes

    def test_encode_decode_array_3d(self):
        """Test encoding and decoding a 3D array."""
        encoded = ArrayUtils.encode_array(self.array_3d)
        decoded = ArrayUtils.decode_array(encoded)

        np.testing.assert_allclose(decoded, self.array_3d, rtol=1e-5)
        assert len(encoded.data) < self.array_3d.nbytes

    def test_encode_decode_array_int(self):
        """Test encoding and decoding an integer array."""
        encoded = ArrayUtils.encode_array(self.array_int)
        decoded = ArrayUtils.decode_array(encoded)

        np.testing.assert_allclose(decoded, self.array_int, rtol=1e-5)
        assert len(encoded.data) < self.array_int.nbytes

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4), (2, 2, 2, 2)])
    def test_encode_decode_preserves_shape(self, shape):
        """Test that encoding and decoding preserves array shape."""
        array = np.random.random(shape).astype(np.float32)
        encoded = ArrayUtils.encode_array(array)
        decoded = ArrayUtils.decode_array(encoded)

        assert decoded.shape == shape
        np.testing.assert_allclose(decoded, array, rtol=1e-5)

    def test_encode_decode_different_dtypes(self):
        """Test encoding and decoding arrays with different data types."""
        test_arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.random.random((5, 5)).astype(np.float32),
        ]

        for test_array in test_arrays:
            encoded = ArrayUtils.encode_array(test_array)
            decoded = ArrayUtils.decode_array(encoded)

            np.testing.assert_allclose(decoded, test_array, rtol=1e-5)
            assert decoded.shape == test_array.shape
            assert decoded.dtype == test_array.dtype
