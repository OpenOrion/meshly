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

    def test_extract_reconstruct_array_1d(self):
        """Test extracting and reconstructing a 1D array."""
        extracted = ArrayUtils.extract(self.array_1d)
        reconstructed = ArrayUtils.reconstruct(extracted)

        np.testing.assert_allclose(reconstructed, self.array_1d, rtol=1e-5)
        assert len(extracted.data) < self.array_1d.nbytes

    def test_extract_reconstruct_array_2d(self):
        """Test extracting and reconstructing a 2D array."""
        extracted = ArrayUtils.extract(self.array_2d)
        reconstructed = ArrayUtils.reconstruct(extracted)

        np.testing.assert_allclose(reconstructed, self.array_2d, rtol=1e-5)
        assert len(extracted.data) < self.array_2d.nbytes

    def test_extract_reconstruct_array_3d(self):
        """Test extracting and reconstructing a 3D array."""
        extracted = ArrayUtils.extract(self.array_3d)
        reconstructed = ArrayUtils.reconstruct(extracted)

        np.testing.assert_allclose(reconstructed, self.array_3d, rtol=1e-5)
        assert len(extracted.data) < self.array_3d.nbytes

    def test_extract_reconstruct_array_int(self):
        """Test extracting and reconstructing an integer array."""
        extracted = ArrayUtils.extract(self.array_int)
        reconstructed = ArrayUtils.reconstruct(extracted)

        np.testing.assert_allclose(reconstructed, self.array_int, rtol=1e-5)
        assert len(extracted.data) < self.array_int.nbytes

    @pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4), (2, 2, 2, 2)])
    def test_extract_reconstruct_preserves_shape(self, shape):
        """Test that extracting and reconstructing preserves array shape."""
        array = np.random.random(shape).astype(np.float32)
        extracted = ArrayUtils.extract(array)
        reconstructed = ArrayUtils.reconstruct(extracted)

        assert reconstructed.shape == shape
        np.testing.assert_allclose(reconstructed, array, rtol=1e-5)

    def test_extract_reconstruct_different_dtypes(self):
        """Test extracting and reconstructing arrays with different data types."""
        test_arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            np.random.random((5, 5)).astype(np.float32),
        ]

        for test_array in test_arrays:
            extracted = ArrayUtils.extract(test_array)
            reconstructed = ArrayUtils.reconstruct(extracted)

            np.testing.assert_allclose(reconstructed, test_array, rtol=1e-5)
            assert reconstructed.shape == test_array.shape
            assert reconstructed.dtype == test_array.dtype
