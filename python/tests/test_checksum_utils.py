"""Tests for ChecksumUtils."""

import shutil
import tempfile
from pathlib import Path

import pytest
from meshly.utils.checksum_utils import ChecksumUtils


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def test_file(temp_dir):
    """Create a simple test file."""
    f = temp_dir / "test_file.txt"
    f.write_text("Hello, World!")
    return f


@pytest.fixture
def test_subdir(temp_dir):
    """Create a test directory with multiple files."""
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "file1.txt").write_text("Content 1")
    (subdir / "file2.txt").write_text("Content 2")

    nested = subdir / "nested"
    nested.mkdir()
    (nested / "file3.txt").write_text("Content 3")

    return subdir


class TestFileChecksum:
    """Tests for file checksum computation."""

    def test_returns_string(self, test_file):
        """Test that file checksum returns a hex string."""
        result = ChecksumUtils.compute_file_checksum(test_file)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64 hex chars

    def test_is_deterministic(self, test_file):
        """Test that same file produces same checksum."""
        result1 = ChecksumUtils.compute_file_checksum(test_file)
        result2 = ChecksumUtils.compute_file_checksum(test_file)
        assert result1 == result2

    def test_differs_for_different_content(self, temp_dir):
        """Test that different content produces different checksum."""
        file1 = temp_dir / "a.txt"
        file2 = temp_dir / "b.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        assert ChecksumUtils.compute_file_checksum(file1) != ChecksumUtils.compute_file_checksum(file2)

    def test_not_found_raises(self, temp_dir):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChecksumUtils.compute_file_checksum(temp_dir / "nonexistent.txt")

    def test_on_directory_raises_error(self, test_subdir):
        """Test that passing a directory raises ValueError."""
        with pytest.raises(ValueError):
            ChecksumUtils.compute_file_checksum(test_subdir)

    def test_fast_mode_uses_metadata(self, temp_dir):
        """Test that fast mode produces valid checksum."""
        large_file = temp_dir / "large.txt"
        large_file.write_text("Some content")

        result_fast = ChecksumUtils.compute_file_checksum(large_file, fast=True)
        result_normal = ChecksumUtils.compute_file_checksum(large_file, fast=False)

        assert isinstance(result_fast, str)
        assert len(result_fast) == 64
        # For small files, fast=True still uses content hash
        assert result_fast == result_normal


class TestDirectoryChecksum:
    """Tests for directory checksum computation."""

    def test_returns_string(self, test_subdir):
        """Test that directory checksum returns a hex string."""
        result = ChecksumUtils.compute_directory_checksum(test_subdir)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_is_deterministic(self, test_subdir):
        """Test that same directory produces same checksum."""
        result1 = ChecksumUtils.compute_directory_checksum(test_subdir)
        result2 = ChecksumUtils.compute_directory_checksum(test_subdir)
        assert result1 == result2

    def test_changes_with_content(self, test_subdir):
        """Test that modifying a file changes directory checksum."""
        checksum_before = ChecksumUtils.compute_directory_checksum(test_subdir)
        (test_subdir / "file1.txt").write_text("Modified content")
        checksum_after = ChecksumUtils.compute_directory_checksum(test_subdir)
        assert checksum_before != checksum_after

    def test_not_found_raises(self, temp_dir):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChecksumUtils.compute_directory_checksum(temp_dir / "nonexistent_dir")

    def test_on_file_raises_error(self, test_file):
        """Test that passing a file raises ValueError."""
        with pytest.raises(ValueError):
            ChecksumUtils.compute_directory_checksum(test_file)

    def test_fast_mode(self, test_subdir):
        """Test that fast mode works for directories."""
        result_fast = ChecksumUtils.compute_directory_checksum(test_subdir, fast=True)
        assert isinstance(result_fast, str)
        assert len(result_fast) == 64

    def test_empty_directory(self, temp_dir):
        """Test checksum of an empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = ChecksumUtils.compute_directory_checksum(empty_dir)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_includes_structure(self, temp_dir):
        """Test that directory structure affects checksum."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "a.txt").write_text("content")
        (dir2 / "b.txt").write_text("content")  # Same content, different name

        assert ChecksumUtils.compute_directory_checksum(dir1) != ChecksumUtils.compute_directory_checksum(dir2)


class TestPathChecksum:
    """Tests for unified path checksum."""

    def test_file(self, test_file):
        """Test that compute_path_checksum works for files."""
        result = ChecksumUtils.compute_path_checksum(test_file)
        assert result == ChecksumUtils.compute_file_checksum(test_file)

    def test_directory(self, test_subdir):
        """Test that compute_path_checksum works for directories."""
        result = ChecksumUtils.compute_path_checksum(test_subdir)
        assert result == ChecksumUtils.compute_directory_checksum(test_subdir)

    def test_not_found_raises(self, temp_dir):
        """Test that missing path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChecksumUtils.compute_path_checksum(temp_dir / "nonexistent")
