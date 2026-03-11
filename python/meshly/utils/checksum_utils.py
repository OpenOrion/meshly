"""Checksum utilities for hashing data, files, and directories."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import orjson
from pydantic import BaseModel

class ChecksumUtils:
    """Utility class for computing checksums."""

    # Thresholds for switching to fast checksum strategy
    LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB
    LARGE_DIR_FILE_COUNT_THRESHOLD = 100

    @staticmethod
    def compute_dict_checksum(data: Union[dict, BaseModel]) -> str:
        """SHA256 checksum computed from data and json_schema.
        
        Checksum Format:
            SHA256 of compact JSON: {"data":<data>,"json_schema":<schema>}
            Keys are sorted, no whitespace (single line).
        
        Why JSON-based:
            The data dict contains $ref entries pointing to asset checksums,
            so this checksum transitively covers all array/binary content.
            This format makes checksum recreation straightforward outside meshly:
            
                import hashlib, json
                payload = {"data": extracted_data, "json_schema": schema}
                compact_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
                checksum = hashlib.sha256(compact_json.encode()).hexdigest()
            
        Returns:
            SHA256 hex digest string
        """
        data_dict = data.model_dump() if isinstance(data, BaseModel) else data
        json_bytes = orjson.dumps(data_dict, option=orjson.OPT_SORT_KEYS)
        return ChecksumUtils.compute_bytes_checksum(json_bytes)

    @staticmethod
    def compute_bytes_checksum(data: bytes) -> str:
        """Compute SHA256 checksum for bytes.

        Args:
            data: Bytes to hash

        Returns:
            16-character hex string (SHA256)
        """
        return hashlib.sha256(data).hexdigest()[:16]



    @staticmethod
    def compute_file_checksum(file_path: Path, fast: bool = False) -> str:
        """Compute checksum of a file.

        Args:
            file_path: Path to the file
            fast: If True, use file metadata (size, mtime) instead of content hash
                  for large files. This is much faster but less accurate.

        Returns:
            Full SHA256 checksum string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_size = file_path.stat().st_size

        if fast and file_size > ChecksumUtils.LARGE_FILE_THRESHOLD:
            return ChecksumUtils._compute_file_metadata_checksum(file_path)

        return ChecksumUtils._compute_file_content_checksum(file_path)

    @staticmethod
    def _compute_file_content_checksum(file_path: Path) -> str:
        """Compute SHA256 checksum of file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _compute_file_metadata_checksum(file_path: Path) -> str:
        """Compute checksum based on file metadata (path, size, mtime)."""
        stat = file_path.stat()
        metadata = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.sha256(metadata.encode()).hexdigest()

    @staticmethod
    def compute_directory_checksum(dir_path: Path, fast: Optional[bool] = None) -> str:
        """Compute checksum of a directory.

        Args:
            dir_path: Path to the directory
            fast: If True, use file metadata instead of content hashes.
                  If None (default), automatically use fast strategy for large directories.

        Returns:
            Full SHA256 checksum string combining all file checksums
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        all_files = sorted(dir_path.rglob("*"))
        file_paths = [f for f in all_files if f.is_file()]

        if fast is None:
            fast = len(file_paths) > ChecksumUtils.LARGE_DIR_FILE_COUNT_THRESHOLD

        hasher = hashlib.sha256()

        for file_path in file_paths:
            rel_path = file_path.relative_to(dir_path)
            hasher.update(str(rel_path).encode())

            if fast:
                file_hash = ChecksumUtils._compute_file_metadata_checksum(file_path)
            else:
                file_hash = ChecksumUtils._compute_file_content_checksum(file_path)

            hasher.update(file_hash.encode())

        return hasher.hexdigest()

    @staticmethod
    def compute_path_checksum(path: Path, fast: Optional[bool] = None) -> str:
        """Compute checksum of a file or directory.

        Args:
            path: Path to file or directory
            fast: If True, use metadata-based checksums for speed.
                  If None, automatically use fast strategy for large files/directories.

        Returns:
            Full SHA256 checksum string
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            return ChecksumUtils.compute_file_checksum(
                path, fast=fast if fast is not None else False
            )
        elif path.is_dir():
            return ChecksumUtils.compute_directory_checksum(path, fast=fast)
        else:
            raise ValueError(f"Path is neither a file nor directory: {path}")
