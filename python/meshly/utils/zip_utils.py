"""
Utility functions for zip file operations used by Packable and Mesh.
"""

import json
import zipfile
from typing import Dict, List, Optional, Any

import numpy as np

from ..array import ArrayUtils, ArrayMetadata, EncodedArray

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False


class ZipUtils:
    """Static utility methods for zip file operations."""

    @staticmethod
    def write_files(
        zipf: zipfile.ZipFile,
        files_to_write: List[tuple],
        date_time: Optional[tuple] = None
    ) -> None:
        """
        Write files to an open zip file.

        Args:
            zipf: Open ZipFile object
            files_to_write: List of (filename, data) tuples
            date_time: Optional date_time tuple for deterministic output
        """
        for filename, data_bytes in sorted(files_to_write):
            if date_time is not None:
                info = zipfile.ZipInfo(filename=filename, date_time=date_time)
            else:
                info = zipfile.ZipInfo(filename=filename)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            if isinstance(data_bytes, str):
                data_bytes = data_bytes.encode('utf-8')
            zipf.writestr(info, data_bytes)

    @staticmethod
    def decode_arrays(
        encoded_arrays: Dict[str, EncodedArray],
        use_jax: bool = False
    ) -> Dict[str, Any]:
        """
        Decode encoded arrays into proper structure.

        Handles both flat arrays (e.g., "vertices") and nested arrays
        (e.g., "markerIndices.boundary" becomes {"markerIndices": {"boundary": ...}})

        Args:
            encoded_arrays: Dict of encoded arrays keyed by dotted path
            use_jax: If True, decode as JAX arrays

        Returns:
            Decoded arrays organized into proper structure
        """
        result: Dict[str, Any] = {}

        for key, encoded in encoded_arrays.items():
            decoded = ArrayUtils.decode_array(encoded)
            if use_jax and HAS_JAX:
                decoded = jnp.array(decoded)

            if "." in key:
                # Nested array - build nested structure
                parts = key.split(".")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = decoded
            else:
                # Flat array
                result[key] = decoded

        return result

    @staticmethod
    def load_arrays(zipf: zipfile.ZipFile, use_jax: bool = False) -> Dict[str, Any]:
        """
        Load and decode all arrays from a zip file's arrays/ folder.

        Handles both flat arrays (e.g., "vertices") and nested arrays
        (e.g., "markerIndices/boundary" becomes {"markerIndices": {"boundary": ...}})

        Args:
            zipf: Open ZipFile object
            use_jax: If True, decode as JAX arrays

        Returns:
            Decoded arrays organized into proper structure
        """
        # Load encoded arrays from zip
        encoded_arrays: Dict[str, EncodedArray] = {}

        for array_file in zipf.namelist():
            if not (array_file.startswith("arrays/") and array_file.endswith("/array.bin")):
                continue

            # Extract array path: "arrays/markerIndices/boundary/array.bin" -> "markerIndices.boundary"
            array_path = array_file[7:-10]  # Remove "arrays/" and "/array.bin"
            key = array_path.replace("/", ".")

            # Load metadata and data
            with zipf.open(f"arrays/{array_path}/metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                metadata = ArrayMetadata(**metadata_dict)

            with zipf.open(array_file) as f:
                encoded_bytes = f.read()

            encoded_arrays[key] = EncodedArray(
                data=encoded_bytes,
                shape=tuple(metadata.shape),
                dtype=np.dtype(metadata.dtype),
                itemsize=metadata.itemsize,
            )

        # Decode all arrays
        return ZipUtils.decode_arrays(encoded_arrays, use_jax)
