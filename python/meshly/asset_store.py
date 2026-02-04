"""Asset store for file-based Packable asset management.

This module provides an AssetStore class that manages binary assets on disk
by their checksum, enabling efficient caching and deduplication of serialized
Packable data.

Directory structure:
    assets_path/
        <checksum1>.bin
        <checksum2>.bin
        ...
    metadata_path/
        <path>/
            data.json
            schema.json

Example:
    from meshly import Packable, AssetStore
    
    # Create a store with separate paths for assets and metadata
    store = AssetStore(
        assets_path="/data/assets",
        metadata_path="/data/runs"
    )
    
    # Save a packable
    my_mesh.save(store, "abc123/result")
    
    # Load from store
    loaded = Mesh.load(store, "abc123/result")
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar

from meshly.common import PathLike
from meshly.constants import ExportConstants

if TYPE_CHECKING:
    from meshly.packable import ExtractedPackable, Packable

T = TypeVar("T", bound="Packable")


class AssetStore:
    """File-based asset store for Packable serialization.
    
    Assets (binary blobs) are stored by their SHA256 checksum, enabling deduplication.
    Metadata is stored at user-specified paths in a separate directory.
    
    Directory structure:
        assets_path/
            <checksum1>.bin
            <checksum2>.bin
        metadata_path/
            <path>/
                data.json
                schema.json
    
    Attributes:
        assets_path: Directory where binary assets are stored
        metadata_path: Directory where metadata is stored
    """
    
    DATA_FILE = "data.json"
    SCHEMA_FILE = "schema.json"
    
    def __init__(
        self,
        assets_path: PathLike,
        metadata_path: Optional[PathLike] = None,
    ):
        """Initialize an asset store.
        
        Args:
            assets_path: Directory for binary assets.
            metadata_path: Directory for metadata. If None, uses assets_path.
        """
        self.assets_path = Path(assets_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.assets_path
        self.assets_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def asset_file(self, checksum: str) -> Path:
        """Get the path for an asset by checksum.
        
        Args:
            checksum: Asset checksum
            
        Returns:
            Path to the asset file (with .bin extension)
        """
        return self.assets_path / f"{checksum}{ExportConstants.ASSET_EXT}"
    
    def metadata_dir(self, path: str) -> Path:
        """Get the metadata directory for a given path.
        
        Args:
            path: Relative path for the metadata
            
        Returns:
            Path to the metadata directory
        """
        return self.metadata_path / path
    
    def asset_exists(self, checksum: str) -> bool:
        """Check if an asset exists by checksum.
        
        Args:
            checksum: Asset checksum
            
        Returns:
            True if asset exists
        """
        return self.asset_file(checksum).exists()
    
    def save_asset(self, data: bytes, checksum: str) -> Path:
        """Save binary asset data by checksum.
        
        Args:
            data: Binary data to save
            checksum: Asset checksum identifier
            
        Returns:
            Path where asset was saved
        """
        path = self.asset_file(checksum)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return path
    
    def load_asset(self, checksum: str) -> bytes:
        """Load an asset by checksum.
        
        Args:
            checksum: Asset checksum
            
        Returns:
            Asset bytes
            
        Raises:
            FileNotFoundError: If asset doesn't exist
        """
        path = self.asset_file(checksum)
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {checksum}")
        return path.read_bytes()
    
    def load_data(self, path: str) -> Optional[dict]:
        """Load packable data by path.
        
        Args:
            path: Relative path for the metadata
            
        Returns:
            Data dict if found, None otherwise
        """
        data_path = self.metadata_dir(path) / self.DATA_FILE
        if not data_path.exists():
            return None
        return json.loads(data_path.read_text())
    
    def load_schema(self, path: str) -> Optional[dict]:
        """Load packable schema by path.
        
        Args:
            path: Relative path for the metadata
            
        Returns:
            Schema dict if found, None otherwise
        """
        schema_path = self.metadata_dir(path) / self.SCHEMA_FILE
        if not schema_path.exists():
            return None
        return json.loads(schema_path.read_text())
    
    def exists(self, path: str) -> bool:
        """Check if a packable exists at path.
        
        Args:
            path: Relative path for the metadata
            
        Returns:
            True if packable data exists
        """
        return (self.metadata_dir(path) / self.DATA_FILE).exists()
    
    def __repr__(self) -> str:
        return f"AssetStore(assets_path={self.assets_path!r}, metadata_path={self.metadata_path!r})"
