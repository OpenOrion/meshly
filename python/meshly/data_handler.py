import stat
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Union
import zipfile
from io import BytesIO
from pathlib import Path
from abc import abstractmethod
from .common import PathLike

HandlerSource = Union[PathLike, BytesIO]

# Type for asset provider: either a dict or a callable that fetches by checksum
# Supports both sync and async fetch functions
# The callable can return None to indicate the asset should be read from cache
AssetFetcher = Callable[[str], Union[bytes, None, Awaitable[Optional[bytes]]]]
AssetProvider = Union[Dict[str, bytes], AssetFetcher, "CachedAssetLoader"]


@dataclass
class CachedAssetLoader:
    """Asset loader with optional disk cache for persistence.

    Wraps a callable asset fetcher with a DataHandler for caching.
    Fetched assets are stored as 'assets/{checksum}.bin' and read
    from cache on subsequent access.
    
    The fetch callable can return None to indicate the asset is not
    available from the remote source, in which case the loader will
    attempt to read from the cache. If not in cache either, a KeyError
    is raised.

    Example:
        def fetch_from_cloud(checksum: str) -> bytes | None:
            try:
                return cloud_storage.download(checksum)
            except NotFoundError:
                return None  # Will fallback to cache

        # Create loader with disk cache
        cache = DataHandler.create(Path("./cache"))
        loader = CachedAssetLoader(fetch_from_cloud, cache)

        lazy = Packable.reconstruct(SimulationCase, data, loader)
    """
    fetch: AssetFetcher
    """Callable that fetches asset bytes by checksum (can return None to use cache)"""
    cache: "DataHandler"
    """DataHandler for caching fetched assets"""""


class DataHandler:
    """Protocol for reading and writing files to various sources."""

    rel_path: str

    def resolved_path(self, subpath: PathLike) -> Path:
        """Resolve the path relative to the repository."""
        if str(subpath).startswith(self.rel_path):
            return Path(str(subpath))
        return Path(f"{self.rel_path}/{subpath}")

    def __init__(self, source: HandlerSource, rel_path=""):
        self.source = source
        self.rel_path = rel_path

    @abstractmethod
    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        raise NotImplementedError

    @abstractmethod
    def read_binary(self, subpath: PathLike) -> bytes:
        """Read binary content from a file."""
        raise NotImplementedError

    @abstractmethod
    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        """Write text content to a file."""
        raise NotImplementedError

    @abstractmethod
    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        """Write binary content to a file."""
        raise NotImplementedError

    @abstractmethod
    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]:
        """List files in the given subpath."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, subpath: PathLike) -> bool:
        """Check if a file exists."""
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, subpath: PathLike) -> None:
        """Remove a file."""
        raise NotImplementedError

    def to_path(self, rel_path: str) -> "DataHandler":
        """Get a handler with a nested relative path."""
        return DataHandler.create(self.source, f"{self.rel_path}/{rel_path}" if self.rel_path != "" else rel_path, self)

    @staticmethod
    def create(source: HandlerSource, rel_path="", existing_handler: Optional["DataHandler"] = None):
        """Create an appropriate handler based on the source type.

        Args:
            source: Path to file/directory or BytesIO object
            rel_path: Relative path prefix for file operations
            existing_handler: Optional existing handler to reuse resources from

        Returns:
            Handler implementation
        """
        if isinstance(source, BytesIO):
            return ZipHandler(
                source,
                rel_path,
                zip_file=existing_handler.zip_file if existing_handler and isinstance(
                    existing_handler, ZipHandler) else None
            )
        else:
            return FileHandler(source, rel_path)

    def finalize(self):
        """Close any resources if needed."""
        pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, calling finalize()."""
        self.finalize()
        return False


class FileHandler(DataHandler):
    """Handler for reading and writing files on the regular file system."""

    source: Path

    def __init__(self, source: PathLike, rel_path: str = ""):
        super().__init__(Path(source), rel_path)

    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        full_path = self.source / self.resolved_path(subpath)
        return full_path.read_text(encoding)

    def read_binary(self, subpath: PathLike) -> bytes:
        full_path = self.source / self.resolved_path(subpath)
        return full_path.read_bytes()

    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        full_path = self.source / self.resolved_path(subpath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        if executable:
            current_permissions = full_path.stat().st_mode
            full_path.chmod(current_permissions | stat.S_IXUSR |
                            stat.S_IXGRP | stat.S_IXOTH)

    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        if isinstance(content, BytesIO):
            content.seek(0)
            content = content.read()

        full_path = self.source / self.resolved_path(subpath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

        if executable:
            current_permissions = full_path.stat().st_mode
            full_path.chmod(current_permissions | stat.S_IXUSR |
                            stat.S_IXGRP | stat.S_IXOTH)

    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]:
        full_path = Path(self.source) / subpath
        if recursive:
            return [p.relative_to(self.source) for p in full_path.rglob("*")]
        else:
            return [p.relative_to(self.source) for p in full_path.glob("*")]

    def exists(self, subpath: PathLike) -> bool:
        full_path = self.source / self.resolved_path(subpath)
        return full_path.exists()

    def remove_file(self, subpath: PathLike) -> None:
        full_path = self.source / self.resolved_path(subpath)
        if full_path.exists():
            full_path.unlink()


class ZipHandler(DataHandler):
    """Handler for reading and writing files in zip archives."""

    # Fixed date_time for deterministic zip output (2020-01-01 00:00:00)
    DETERMINISTIC_DATE_TIME = (2020, 1, 1, 0, 0, 0)

    def __init__(self, source: Union[PathLike, BytesIO], rel_path: str = "", zip_file: Optional[zipfile.ZipFile] = None):
        super().__init__(source, rel_path)
        if zip_file is not None:
            self.zip_file: zipfile.ZipFile = zip_file
            self._mode = zip_file.mode
        else:
            # Try to open for reading first, fall back to writing if it's empty or doesn't exist
            try:
                self.zip_file = zipfile.ZipFile(source, "r")
                self._mode = "r"
            except (zipfile.BadZipFile, FileNotFoundError):
                self.zip_file = zipfile.ZipFile(source, "w")
                self._mode = "w"

    def _ensure_mode(self, required_mode: str):
        """Switch zip file mode if needed."""
        if self._mode != required_mode:
            self.zip_file.close()
            self.zip_file = zipfile.ZipFile(self.source, required_mode)
            self._mode = required_mode

    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        self._ensure_mode("r")
        return self.zip_file.read(str(self.resolved_path(subpath))).decode(encoding)

    def read_binary(self, subpath: PathLike) -> bytes:
        self._ensure_mode("r")
        return self.zip_file.read(str(self.resolved_path(subpath)))

    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        self._ensure_mode("w")
        zip_info = zipfile.ZipInfo(str(self.resolved_path(
            subpath)), date_time=self.DETERMINISTIC_DATE_TIME)
        if executable:
            zip_info.external_attr = 0o755 << 16
        self.zip_file.writestr(zip_info, content)

    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        self._ensure_mode("w")
        if isinstance(content, BytesIO):
            content.seek(0)
            content = content.read()

        zip_info = zipfile.ZipInfo(str(self.resolved_path(
            subpath)), date_time=self.DETERMINISTIC_DATE_TIME)
        if executable:
            zip_info.external_attr = 0o755 << 16
        self.zip_file.writestr(zip_info, content)

    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]:
        if subpath == "" or recursive:
            return [
                Path(p.filename)
                for p in self.zip_file.infolist()
                if p.filename.startswith(str(self.resolved_path(subpath)))
            ]
        else:
            return [
                Path(p.filename)
                for p in self.zip_file.infolist()
                if str(Path(p.filename).parent) == self.resolved_path(subpath)
            ]

    def exists(self, subpath: PathLike) -> bool:
        try:
            self.zip_file.getinfo(str(self.resolved_path(subpath)))
            return True
        except KeyError:
            return False

    def remove_file(self, subpath: PathLike) -> None:
        # Note: zipfile doesn't support removing files directly.
        # This would require recreating the zip without the file.
        raise NotImplementedError("ZipHandler does not support removing files")

    def finalize(self):
        """Close the zip file."""
        if hasattr(self, 'zip_file') and self.zip_file:
            self.zip_file.close()
