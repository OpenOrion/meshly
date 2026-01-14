import gzip
import stat
from typing import Callable, List, Optional, Union
import zipfile
from io import BytesIO
from pathlib import Path
from abc import abstractmethod

from .common import PathLike


ZipBuffer = BytesIO

# Type aliases for cache callbacks
CacheLoader = Callable[[str], Optional[bytes]]
"""Load cached packable by SHA256 hash. Returns bytes or None if not found."""

CacheSaver = Callable[[str, bytes], None]
"""Save packable bytes to cache with SHA256 hash as key."""

ReadHandlerSource = Union[PathLike, ZipBuffer]
WriteHandlerDestination = Union[PathLike, ZipBuffer]


class DataHandler:
    rel_path: str

    def resolved_path(self, subpath: PathLike) -> Path:
        """Resolve the path relative to the repository."""
        if str(subpath).startswith(self.rel_path):
            return Path(str(subpath))
        return Path(f"{self.rel_path}/{subpath}")


class ReadHandler(DataHandler):
    """Protocol for reading files from various sources."""

    def __init__(self, source: ReadHandlerSource, rel_path=""):
        self.source = source
        self.rel_path = rel_path

    @abstractmethod
    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        ...

    @abstractmethod
    def read_binary(self, subpath: PathLike) -> bytes:
        """Read binary content from a file."""
        ...

    @abstractmethod
    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]:
        """List files in the given subpath."""
        ...

    def to_path(self, rel_path: str):
        """Get the original source as a PathLike if applicable."""
        return ReadHandler.create_handler(self.source, f"{self.rel_path}/{rel_path}" if self.rel_path != "" else rel_path, self)

    @staticmethod
    def create_handler(source: ReadHandlerSource, rel_path="", existing_handler: Optional["ReadHandler"] = None):
        """
        Create an appropriate read handler based on the source type.

        Args:
            source: Path to file/directory or BytesIO object
            rel_path: Relative path prefix for file operations
            existing_handler: Optional existing handler to reuse resources from

        Returns:
            ReadHandler implementation
        """
        if isinstance(source, ZipBuffer):
            return ZipReadHandler(
                source,
                rel_path,
                zip_file=existing_handler.zip_file if existing_handler and isinstance(
                    existing_handler, ZipReadHandler) else None
            )
        else:
            return FileReadHandler(source, rel_path)

    @staticmethod
    def create_cache_loader(source: ReadHandlerSource):
        """
        Create a CacheLoader function that reads from a handler source.

        Args:
            source: Path to cache directory or ZipBuffer containing cached packables

        Returns:
            CacheLoader function: (hash: str) -> Optional[bytes]

        Example:
            cache_loader = ReadHandler.create_cache_loader("/path/to/cache")
            mesh = Mesh.load_from_zip("mesh.zip", cache_loader=cache_loader)
        """
        handler = ReadHandler.create_handler(source)

        def loader(hash_digest: str) -> Optional[bytes]:
            try:
                return handler.read_binary(f"{hash_digest}.zip")
            except (FileNotFoundError, KeyError):
                return None

        return loader


class WriteHandler(DataHandler):
    """Protocol for writing files to various destinations."""

    def __init__(self, destination: WriteHandlerDestination, rel_path=""):
        self.destination = destination
        self.rel_path = rel_path

    @abstractmethod
    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        """Write text content to a file."""
        ...

    @abstractmethod
    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        """Write binary content to a file."""
        ...

    def to_path(self, rel_path: str):
        """Get the original source as a PathLike if applicable."""
        return WriteHandler.create_handler(self.destination, f"{self.rel_path}/{rel_path}" if self.rel_path != "" else rel_path, self)

    @staticmethod
    def create_handler(destination: WriteHandlerDestination, rel_path: str = "", existing_handler: Optional["WriteHandler"] = None):
        """
        Create an appropriate write handler based on the destination type.

        Args:
            destination: Path to file/directory or BytesIO object
            is_zip: Whether to create a zip file

        Returns:
            WriteFileHandlerProtocol implementation
        """
        if isinstance(destination, ZipBuffer):
            return ZipWriteHandler(
                destination,
                rel_path,
                zip_file=existing_handler.zip_file if existing_handler and isinstance(
                    existing_handler, ZipWriteHandler) else None
            )
        else:
            return FileWriteHandler(destination, rel_path)

    @staticmethod
    def create_cache_saver(destination: WriteHandlerDestination):
        """
        Create a CacheSaver function that writes to a handler destination.

        Args:
            destination: Path to cache directory or ZipBuffer for cached packables

        Returns:
            CacheSaver function: (hash: str, data: bytes) -> None

        Example:
            cache_saver = WriteHandler.create_cache_saver("/path/to/cache")
            mesh.save_to_zip("mesh.zip", cache_saver=cache_saver)
        """
        handler = WriteHandler.create_handler(destination)
        written_hashes: set = set()

        def saver(hash_digest: str, data: bytes) -> None:
            if hash_digest not in written_hashes:
                handler.write_binary(f"{hash_digest}.zip", data)
                written_hashes.add(hash_digest)

        return saver

    def finalize(self):
        """Close any resources if needed."""
        pass


class FileReadHandler(ReadHandler):
    """Handler for reading files from the regular file system."""

    source: Path

    def __init__(self, source: PathLike, rel_path: str = ""):
        super().__init__(Path(source), rel_path)

    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        full_path = self.source / self.resolved_path(subpath)
        return full_path.read_text(encoding)

    def read_binary(self, subpath: PathLike) -> bytes:
        full_path = self.source / self.resolved_path(subpath)
        return full_path.read_bytes()

    def list_files(self, subpath: PathLike = "", recursive: bool = False) -> List[Path]:
        full_path = Path(self.source) / subpath
        if recursive:
            return [p.relative_to(self.source) for p in full_path.rglob("*")]
        else:
            return [p.relative_to(self.source) for p in full_path.glob("*")]


class ZipReadHandler(ReadHandler):
    """Handler for reading files from zip archives."""

    def __init__(self, source: Union[PathLike, BytesIO], rel_path: str = "", zip_file: Optional[zipfile.ZipFile] = None):
        super().__init__(source, rel_path)
        self.zip_file: zipfile.ZipFile = zip_file or zipfile.ZipFile(
            source, "r")

    def read_text(self, subpath: PathLike, encoding: str = "utf-8") -> str:
        return self.zip_file.read(str(self.resolved_path(subpath))).decode(encoding)

    def read_binary(self, subpath: PathLike) -> bytes:
        return self.zip_file.read(str(self.resolved_path(subpath)))

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

    def finalize(self):
        """Close the zip file."""
        if self.zip_file:
            self.zip_file.close()


class FileWriteHandler(WriteHandler):
    """Handler for writing files to the regular file system."""

    destination: Path

    def __init__(self, destination: PathLike, rel_path: str = ""):
        super().__init__(Path(destination), rel_path)

    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        full_path = self.destination / self.resolved_path(subpath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        if executable:
            # Make file executable by owner, group, and others
            current_permissions = full_path.stat().st_mode
            full_path.chmod(current_permissions | stat.S_IXUSR |
                            stat.S_IXGRP | stat.S_IXOTH)

    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        if isinstance(content, BytesIO):
            content.seek(0)
            content = content.read()

        full_path = self.destination / self.resolved_path(subpath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

        if executable:
            # Make file executable by owner, group, and others
            current_permissions = full_path.stat().st_mode
            full_path.chmod(current_permissions | stat.S_IXUSR |
                            stat.S_IXGRP | stat.S_IXOTH)


class ZipWriteHandler(WriteHandler):
    """Handler for writing files to zip archives."""

    # Fixed date_time for deterministic zip output (2020-01-01 00:00:00)
    DETERMINISTIC_DATE_TIME = (2020, 1, 1, 0, 0, 0)

    def __init__(self, destination: Union[PathLike, BytesIO], rel_path: str = "", zip_file: Optional[zipfile.ZipFile] = None):
        super().__init__(destination, rel_path)
        self.zip_file = zip_file or zipfile.ZipFile(destination, "w")

    def write_text(self, subpath: PathLike, content: str, executable: bool = False) -> None:
        zip_info = zipfile.ZipInfo(str(self.resolved_path(
            subpath)), date_time=self.DETERMINISTIC_DATE_TIME)
        if executable:
            # Set Unix file permissions for executable files (0o755)
            zip_info.external_attr = 0o755 << 16
        self.zip_file.writestr(zip_info, content)

    def write_binary(self, subpath: PathLike, content: Union[bytes, BytesIO], executable: bool = False) -> None:
        if isinstance(content, BytesIO):
            content.seek(0)
            content = content.read()

        zip_info = zipfile.ZipInfo(str(self.resolved_path(
            subpath)), date_time=self.DETERMINISTIC_DATE_TIME)
        if executable:
            # Set Unix file permissions for executable files (0o755)
            zip_info.external_attr = 0o755 << 16
        self.zip_file.writestr(zip_info, content)

    def finalize(self):
        """Close the zip file."""
        if hasattr(self, 'zip_file') and self.zip_file:
            self.zip_file.close()
