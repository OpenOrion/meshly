"""Resource references for file handling in Packable serialization.

ResourceRef allows file paths to be used as Pydantic fields that automatically
get serialized by checksum when extracted/reconstructed.
"""

from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, PrivateAttr, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from meshly.utils.checksum_utils import ChecksumUtils


class ResourceRefMetadata(BaseModel):
    """Metadata for resource $ref objects in data.json.
    
    Resources are gzip-compressed file data.
    
    Example $ref in data.json:
        {"$ref": "abc123", "name": "material.mtl"}
    """
    
    ref: str = Field(..., alias="$ref", description="Asset checksum reference")
    name: Optional[str] = Field(None, description="Original filename (e.g., 'material.mtl')")
    
    model_config = {"populate_by_name": True}


class _ResourceAnnotation:
    """Pydantic annotation for resource references with custom JSON schema."""
    
    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Return JSON schema with type as 'resource'."""
        return {"type": "resource"}


class ResourceRef(BaseModel):
    """Reference to a file resource that can be serialized by checksum.

    When used in a Pydantic model that gets extracted via Packable.extract():
    - On host: path is the local file path
    - On extract: file is read, checksum computed, stored as {"$ref": checksum, "ext": extension}
    - On reconstruct: loaded from assets by checksum with extension

    Example:
        from meshly import Packable, Resource

        class SimulationCase(Packable):
            geometry: Resource  # File path that gets uploaded/serialized by checksum

        # Usage
        case = SimulationCase(geometry="model.stl")

        # Serialize for transmission
        extracted = Packable.extract(case)
        # extracted.data = {"geometry": {"$ref": "a1b2c3d4", "ext": ".stl"}}
        # extracted.assets = {"a1b2c3d4": <stl file bytes>}

        # Reconstruct from serialized data
        case2 = Packable.reconstruct(SimulationCase, extracted.data, extracted.assets)
    """

    model_config = ConfigDict(frozen=True)

    path: Optional[str] = None
    ext: Optional[str] = None
    _explicit_checksum: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _validate_input(cls, data: Any, handler):
        """Handle various input formats."""
        checksum_to_set = None

        if isinstance(data, ResourceRef):
            checksum_to_set = data._explicit_checksum
            data = {"path": data.path, "ext": data.ext}

        elif isinstance(data, (str, Path)):
            p = Path(data)
            data = {"path": str(p), "ext": p.suffix}

        elif isinstance(data, dict):
            if "$ref" in data:
                # Reconstructing from serialized data
                checksum_to_set = data["$ref"]
                data = {"path": None, "ext": data.get("ext", "")}
            elif "path" in data:
                path = data.get("path")
                checksum_to_set = data.get("_explicit_checksum")
                data = {
                    "path": path,
                    "ext": data.get("ext") or (Path(path).suffix if path else None),
                }

        # Call the default handler to create the model
        instance = handler(data)

        # Set the private attribute after creation
        if checksum_to_set is not None:
            object.__setattr__(instance, "_explicit_checksum", checksum_to_set)

        return instance

    @cached_property
    def checksum(self) -> Optional[str]:
        """Get checksum - computed lazily from file if not set explicitly."""
        if self._explicit_checksum is not None:
            return self._explicit_checksum

        if self.path and Path(self.path).exists():

            data = Path(self.path).read_bytes()
            return ChecksumUtils.compute_bytes_checksum(data)

        return None

    def __repr__(self) -> str:
        if self._explicit_checksum:
            return f"ResourceRef(checksum={self._explicit_checksum!r}, ext={self.ext!r})"
        return f"ResourceRef(path={self.path!r})"

    def __str__(self) -> str:
        return self.path or self._explicit_checksum or ""

    def read_bytes(self) -> bytes:
        """Read the resource file data.

        Returns:
            File bytes

        Raises:
            FileNotFoundError: If path doesn't exist and no cached data
        """
        if self.path:
            p = Path(self.path)
            if p.exists():
                return p.read_bytes()

        raise FileNotFoundError(
            f"ResourceRef has no readable path. Path: {self.path}, Checksum: {self._explicit_checksum}, Ext: {self.ext}"
        )

    def resolve_path(self, resource_path: Optional[Path] = None) -> Path:
        """Get the file path for reading.

        Args:
            resource_path: Optional resource directory path to search for cached files

        Returns:
            Path object

        Raises:
            ValueError: If no path available
        """
        if self.path and Path(self.path).exists():
            return Path(self.path)

        cs = self._explicit_checksum or self.checksum
        if cs and resource_path:
            for suffix in [self.ext or "", ""]:
                cache_path = resource_path / (cs + suffix)
                if cache_path.exists():
                    return cache_path

        raise ValueError(f"ResourceRef has no path (checksum={cs})")


# Annotated type alias for use in Pydantic models with proper JSON schema
Resource = Annotated[ResourceRef, _ResourceAnnotation()]
"""Resource reference with custom JSON schema type='resource'."""
