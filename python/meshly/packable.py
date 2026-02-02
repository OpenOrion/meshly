"""Base packable class for encoded array storage.

This module provides a base class that handles automatic numpy/JAX array
detection and serialization to zip files. Classes like Mesh inherit from
this base to get automatic array encoding/decoding support.

Custom data classes can inherit from Packable to store simulation
results, time-series data, or any structured data with numpy arrays.

Packables can be nested. By default, nested Packables are "expanded" during
extraction (their fields become part of the parent's data dict with $refs for
arrays). Set `is_contained = True` on a Packable class to make it extract
as a single zip blob reference instead.
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, TypeVar, Union

from pydantic import BaseModel, Field

from meshly.array import ArrayType, ArrayUtils
from meshly.common import AssetProvider, PathLike, RefInfo
from meshly.constants import ExportConstants
from meshly.utils.schema_utils import SchemaUtils
from meshly.utils.serialization_utils import SerializationUtils
from meshly.utils.json_schema import JsonSchema
from meshly.utils.dynamic_model import DynamicModelBuilder, LazyModel

TModel = TypeVar("TModel", bound=BaseModel)


class PackableRefInfo(RefInfo):
    """Ref model for self-contained packable $ref (encoded as zip)."""
    ref: str = Field(..., alias="$ref")


class PackableMetadata(BaseModel):
    """Metadata containing the serializable data and JSON schema.
    
    This is the JSON-serializable portion of extracted Packable data,
    containing both the data dict with $ref checksums and the JSON schema.
    """
    
    data: dict[str, Any] = Field(..., description="Serializable dict with primitive fields and checksum refs for arrays")
    schema: dict[str, Any] = Field(..., description="JSON Schema with encoding info")

    @staticmethod
    def extract_checksums(data: dict[str, Any]) -> list[str]:
        """Extract all $ref checksums from a serialized data dict.

        Recursively walks the data structure to find all {"$ref": checksum} entries.

        Args:
            data: Serialized data dict with $ref references

        Returns:
            List of all unique checksums found
        """
        checksums: set[str] = set()

        def _extract(obj: Any) -> None:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    checksums.add(obj["$ref"])
                else:
                    for v in obj.values():
                        _extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        _extract(data)
        return list(checksums)


class ExtractedPackable(BaseModel):
    """Result of extracting a Packable for serialization.

    Contains the metadata (data dict + schema) and the binary assets.
    The data dict may contain a reserved key '$module' with the fully qualified
    class name for automatic class resolution during reconstruction.
    """
    
    model_config = {"arbitrary_types_allowed": True}

    metadata: PackableMetadata = Field(..., description="JSON-serializable metadata (data + schema)")
    assets: dict[str, bytes] = Field(default_factory=dict, description="Map of checksum -> encoded bytes for all arrays")


class Packable(BaseModel):
    """Base class for data containers with automatic array serialization.

    Subclasses can define numpy array attributes which will be automatically
    detected, encoded, and saved to zip files. Non-array fields are preserved
    in metadata.

    Nested Packables are supported. By default, nested Packables are "expanded"
    during extraction (fields inlined with $refs for arrays). Set
    `is_contained = True` to make the class extract as a single zip blob.

    Example:
        class SimulationResult(Packable):
            time: float
            temperature: np.ndarray
            velocity: np.ndarray

        result = SimulationResult(
            time=0.1,
            temperature=np.array([300.0, 301.0, 302.0]),
            velocity=np.zeros((3, 3))
        )
        result.save_to_zip("result.zip")
        loaded = SimulationResult.load_from_zip("result.zip")

        # Self-contained Packable (extracts as single $ref)
        class Mesh(Packable):
            is_contained: ClassVar[bool] = True
            vertices: np.ndarray
            faces: np.ndarray
    """

    is_contained: ClassVar[bool] = False
    """If True, this Packable extracts as a single zip blob reference.
    If False (default), fields are expanded with $refs for individual arrays."""

    class Config:
        arbitrary_types_allowed = True

    def extract(self, include_computed: bool = True) -> "ExtractedPackable":
        """Extract arrays and Packables from this model into serializable data and assets.
        
        Args:
            include_computed: If True, includes @computed_field properties in output.
        
        Returns:
            ExtractedPackable with metadata (data + schema) and binary assets.
        """
        extracted_result = SerializationUtils.extract_basemodel(
            self, 
            include_computed=include_computed
        )

        return ExtractedPackable(
            metadata=PackableMetadata(
                data=extracted_result.value,
                schema=type(self).model_json_schema()
            ),
            assets=extracted_result.assets,
        )

    def encode(self) -> bytes:
        """Encode this Packable to bytes (zip format).
        
        Calls extract() internally to serialize all fields and assets.
            
        Returns:
            Encoded bytes in zip format.
        """
        extracted = self.extract()
        
        destination = BytesIO()
        
        with zipfile.ZipFile(destination, "w") as zf:
            # Write JSON Schema with encoding info (if available)
            if extracted.metadata.schema is not None:
                info = zipfile.ZipInfo(ExportConstants.SCHEMA_FILE, date_time=ExportConstants.EXPORT_TIME)
                zf.writestr(info, json.dumps(extracted.metadata.schema, indent=2, sort_keys=True))

            # Write data as JSON
            info = zipfile.ZipInfo(ExportConstants.DATA_FILE, date_time=ExportConstants.EXPORT_TIME)
            zf.writestr(info, json.dumps(extracted.metadata.data, indent=2, sort_keys=True))

            # Write assets by checksum
            for checksum in sorted(extracted.assets.keys()):
                info = zipfile.ZipInfo(ExportConstants.asset_path(checksum), date_time=ExportConstants.EXPORT_TIME)
                zf.writestr(info, extracted.assets[checksum])

        return destination.getvalue()

    @classmethod
    def decode(
        cls,
        buf: bytes,
        array_type: ArrayType = "numpy",
    ):
        """Decode bytes to a reconstructed Packable.
        
        Args:
            buf: Encoded bytes (zip format)
            array_type: Target array backend type ("numpy" or "jax")
        
        Returns:
            Reconstructed Packable instance.
        """
        with zipfile.ZipFile(BytesIO(buf), "r") as zf:
            # Read data
            data = json.loads(zf.read(ExportConstants.DATA_FILE).decode("utf-8"))

            # Read schema if present
            try:
                schema = json.loads(zf.read(ExportConstants.SCHEMA_FILE).decode("utf-8"))
            except KeyError:
                schema = {}

            # Build assets dict from files
            assets: dict[str, bytes] = {}
            for file_path in zf.namelist():
                if file_path.startswith(ExportConstants.ASSETS_DIR) and file_path.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.checksum_from_path(file_path)
                    assets[checksum] = zf.read(file_path)

        # If called on base Packable, use schema-based decoding
        if cls is Packable:
            if not schema:
                raise ValueError(
                    "schema.json not found. Use a specific subclass for decode() "
                    "or ensure the packable was created with Array/VertexBuffer/IndexSequence types."
                )
            json_schema = JsonSchema.model_validate(schema)
            return Packable.reconstruct(json_schema, data, assets, array_type)
        
        return Packable.reconstruct(cls, data, assets, array_type)

    @staticmethod
    def reconstruct(
        model_class: Union[type[TModel], JsonSchema, None],
        data: Union[dict[str, Any]],
        assets: AssetProvider = {},
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> Union[TModel, LazyModel, BaseModel]:
        """Reconstruct a Pydantic BaseModel from extracted data and assets.

        Args:
            model_class: The Pydantic model class to reconstruct, or a JsonSchema
                        for schema-based decoding (builds a dynamic model).
            data: Either a dict containing the extracted data.
            assets: Asset provider (dict or callable).
            array_type: Optional array type for conversion.
            is_lazy: If True, returns a lazy proxy that defers asset loading until field access.

        Returns:
            Reconstructed model instance, lazy proxy, or dynamically built model (if JsonSchema passed).
        
        Example:
            extracted = Packable.extract(my_model)
            restored = Packable.reconstruct(MyModel, extracted.data, extracted.assets)
            
            # Lazy loading with known model class
            lazy_model = Packable.reconstruct(MyModel, data, fetch_asset_fn, is_lazy=True)
            
            # Lazy loading with schema (dynamic model)
            lazy_dynamic = Packable.reconstruct(schema, data, fetch_asset_fn, is_lazy=True)
        """

        # Schema-based decoding (builds dynamic model)
        if isinstance(model_class, JsonSchema):
            return DynamicModelBuilder.instantiate(model_class, data, assets, array_type, is_lazy)

        assert issubclass(model_class, BaseModel), "model_class must be a Pydantic BaseModel subclass"
        
        # For typed model classes, generate schema and use DynamicModelBuilder for lazy loading
        if is_lazy:
            schema = JsonSchema.model_validate(model_class.model_json_schema())
            return DynamicModelBuilder.instantiate(schema, data, assets, array_type, is_lazy=True)
    
        resolved_data = SchemaUtils.resolve_from_class(model_class, data, assets, array_type)
        return model_class(**resolved_data)

    @classmethod
    def load_from_zip(
        cls,
        source: Union[PathLike, BytesIO],
        array_type: ArrayType = "numpy",
    ):
        """Load a Packable from a zip file."""
        if isinstance(source, BytesIO):
            buf = source.read()
        else:
            buf = Path(source).read_bytes()
        
        return cls.decode(buf, array_type)

    def save_to_zip(self, destination: Union[PathLike, BytesIO]) -> None:
        """Save this container to a zip file."""
        encoded = self.encode()
        if isinstance(destination, BytesIO):
            destination.write(encoded)
        else:
            Path(destination).write_bytes(encoded)

    def __reduce__(self):
        """Support for pickle serialization."""
        encoded = self.encode()
        return (self.__class__.load_from_zip, (BytesIO(encoded),))

    def convert_to(self, array_type: ArrayType):
        """Create a new Packable with all arrays converted to the specified type."""
        data_copy = self.model_copy(deep=True)

        for field_name in data_copy.model_fields_set:
            value = getattr(data_copy, field_name)            
            converted = ArrayUtils.convert_recursive(value, array_type)
            setattr(data_copy, field_name, converted)

        return data_copy
