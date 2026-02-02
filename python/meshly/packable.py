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
from dataclasses import dataclass
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
from meshly.utils.dynamic_model import DynamicModelBuilder, LazyDynamicModel

TModel = TypeVar("TModel", bound=BaseModel)


class PackableRefInfo(RefInfo):
    """Ref model for self-contained packable $ref (encoded as zip)."""
    ref: str = Field(..., alias="$ref")


@dataclass
class SerializedPackableData:
    """Result of extracting a Packable for serialization.

    Contains the serializable data dict with checksum references,
    the encoded assets (arrays as bytes), and the JSON schema.
    
    The data dict may contain a reserved key '$module' with the fully qualified
    class name for automatic class resolution during reconstruction.
    """

    data: dict[str, Any]
    """Serializable dict with primitive fields and checksum refs for arrays"""
    assets: dict[str, bytes]
    """Map of checksum -> encoded bytes for all arrays"""
    schema: dict[str, Any]
    """JSON Schema with encoding info"""

    @property
    def checksums(self) -> list[str]:
        """Get list of all asset checksums."""
        return list(self.assets.keys())

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


@dataclass
class ExtractedAssets:
    """Result of extracting assets from values."""

    assets: dict[str, bytes]
    """Map of checksum -> encoded bytes"""


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

    def encode(self) -> bytes:
        """Serialize this Packable to bytes (zip format).

        Uses extract() internally to serialize all fields and assets.
        """
        extracted = Packable.extract(self)

        destination = BytesIO()
        
        with zipfile.ZipFile(destination, "w") as zf:
            # Write JSON Schema with encoding info (if available)
            if extracted.schema is not None:
                info = zipfile.ZipInfo(ExportConstants.SCHEMA_FILE, date_time=ExportConstants.EXPORT_TIME)
                zf.writestr(info, json.dumps(extracted.schema, indent=2, sort_keys=True))

            # Write data as JSON
            info = zipfile.ZipInfo(ExportConstants.DATA_FILE, date_time=ExportConstants.EXPORT_TIME)
            zf.writestr(info, json.dumps(extracted.data, indent=2, sort_keys=True))

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
        """Deserialize a Packable from bytes.

        When called on a specific subclass (e.g., Mesh.decode(...)), uses the class definition.
        When called on base Packable class, uses embedded schema.json and returns a dict.
        
        Args:
            buf: Encoded bytes (zip format)
            array_type: Target array type ("numpy" or "jax")
        
        Returns:
            Reconstructed model instance, or dict if called on Packable base class.
        """
        with zipfile.ZipFile(BytesIO(buf), "r") as zf:
            # Read data
            data = json.loads(zf.read(ExportConstants.DATA_FILE).decode("utf-8"))

            # Build assets dict from files
            assets: dict[str, bytes] = {}
            for file_path in zf.namelist():
                if file_path.startswith(ExportConstants.ASSETS_DIR) and file_path.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.checksum_from_path(file_path)
                    assets[checksum] = zf.read(file_path)

            # If called on base Packable, use schema-based decoding
            if cls is Packable:
                try:
                    schema_dict = json.loads(zf.read(ExportConstants.SCHEMA_FILE).decode("utf-8"))
                    schema = JsonSchema.model_validate(schema_dict)
                except KeyError:
                    raise ValueError(
                        "schema.json not found. Use a specific subclass for decode() "
                        "or ensure the packable was created with Array/VertexBuffer/IndexSequence types."
                    )
                return Packable.reconstruct(schema, data, assets, array_type)

        return Packable.reconstruct(cls, data, assets, array_type)

    @staticmethod
    def extract(obj: BaseModel, include_computed: bool = True) -> SerializedPackableData:
        """Extract arrays and Packables from a BaseModel into serializable data and assets.
        
        Args:
            obj: The Pydantic BaseModel instance to extract.
            include_computed: If True, includes @computed_field properties in output.
        """
        if not isinstance(obj, BaseModel):
            raise TypeError(f"extract() requires a Pydantic BaseModel, got {type(obj).__name__}")
        
        extracted_result = SerializationUtils.extract_basemodel(
            obj, 
            include_computed=include_computed
        )

        return SerializedPackableData(
            data=extracted_result.value,
            assets=extracted_result.assets,
            schema=type(obj).model_json_schema()
        )


    @staticmethod
    def reconstruct(
        model_class: Union[type[TModel], JsonSchema, None],
        data: Union[dict[str, Any]],
        assets: AssetProvider = {},
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> Union[TModel, LazyDynamicModel, BaseModel]:
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
            return cls.decode(source.read(), array_type)
        else:
            return cls.decode(Path(source).read_bytes(), array_type)

    def save_to_zip(self, destination: Union[PathLike, BytesIO]) -> None:
        """Save this container to a zip file."""
        encoded = self.encode()
        if isinstance(destination, BytesIO):
            destination.write(encoded)
        else:
            Path(destination).write_bytes(encoded)

    def __reduce__(self):
        """Support for pickle serialization."""
        return (self.__class__.decode, (self.encode(),))

    def convert_to(self, array_type: ArrayType):
        """Create a new Packable with all arrays converted to the specified type."""
        data_copy = self.model_copy(deep=True)

        for field_name in data_copy.model_fields_set:
            value = getattr(data_copy, field_name)            
            converted = ArrayUtils.convert_recursive(value, array_type)
            setattr(data_copy, field_name, converted)

        return data_copy
