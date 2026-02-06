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

Serialization options:
- save_to_zip() / load_from_zip(): Single self-contained zip file
- save() / load(): File-based asset store with deduplication
"""

import json
import zipfile
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic.json_schema import JsonSchemaValue, GetJsonSchemaHandler
from pydantic_core import core_schema as pydantic_core_schema

from meshly.array import ArrayType, ArrayUtils
from meshly.common import AssetProvider, PathLike, RefInfo
from meshly.constants import ExportConstants
from meshly.utils.checksum_utils import ChecksumUtils
from meshly.utils.schema_utils import SchemaUtils
from meshly.utils.serialization_utils import SerializationUtils
from meshly.utils.json_schema import JsonSchema
from meshly.utils.dynamic_model import DynamicModelBuilder, LazyModel

if TYPE_CHECKING:
    pass

TModel = TypeVar("TModel", bound=BaseModel)


class PackableRefInfo(RefInfo):
    """Ref model for self-contained packable $ref (encoded as zip)."""
    ref: str = Field(..., alias="$ref")


class ExtractedPackable(BaseModel):
    """Result of extracting a Packable for serialization.

    Contains the data dict, JSON schema, and binary assets.
    The schema contains 'x-module' with the fully qualified class path.
    The schema's 'x-base' indicates base class type ('packable', 'mesh', etc).
    
    Use model_dump() to get a JSON-serializable dict (assets are excluded).
    """
    
    model_config = {"arbitrary_types_allowed": True}

    data: dict[str, Any] = Field(..., description="Serializable dict with primitive fields and checksum refs for arrays")
    json_schema: Optional[dict[str, Any]] = Field(default=None, description="JSON Schema with encoding info")
    assets: dict[str, bytes] = Field(default_factory=dict, exclude=True, description="Map of checksum -> encoded bytes for all arrays")

    def extract_checksums(self) -> list[str]:
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

        _extract(self.data)
        return list(checksums)


class PackableStore(BaseModel):
    """Configuration for file-based Packable asset storage.
    
    Assets (binary blobs) are stored by their SHA256 checksum, enabling deduplication.
    Extracted packable data is stored at user-specified keys as JSON files.
    
    Directory structure:
        assets_path/
            <checksum1>.bin
            <checksum2>.bin
        extracted_path/
            <key>.json  (contains both data and json_schema)
    
    Example:
        store = PackableStore(assets_path=Path("/data/assets"), extracted_path=Path("/data/runs"))
        my_mesh.save(store, "experiment/result")
        loaded = Mesh.load(store, "experiment/result")
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    assets_path: Path = Field(..., description="Directory for binary assets")
    extracted_path: Optional[Path] = Field(default=None, description="Directory for extracted JSON files. If None, uses assets_path")
    
    def asset_file(self, checksum: str) -> Path:
        """Get the filesystem path for a binary asset."""
        return self.assets_path / f"{checksum}.bin"
    
    def get_extracted_path(self, key: str) -> Path:
        """Get the filesystem path for an extracted packable's JSON file."""
        extracted_dir = self.extracted_path if self.extracted_path is not None else self.assets_path
        return extracted_dir / f"{key}.json"
    
    def save_asset(self, data: bytes, checksum: str) -> None:
        """Save binary asset data to storage.
        
        Args:
            data: Binary data to save
            checksum: SHA256 checksum identifier for the asset
        """
        asset_path = self.asset_file(checksum)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(data)
    
    def load_asset(self, checksum: str) -> bytes:
        """Load binary asset data from storage.
        
        Args:
            checksum: SHA256 checksum identifier for the asset
            
        Returns:
            Binary data of the asset
            
        Raises:
            FileNotFoundError: If asset doesn't exist
        """
        return self.asset_file(checksum).read_bytes()
    
    def asset_exists(self, checksum: str) -> bool:
        """Check if an asset exists in storage.
        
        Args:
            checksum: SHA256 checksum identifier for the asset
            
        Returns:
            True if asset exists, False otherwise
        """
        return self.asset_file(checksum).exists()

    def save_extracted(self, key: str, extracted: "ExtractedPackable") -> None:
        """Save extracted packable JSON to storage.
        
        Args:
            key: Identifier for the packable
            extracted: ExtractedPackable to save (assets are not saved here)
        """
        extracted_json_path = self.get_extracted_path(key)
        extracted_json_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_json_path.write_text(json.dumps(extracted.model_dump(), indent=2, sort_keys=True))
    
    def load_extracted(self, key: str) -> "ExtractedPackable":
        """Load extracted packable from storage.
        
        Args:
            key: Identifier for the packable
            
        Returns:
            ExtractedPackable with data and json_schema (assets loaded via load_asset)
            
        Raises:
            FileNotFoundError: If extracted file doesn't exist
        """
        extracted_json_path = self.get_extracted_path(key)
        if not extracted_json_path.exists():
            raise FileNotFoundError(f"Extracted packable not found: {key}")
        extracted_data = json.loads(extracted_json_path.read_text())
        return ExtractedPackable(
            data=extracted_data["data"],
            json_schema=extracted_data.get("json_schema"),
        )
    
    def extracted_exists(self, key: str) -> bool:
        """Check if an extracted packable exists in storage.
        
        Args:
            key: Identifier for the packable
            
        Returns:
            True if extracted file exists, False otherwise
        """
        return self.get_extracted_path(key).exists()


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

    _cached_extract: Optional["ExtractedPackable"] = PrivateAttr(default=None)
    """Cached result of extract() to avoid recomputation."""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema_obj: pydantic_core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Inject x-base and x-module hints into JSON schema."""
        json_schema = handler(core_schema_obj)
        json_schema = handler.resolve_ref_schema(json_schema)
        # Add base class hint: "Packable", "Mesh", or "BaseModel"
        json_schema['x-base'] = 'Packable'
        # Add module path for reconstruction
        json_schema['x-module'] = f"{cls.__module__}.{cls.__qualname__}"
        return json_schema

    def extract(self) -> "ExtractedPackable":
        """Extract arrays and Packables from this model into serializable data and assets.
        
        Results are cached for efficiency. Subsequent calls return the cached result.
        
        Returns:
            ExtractedPackable with metadata (data + schema) and binary assets.
        """
        if self._cached_extract is not None:
            return self._cached_extract
        
        extracted_result = SerializationUtils.extract_basemodel(
            self, 
            include_computed=True,
        )

        assert isinstance(extracted_result.value, dict), "Extracted value must be a dict for Packable models"

        self._cached_extract = ExtractedPackable(
            data=extracted_result.value,
            json_schema=type(self).model_json_schema(),
            assets=extracted_result.assets,
        )
        return self._cached_extract

    def encode(self) -> bytes:
        """Encode this Packable to bytes (zip format).
        
        Calls extract() internally to serialize all fields and assets.
            
        Returns:
            Encoded bytes in zip format.
        """
        extracted = self.extract()
        
        destination = BytesIO()
        
        with zipfile.ZipFile(destination, "w") as zf:
            # Write extracted data (data + schema) as single JSON
            info = zipfile.ZipInfo(ExportConstants.EXTRACTED_FILE, date_time=ExportConstants.EXPORT_TIME)
            zf.writestr(info, json.dumps(extracted.model_dump(), indent=2, sort_keys=True))

            # Write assets by checksum
            for checksum in sorted(extracted.assets.keys()):
                info = zipfile.ZipInfo(ExportConstants.asset_path(checksum), date_time=ExportConstants.EXPORT_TIME)
                zf.writestr(info, extracted.assets[checksum])

        return destination.getvalue()

    @cached_property
    def checksum(self) -> str:
        """SHA256 checksum of this Packable's encoded bytes (cached)."""
        return ChecksumUtils.compute_bytes_checksum(self.encode())

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
            # Read extracted data (data + schema)
            extracted_json = json.loads(zf.read(ExportConstants.EXTRACTED_FILE).decode("utf-8"))

            # Build assets dict from files
            assets: dict[str, bytes] = {}
            for file_path in zf.namelist():
                if file_path.startswith(ExportConstants.ASSETS_DIR) and file_path.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.checksum_from_path(file_path)
                    assets[checksum] = zf.read(file_path)

        # Create ExtractedPackable from zip contents
        extracted = ExtractedPackable(
            data=extracted_json["data"],
            json_schema=extracted_json.get("json_schema"),
            assets=assets,
        )
        return cls.reconstruct(extracted, array_type=array_type)

    @classmethod
    def reconstruct(
        cls,
        extracted: "ExtractedPackable",
        assets: Optional[AssetProvider] = None,
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> Union[TModel, LazyModel, BaseModel]:
        """Reconstruct a Packable from extracted data and assets.

        Args:
            extracted: ExtractedPackable containing data, json_schema, and assets.
            assets: Optional asset provider override (dict or callable). If None, uses extracted.assets.
            array_type: Target array backend type ("numpy" or "jax").
            is_lazy: If True, returns a lazy proxy that defers asset loading until field access.

        Returns:
            Reconstructed Packable instance, lazy proxy, or dynamically built model.
        
        Example:
            extracted = my_model.extract()
            restored = MyModel.reconstruct(extracted)
            
            # Lazy loading
            lazy_model = MyModel.reconstruct(extracted, is_lazy=True)
            
            # With external asset provider
            restored = MyModel.reconstruct(extracted, assets=store.load_asset)
            
            # Schema-based decoding (when called on base Packable)
            dynamic = Packable.reconstruct(extracted)
        """
        asset_provider: AssetProvider = assets if assets is not None else extracted.assets
        
        # If called on base Packable, use schema-based decoding
        if cls is Packable:
            if extracted.json_schema is None:
                raise ValueError("Cannot reconstruct on base Packable without json_schema")
            json_schema = JsonSchema.model_validate(extracted.json_schema)
            return DynamicModelBuilder.instantiate(json_schema, extracted.data, asset_provider, array_type, is_lazy)
        
        # For typed model classes, generate schema and use DynamicModelBuilder for lazy loading
        if is_lazy:
            schema = JsonSchema.model_validate(cls.model_json_schema())
            return DynamicModelBuilder.instantiate(schema, extracted.data, asset_provider, array_type, is_lazy=True)
    
        resolved_data = SchemaUtils.resolve_from_class(cls, extracted.data, asset_provider, array_type)
        return cls(**resolved_data)

    @staticmethod
    def reconstruct_polymorphic(
        model_classes: list[type[TModel]],
        extracted: "ExtractedPackable",
        assets: Optional[AssetProvider] = None,
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> Union[TModel, LazyModel]:
        """Reconstruct a Packable by matching x-module in schema against a list of model classes.

        Args:
            model_classes: List of candidate model classes to match against.
            extracted: ExtractedPackable containing data, json_schema, and assets.
            assets: Optional asset provider override. If None, uses extracted.assets.
            array_type: Target array backend type ("numpy" or "jax").
            is_lazy: If True, returns a lazy proxy that defers asset loading.

        Returns:
            Reconstructed instance of the matching model class.
            
        Raises:
            ValueError: If x-module is not in json_schema or no matching class is found.
        
        Example:
            model = Packable.reconstruct_polymorphic([ModelA, ModelB], extracted)
        """
        if not extracted.json_schema:
            raise ValueError(
                "Cannot reconstruct polymorphic: json_schema is required"
            )
        module_name = extracted.json_schema.get("x-module")
        if not module_name:
            raise ValueError(
                "Cannot reconstruct polymorphic: json_schema does not contain x-module field"
            )
        
        for model_cls in model_classes:
            cls_module = f"{model_cls.__module__}.{model_cls.__qualname__}"
            if cls_module == module_name:
                return model_cls.reconstruct(extracted, assets, array_type, is_lazy)
        
        available = [f"{c.__module__}.{c.__qualname__}" for c in model_classes]
        raise ValueError(
            f"No matching model class found for x-module='{module_name}'. "
            f"Available classes: {available}"
        )

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

    def save(
        self,
        store: PackableStore,
        key: Optional[str] = None,
    ) -> str:
        """Save this Packable to an asset store.
        
        Extracts the Packable and saves assets to the store's asset directory,
        with extracted data saved to the specified key.
        
        Args:
            store: PackableStore config
            key: Identifier for the packable. If None, uses the content checksum.
            
        Returns:
            The key where the Packable was saved
            
        Example:
            from meshly import PackableStore
            
            store = PackableStore(assets_path=Path("/path/to/assets"))
            
            # Save with auto-generated checksum key
            key = my_mesh.save(store)
            
            # Save with explicit key
            settings.save(store, f"{checksum}/settings")
            result.save(store, f"{checksum}/result")
        """
        extracted = self.extract()
        result_key = key or self.checksum
        
        # Save all binary assets (deduplicated by checksum)
        for asset_checksum, asset_bytes in extracted.assets.items():
            if not store.asset_exists(asset_checksum):
                store.save_asset(asset_bytes, asset_checksum)
        
        # Save extracted data (data + schema) as JSON
        store.save_extracted(result_key, extracted)
        
        return result_key

    @classmethod
    def load(
        cls,
        store: PackableStore,
        key: str,
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ):
        """Load a Packable from an asset store by key.
        
        Args:
            store: PackableStore config
            key: Identifier for the packable
            array_type: Target array backend type ("numpy" or "jax")
            is_lazy: If True, returns a lazy proxy that defers asset loading
            
        Returns:
            Reconstructed Packable instance, or None if not found
            
        Example:
            from meshly import PackableStore, Mesh
            
            store = PackableStore(assets_path=Path("/path/to/assets"))
            mesh = Mesh.load(store, "abc123")
            
            # Load from specific key
            result = Result.load(store, f"{checksum}/result")
        """
        if not store.extracted_exists(key):
            return None
        extracted = store.load_extracted(key)
        return cls.reconstruct(extracted, assets=store.load_asset, array_type=array_type, is_lazy=is_lazy)

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
