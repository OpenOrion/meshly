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

Checksum Scheme:
    Packable checksums are computed from the SHA256 of the encoded zip bytes.
    This ensures the checksum captures the exact binary representation.
    
    The checksum property is final and cannot be overridden by subclasses.
    Attempting to override will raise a TypeError.
    
    To recreate a checksum externally:
        import hashlib
        checksum = hashlib.sha256(packable.encode()).hexdigest()
"""

import os
import time
import zipfile
from functools import cached_property, lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TypeVar, Union

import orjson
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


def _reconstruct_packable(cls, data: dict):
    """Helper function for pickle reconstruction of Packable objects."""
    return cls.model_construct(**data)



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


class PackableStore(BaseModel):
    """Configuration for file-based Packable asset storage.
    
    Assets (binary blobs) are stored by their SHA256 checksum, enabling deduplication.
    Extracted packable data is stored at user-specified keys as JSON files.
    
    Directory structure:
        root_dir/
            assets/           (ExportConstants.ASSETS_DIR)
                <checksum1>.bin
                <checksum2>.bin
            runs/             (ExportConstants.EXTRACTED_DIR)
                <key>.json
    
    Example:
        store = PackableStore(root_dir=Path("/data/my_package"))
        my_mesh.save(store, "experiment/result")
        loaded = Mesh.load(store, "experiment/result")
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    root_dir: PathLike = Field(..., description="Root directory for all storage")
    extracted_dir: str = Field(default="runs", description="Subdirectory for extracted JSON files")

    @property
    def assets_path(self) -> Path:
        """Directory for binary assets."""
        return Path(self.root_dir) / ExportConstants.ASSETS_DIR

    @property
    def extracted_path(self) -> Path:
        """Directory for extracted JSON files."""
        return Path(self.root_dir) / self.extracted_dir

    def asset_file(self, checksum: str) -> Path:
        """Get the filesystem path for a binary asset."""
        return Path(self.root_dir) / ExportConstants.get_rel_asset_path(checksum)
    
    
    def get_extracted_path(self, key: str) -> Path:
        """Get the filesystem path for an extracted packable's JSON file."""
        return self.extracted_path / f"{key}.json"
    
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
        extracted_json_path.write_bytes(orjson.dumps(extracted.model_dump()))
    
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
        extracted_data = orjson.loads(extracted_json_path.read_bytes())
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

    _cached_encode: Optional[bytes] = PrivateAttr(default=None)
    """Cached encoded bytes for reconstructed Packables to avoid re-encoding."""

    _original_json_schema: Optional[dict[str, Any]] = PrivateAttr(default=None)
    """JSON schema carried from reconstruction, used when model_json_schema() is unavailable
    (e.g. dynamic models with numpy fields that Pydantic cannot generate a schema for)."""

    class Config:
        arbitrary_types_allowed = True

    def __init_subclass__(cls, **kwargs):
        """Prevent subclasses from overriding the checksum property."""
        super().__init_subclass__(**kwargs)
        # Check if this class defines its own 'checksum' (not inherited)
        if 'checksum' in cls.__dict__:
            raise TypeError(
                f"Cannot override 'checksum' property in {cls.__name__}. "
                f"The checksum is computed from encoded bytes and is final."
            )

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

    @classmethod
    @lru_cache(maxsize=None)
    def cached_json_schema(cls) -> dict[str, Any]:
        """Get cached JSON schema for this class.
        
        This caches model_json_schema() at the class level to avoid
        regenerating it for every instance during extraction.
        """
        return cls.model_json_schema()

    def _get_json_schema(self) -> dict[str, Any]:
        """Return the JSON schema for this instance.

        Prefers the class-level cached schema from ``cached_json_schema()``.
        Falls back to ``_original_json_schema`` carried from reconstruction
        when the class-level call fails (e.g. dynamic models whose numpy
        fields prevent Pydantic from generating a schema).
        """
        try:
            return type(self).cached_json_schema()
        except Exception:
            if self._original_json_schema is not None:
                return self._original_json_schema
            raise

    def extract(self) -> "ExtractedPackable":
        """Extract arrays and Packables from this model into serializable data and assets.
        
        Results are cached for efficiency. Subsequent calls return the cached result.
        
        Returns:
            ExtractedPackable with metadata (data + schema + checksum) and binary assets.
        """
        if self._cached_extract is not None:
            return self._cached_extract
        
        extracted_result = SerializationUtils.extract_basemodel(
            self, 
            include_computed=True,
        )

        assert isinstance(extracted_result.value, dict), "Extracted value must be a dict for Packable models"

        try:
            schema = type(self).cached_json_schema()
        except Exception:
            if self._original_json_schema is not None:
                schema = self._original_json_schema
            else:
                raise

        extracted = ExtractedPackable(
            data=extracted_result.value,
            json_schema=self._get_json_schema(),
            assets=extracted_result.assets,
        )
        
        self._cached_extract = extracted
        return self._cached_extract

    @cached_property
    def _encoded(self) -> bytes:
        """Cached encoded bytes (zip format)."""
        extracted = self.extract()
        
        destination = BytesIO()
        
        # Use ZIP_STORED for assets (already compressed by meshoptimizer)
        # This avoids double-compression overhead
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as zf:
            # Write extracted data (data + schema) as single JSON
            info = zipfile.ZipInfo(ExportConstants.EXTRACTED_FILE_NAME, date_time=ExportConstants.EXPORT_TIME)
            zf.writestr(info, orjson.dumps(extracted.model_dump()))

            # Write assets by checksum (already compressed, skip compression)
            for checksum in sorted(extracted.assets.keys()):
                info = zipfile.ZipInfo(ExportConstants.get_rel_asset_path(checksum), date_time=ExportConstants.EXPORT_TIME)
                zf.writestr(info, extracted.assets[checksum])

        return destination.getvalue()

    def encode(self) -> bytes:
        """Encode this Packable to bytes (zip format).
        
        Calls extract() internally to serialize all fields and assets.
        Results are cached for efficiency.
            
        Returns:
            Encoded bytes in zip format.
        """
        return self._encoded

    @cached_property
    def checksum(self) -> str:
        """SHA256 checksum of this Packable's encoded zip bytes (cached, final).
        
        This property cannot be overridden by subclasses. Attempting to do so
        will raise a TypeError at class definition time.
        
        To recreate this checksum outside meshly:
            import hashlib
            checksum = hashlib.sha256(packable.encode()).hexdigest()
        """
        return ChecksumUtils.compute_bytes_checksum(self._encoded)

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
            Reconstructed Packable instance with cached encode/checksum.
        """
        with zipfile.ZipFile(BytesIO(buf), "r") as zf:
            # Read extracted data (data + schema)
            extracted_json = orjson.loads(zf.read(ExportConstants.EXTRACTED_FILE_NAME))

            # Build assets dict from files
            assets: dict[str, bytes] = {}
            for file_path in zf.namelist():
                if file_path.startswith(ExportConstants.ASSETS_DIR) and file_path.endswith(ExportConstants.ASSET_EXT):
                    checksum = ExportConstants.get_relative_asset_checksum(file_path)
                    assets[checksum] = zf.read(file_path)

        # Create ExtractedPackable from zip contents
        extracted = ExtractedPackable(
            data=extracted_json["data"],
            json_schema=extracted_json.get("json_schema"),
            assets=assets,
        )
        result = cls.reconstruct(extracted, array_type=array_type)
        
        # Cache for efficiency
        result._cached_encode = buf
        result._cached_extract = extracted
        return result

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
            result = DynamicModelBuilder.instantiate(json_schema, extracted.data, asset_provider, array_type, is_lazy)
        elif is_lazy:
            schema = JsonSchema.model_validate(cls.model_json_schema())
            result = DynamicModelBuilder.instantiate(schema, extracted.data, asset_provider, array_type, is_lazy=True)
        else:
            resolved_data = SchemaUtils.resolve_from_class(cls, extracted.data, asset_provider, array_type)
            result = cls(**resolved_data)
        
        # Preserve the original schema on Packable instances so that
        # _get_json_schema() can fall back to it when the class-level
        # cached_json_schema() is unavailable (dynamic models).
        if isinstance(result, Packable) and not is_lazy:
            result._original_json_schema = extracted.json_schema
        
        return result

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
            
            store = PackableStore(root_dir=Path("/path/to/data"))
            
            # Save with auto-generated checksum key
            key = my_mesh.save(store)
            
            # Save with explicit key
            settings.save(store, f"{checksum}/settings")
            result.save(store, f"{checksum}/result")
        """
        start_time = time.time()

        extracted = self.extract()   
        elapsed_ms = (time.time() - start_time) * 1000
        # print(f"Extracted packable in {elapsed_ms:.1f} ms with {len(extracted.assets)} assets")
        result_key = key or self.checksum
        
        # Save new binary assets (skip existing)
        assets_dir = store.assets_path
        assets_dir_exists = assets_dir.exists()
        new_assets = {
            cs: data for cs, data in extracted.assets.items()
            if not assets_dir_exists or not store.asset_exists(cs)
        }
        if new_assets:
            assets_dir.mkdir(parents=True, exist_ok=True)
            for cs, data in new_assets.items():
                fd = os.open(str(store.asset_file(cs)), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
                os.write(fd, data)
                os.close(fd)
        
        # Save extracted data (data + schema + checksum) as JSON
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
            
            store = PackableStore(root_dir=Path("/path/to/data"))
            mesh = Mesh.load(store, "abc123")
            
            # Load from specific key
            result = Result.load(store, f"{checksum}/result")
        """
        if not store.extracted_exists(key):
            return None
        extracted = store.load_extracted(key)
        return cls.reconstruct(extracted, assets=store.load_asset, array_type=array_type, is_lazy=is_lazy)

    def __reduce__(self):
        """Support for pickle serialization using standard dict approach."""
        return (
            _reconstruct_packable,
            (self.__class__, dict(self)),
        )

    def convert_to(self, array_type: ArrayType):
        """Create a new Packable with all arrays converted to the specified type."""
        data_copy = self.model_copy(deep=True)

        for field_name in data_copy.model_fields_set:
            value = getattr(data_copy, field_name)            
            converted = ArrayUtils.convert_recursive(value, array_type)
            setattr(data_copy, field_name, converted)

        return data_copy
