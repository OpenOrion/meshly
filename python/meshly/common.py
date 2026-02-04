# Common type definitions
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


PathLike = Union[str, Path]

# Type for asset provider: either a dict or a callable that fetches by checksum
# Supports both sync and async fetch functions
# The callable can return None to indicate the asset should be read from cache
AssetFetcher = Callable[[str], Union[bytes, None, Awaitable[Optional[bytes]]]]
AssetProvider = Union[Dict[str, bytes], AssetFetcher]


class RefInfo(BaseModel):
    """Base class for models that serialize with $ref checksum pattern.
    
    Subclasses should define a `ref` field with `alias="$ref"` for the checksum.
    Use `model_dump(by_alias=True, exclude_defaults=True)` for serialization.
    
    Example:
        class ArrayRefModel(RefModel):
            ref: str | None = Field(None, alias="$ref")
            dtype: str
            shape: list[int]
            
        class ResourceRef(RefModel):
            data: bytes = Field(exclude=True)
            
            @computed_field(alias="$ref")
            @property
            def checksum(self) -> str:
                return compute_checksum(self.data)
    """
    model_config = ConfigDict(populate_by_name=True)
