"""Resource references for file handling in Packable serialization.

ResourceRef allows bytes data to be serialized by checksum when extracted/reconstructed.
"""

import gzip
from pathlib import Path
from typing import Optional, Union

from pydantic import ConfigDict, Field, PrivateAttr, computed_field

from meshly.common import RefInfo
from meshly.utils.checksum_utils import ChecksumUtils


class Resource(RefInfo):
    """Reference to binary resource data that can be serialized by checksum.

    When used in a Pydantic model that gets extracted via Packable.extract():
    - On extract: checksum computed, stored as {"$ref": checksum, "ext": extension}
    - On reconstruct: loaded from assets by checksum

    Example:
        from meshly import Packable, ResourceRef

        class SimulationCase(Packable):
            geometry: ResourceRef  # Binary data serialized by checksum

        # Usage - create from file path
        case = SimulationCase(geometry=ResourceRef.from_path("model.stl"))

        # Or create from bytes directly
        case = SimulationCase(geometry=ResourceRef(data=stl_bytes, ext=".stl"))

        # Serialize for transmission
        extracted = Packable.extract(case)
        # extracted.data = {"geometry": {"$ref": "a1b2c3d4", "ext": ".stl"}}
        # extracted.assets = {"a1b2c3d4": <stl file bytes>}

        # Reconstruct from serialized data
        case2 = Packable.reconstruct(SimulationCase, extracted.data, extracted.assets)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: bytes = Field(exclude=True)
    ext: Optional[str] = None
    name: Optional[str] = None

    # Cached compression results (computed once, reused)
    _compressed: Optional[bytes] = PrivateAttr(default=None)
    _checksum: Optional[str] = PrivateAttr(default=None)

    @staticmethod
    def from_path(path: Union[str, Path]) -> "Resource":
        """Create a ResourceRef from a file path."""
        p = Path(path)
        return Resource(data=p.read_bytes(), ext=p.suffix, name=p.stem)

    @property
    def compressed(self) -> bytes:
        """Get gzip-compressed data (cached).
        
        Uses mtime=0 for deterministic compression (no timestamp in header).
        """
        if self._compressed is None:
            object.__setattr__(self, '_compressed', gzip.compress(self.data, compresslevel=1, mtime=0))
        return self._compressed  # type: ignore

    @computed_field(alias="$ref")
    @property
    def checksum(self) -> str:
        """Get checksum of compressed data (cached).

        Matches the bytes stored in assets, so upload-side hash checks stay consistent.
        """
        if self._checksum is None:
            object.__setattr__(self, '_checksum', ChecksumUtils.compute_bytes_checksum(self.compressed))
        return self._checksum  # type: ignore
