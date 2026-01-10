"""
Snapshot system for storing simulation data at different time points.

This module provides classes for saving and loading simulation snapshots
containing multiple fields of data in a compressed zip format.
"""

import json
import zipfile
from typing import Dict, Optional, List
from io import BytesIO
import numpy as np
from pydantic import BaseModel, Field

from .array import ArrayUtils, ArrayMetadata, EncodedArray
from .common import PathLike


class FieldMetadata(BaseModel):
    """Metadata for field data (without the actual array)"""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (scalar, vector, tensor)")
    units: Optional[str] = Field(None, description="Field units")
    shape: List[int] = Field(..., description="Shape of the field data array")
    dtype: str = Field(..., description="Data type of the field data array")
    itemsize: int = Field(..., description="Size of each item in bytes")


class FieldData(BaseModel):
    """General field data for simulation results"""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (scalar, vector, tensor)")
    data: np.ndarray = Field(..., description="Field data array")
    units: Optional[str] = Field(None, description="Field units")

    class Config:
        arbitrary_types_allowed = True


class SnapshotMetadata(BaseModel):
    """Metadata for a snapshot (excluding field data arrays)"""
    time: float = Field(..., description="Time value")
    fields: Dict[str, FieldMetadata] = Field(
        default_factory=dict, description="Field metadata")


class Snapshot(BaseModel):
    """
    Snapshot of simulation results at a specific time.

    A snapshot contains a time value and a dictionary of field data.
    It can be saved to and loaded from a single compressed zip file.

    Zip structure:
        metadata.json - Contains time and field metadata
        fields/
            <field_name>.bin - Encoded array data for each field
    """
    time: float = Field(..., description="Time value")
    fields: Dict[str, FieldData] = Field(
        default_factory=dict, description="Field data")

    class Config:
        arbitrary_types_allowed = True

    def get_field(self, name: str) -> Optional[FieldData]:
        """
        Get a field by name.

        Args:
            name: Field name

        Returns:
            FieldData if found, None otherwise
        """
        return self.fields.get(name)

    def add_field(
        self,
        name: str,
        data: np.ndarray,
        field_type: str = "scalar",
        units: Optional[str] = None
    ) -> None:
        """
        Add a field to the snapshot.

        Args:
            name: Field name
            data: Field data array
            field_type: Field type (scalar, vector, tensor)
            units: Optional field units
        """
        self.fields[name] = FieldData(
            name=name,
            type=field_type,
            data=data,
            units=units
        )

    @property
    def field_names(self) -> List[str]:
        """Get list of field names in this snapshot."""
        return list(self.fields.keys())


class SnapshotUtils:
    """Utility class for working with snapshots."""

    @staticmethod
    def save_to_zip(
        snapshot: Snapshot,
        destination: PathLike | BytesIO,
        date_time: Optional[tuple] = None
    ) -> None:
        """
        Save a snapshot to a zip file.

        Args:
            snapshot: Snapshot to save
            destination: Path to the output zip file or BytesIO object
            date_time: Optional date_time tuple for deterministic zip files
        """
        # Build field metadata and encode arrays
        field_metadata: Dict[str, FieldMetadata] = {}
        encoded_fields: Dict[str, bytes] = {}

        for field_name, field_data in snapshot.fields.items():
            # Encode the field array
            encoded = ArrayUtils.encode_array(field_data.data)
            encoded_fields[field_name] = encoded.data

            # Create field metadata
            field_metadata[field_name] = FieldMetadata(
                name=field_data.name,
                type=field_data.type,
                units=field_data.units,
                shape=list(encoded.shape),
                dtype=str(encoded.dtype),
                itemsize=encoded.itemsize
            )

        # Create snapshot metadata
        snapshot_metadata = SnapshotMetadata(
            time=snapshot.time,
            fields=field_metadata
        )

        # Write to zip file
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Prepare files to write
            files_to_write: List[tuple] = [
                ("metadata.json", json.dumps(
                    snapshot_metadata.model_dump(), indent=2, sort_keys=True))
            ]

            # Add encoded field arrays
            for field_name, encoded_data in encoded_fields.items():
                files_to_write.append(
                    (f"fields/{field_name}.bin", encoded_data))

            # Write files in sorted order for deterministic output
            for filename, data in sorted(files_to_write):
                if date_time is not None:
                    info = zipfile.ZipInfo(
                        filename=filename, date_time=date_time)
                else:
                    info = zipfile.ZipInfo(filename=filename)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16  # Fixed file permissions
                if isinstance(data, str):
                    data = data.encode('utf-8')
                zipf.writestr(info, data)

    @staticmethod
    def load_from_zip(source: PathLike | BytesIO) -> Snapshot:
        """
        Load a snapshot from a zip file.

        Args:
            source: Path to the input zip file or BytesIO object

        Returns:
            Loaded Snapshot instance
        """
        with zipfile.ZipFile(source, "r") as zipf:
            # Load metadata
            with zipf.open("metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                snapshot_metadata = SnapshotMetadata(**metadata_dict)

            # Load field data
            fields: Dict[str, FieldData] = {}

            for field_name, field_meta in snapshot_metadata.fields.items():
                # Load encoded array data
                field_path = f"fields/{field_name}.bin"
                with zipf.open(field_path) as f:
                    encoded_data = f.read()

                # Create EncodedArray and decode
                encoded_array = EncodedArray(
                    data=encoded_data,
                    shape=tuple(field_meta.shape),
                    dtype=np.dtype(field_meta.dtype),
                    itemsize=field_meta.itemsize
                )
                decoded_array = ArrayUtils.decode_array(encoded_array)

                # Create FieldData
                fields[field_name] = FieldData(
                    name=field_meta.name,
                    type=field_meta.type,
                    data=decoded_array,
                    units=field_meta.units
                )

            return Snapshot(
                time=snapshot_metadata.time,
                fields=fields
            )
