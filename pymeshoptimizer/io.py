"""
Utilities for storing encoded meshes and arrays in zip files.

This module provides functions for storing and loading encoded meshes and arrays in zip files,
where binary data is stored as binary files and metadata is stored as JSON.
"""
import json
import os
import struct
import zipfile
import io
from typing import Optional, Dict, Any, Tuple, Union, List

import numpy as np
from pydantic import BaseModel, Field, RootModel

from .mesh import EncodedMesh

from .arrayutils import EncodedArray, decode_array, encode_array


class ArrayMetadata(BaseModel):
    """Metadata for an encoded array."""
    shape: Tuple[int, ...] = Field(..., description="Shape of the array")
    dtype: str = Field(..., description="Data type of the array")
    itemsize: int = Field(..., description="Size of each item in bytes")
    filename: Optional[str] = Field(None, description="Filename for the array data (used in multi-array zip files)")


class MeshMetadata(BaseModel):
    """Metadata for an encoded mesh."""
    vertex_count: int = Field(..., description="Number of vertices in the mesh")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_size: int = Field(..., description="Size of each index in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices in the mesh")


class ArraysMetadata(RootModel):
    """Container for multiple array metadata entries."""
    root: Dict[str, ArrayMetadata] = Field(..., description="Dictionary mapping array names to their metadata")

    def __getitem__(self, key):
        return self.root[key]
    
    def __setitem__(self, key, value):
        self.root[key] = value
    
    def items(self):
        return self.root.items()
    
    def keys(self):
        return self.root.keys()
    
    def values(self):
        return self.root.values()
    
    @classmethod
    def create_empty(cls):
        return cls(root={})


def save_encoded_array_to_file(encoded_array: EncodedArray, file_path: str) -> None:
    """
    Save an encoded array to a file.
    
    Args:
        encoded_array: EncodedArray object to save
        file_path: Path to the output file
    """
    # Create metadata
    metadata = ArrayMetadata(
        shape=encoded_array.shape,
        dtype=str(encoded_array.dtype),
        itemsize=encoded_array.itemsize
    )
    
    # Convert metadata to JSON
    metadata_json = metadata.model_dump_json()
    
    # Write to file
    with open(file_path, 'wb') as f:
        # Write metadata length as 4-byte integer
        f.write(struct.pack('<I', len(metadata_json)))
        
        # Write metadata as JSON
        f.write(metadata_json.encode('utf-8'))
        
        # Write encoded data
        f.write(encoded_array.data)

def load_encoded_array_from_file(file_path: str) -> EncodedArray:
    """
    Load an encoded array from a file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        EncodedArray object loaded from the file
    """
    with open(file_path, 'rb') as f:
        # Read metadata length
        metadata_length = struct.unpack('<I', f.read(4))[0]
        
        # Read metadata
        metadata_json = f.read(metadata_length).decode('utf-8')
        metadata = ArrayMetadata.model_validate_json(metadata_json)
        
        # Read encoded data
        encoded_data = f.read()
    
    # Create EncodedArray object
    return EncodedArray(
        data=encoded_data,
        shape=metadata.shape,
        dtype=np.dtype(metadata.dtype),
        itemsize=metadata.itemsize
    )

def save_array_to_file(array: np.ndarray, file_path: str) -> None:
    """
    Encode and save a numpy array to a file.
    
    Args:
        array: numpy array to encode and save
        file_path: Path to the output file
    """
    encoded_array = encode_array(array)
    save_encoded_array_to_file(encoded_array, file_path)

def load_array_from_file(file_path: str) -> np.ndarray:
    """
    Load and decode a numpy array from a file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Decoded numpy array
    """
    encoded_array = load_encoded_array_from_file(file_path)
    return decode_array(encoded_array)

def save_encoded_mesh_to_zip(encoded_mesh: EncodedMesh,
                           zip_path: str,
                           vertices_filename: str = "vertices.bin",
                           indices_filename: str = "indices.bin",
                           metadata_filename: str = "metadata.json") -> None:
    """Save an encoded mesh to a zip file."""
    metadata = MeshMetadata(
        vertex_count=encoded_mesh.vertex_count,
        vertex_size=encoded_mesh.vertex_size,
        index_size=encoded_mesh.index_size,
        index_count=encoded_mesh.index_count
    )
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(vertices_filename, encoded_mesh.vertices)
        if encoded_mesh.indices is not None:
            zipf.writestr(indices_filename, encoded_mesh.indices)
        zipf.writestr(metadata_filename, metadata.model_dump_json(indent=2))

def load_encoded_mesh_from_zip(zip_path: str,
                              vertices_filename: str = "vertices.bin",
                              indices_filename: str = "indices.bin",
                              metadata_filename: str = "metadata.json") -> EncodedMesh:
    """Load an encoded mesh from a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(metadata_filename) as f:
            metadata = MeshMetadata.model_validate_json(f.read())
        
        with zipf.open(vertices_filename) as f:
            vertices = f.read()
        
        indices = None
        if indices_filename in zipf.namelist():
            with zipf.open(indices_filename) as f:
                indices = f.read()
        
        return EncodedMesh(
            vertices=vertices,
            indices=indices,
            vertex_count=metadata.vertex_count,
            vertex_size=metadata.vertex_size,
            index_count=metadata.index_count,
            index_size=metadata.index_size
        )

def save_encoded_array_to_zip(encoded_array: EncodedArray,
                            zip_path: str,
                            data_filename: str = "data.bin",
                            metadata_filename: str = "metadata.json") -> None:
    """Save an encoded array to a zip file."""
    metadata = ArrayMetadata(
        shape=encoded_array.shape,
        dtype=str(encoded_array.dtype),
        itemsize=encoded_array.itemsize
    )
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(data_filename, encoded_array.data)
        zipf.writestr(metadata_filename, metadata.model_dump_json(indent=2))

def load_encoded_array_from_zip(zip_path: str,
                               data_filename: str = "data.bin",
                               metadata_filename: str = "metadata.json") -> EncodedArray:
    """Load an encoded array from a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(metadata_filename) as f:
            metadata = ArrayMetadata.model_validate_json(f.read())
        
        with zipf.open(data_filename) as f:
            data = f.read()
    
    return EncodedArray(
        data=data,
        shape=metadata.shape,
        dtype=np.dtype(metadata.dtype),
        itemsize=metadata.itemsize
    )

def save_arrays_to_zip(arrays: Dict[str, np.ndarray], zip_path: str) -> None:
    """
    Encode and save multiple numpy arrays to a zip file.
    
    Args:
        arrays: Dictionary mapping array names to numpy arrays
        zip_path: Path to the output zip file
    """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        all_metadata = ArraysMetadata.create_empty()
        
        for name, array in arrays.items():
            # Encode the array
            encoded = encode_array(array)
            
            all_metadata[name] = ArrayMetadata(
                shape=encoded.shape,
                dtype=str(encoded.dtype),
                itemsize=encoded.itemsize,
                filename=f"{name}.bin"
            )
            zipf.writestr(f"{name}.bin", encoded.data)
        
        zipf.writestr("metadata.json", all_metadata.model_dump_json(indent=2))

def load_arrays_from_zip(zip_path: str) -> Dict[str, np.ndarray]:
    """
    Load and decode multiple arrays from a zip file.
    
    Args:
        zip_path: Path to the input zip file
        
    Returns:
        Dictionary mapping array names to decoded numpy arrays
    """
    result = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open("metadata.json") as f:
            all_metadata = ArraysMetadata.model_validate_json(f.read())
        
        for name, metadata in all_metadata.items():
            with zipf.open(metadata.filename) as f:
                data = f.read()
            
            # Create encoded array
            encoded = EncodedArray(
                data=data,
                shape=metadata.shape,
                dtype=np.dtype(metadata.dtype),
                itemsize=metadata.itemsize
            )
            
            # Decode array before adding to result
            result[name] = decode_array(encoded)
    
    return result

def save_array_to_zip(array: np.ndarray, zip_path: str) -> None:
    """
    Encode and save a numpy array to a zip file.
    
    Args:
        array: numpy array to encode and save
        zip_path: Path to the output zip file
    """
    encoded_array = encode_array(array)
    save_encoded_array_to_zip(encoded_array, zip_path)

def load_array_from_zip(zip_path: str) -> np.ndarray:
    """
    Load and decode a numpy array from a zip file.
    
    Args:
        zip_path: Path to the input zip file
        
    Returns:
        Decoded numpy array
    """
    encoded_array = load_encoded_array_from_zip(zip_path)
    return decode_array(encoded_array)

def save_mesh_to_zip(mesh, zip_path: str) -> None:
    """
    Encode and save a mesh to a zip file.
    
    Args:
        mesh: Mesh object to encode and save
        zip_path: Path to the output zip file
    """
    encoded_mesh = mesh.encode()
    save_encoded_mesh_to_zip(encoded_mesh, zip_path)

def load_mesh_from_zip(mesh_class, zip_path: str) -> 'Mesh':
    """
    Load a mesh from a zip file.
    
    Args:
        mesh_class: Class to use for creating the mesh (e.g., Mesh)
        zip_path: Path to the input zip file
        
    Returns:
        Mesh object loaded from the zip file
    """
    encoded_mesh = load_encoded_mesh_from_zip(zip_path)
    return mesh_class.decode(encoded_mesh)

def save_combined_data_to_zip(
    encoded_mesh: EncodedMesh,
    encoded_arrays: Dict[str, EncodedArray],
    metadata: Optional[Dict[str, Any]] = None,
    zip_path: str = "combined_data.zip",
    mesh_dir: str = "mesh",
    arrays_dir: str = "arrays"
) -> None:
    """Save an encoded mesh and multiple encoded arrays to a single zip file."""
    if encoded_mesh is None:
        raise ValueError("encoded_mesh cannot be None")
    
    if not encoded_arrays:
        raise ValueError("encoded_arrays dictionary cannot be empty")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Save the encoded mesh
            zipf.writestr(f"{mesh_dir}/vertices.bin", encoded_mesh.vertices)
            zipf.writestr(f"{mesh_dir}/indices.bin", encoded_mesh.indices)
            
            mesh_metadata = MeshMetadata(
                vertex_count=encoded_mesh.vertex_count,
                vertex_size=encoded_mesh.vertex_size,
                index_count=encoded_mesh.index_count,
                index_size=encoded_mesh.index_size
            )
            zipf.writestr(f"{mesh_dir}/metadata.json", mesh_metadata.model_dump_json(indent=2))
            
            # Save the encoded arrays
            arrays_metadata = ArraysMetadata.create_empty()
            for name, encoded_array in encoded_arrays.items():
                zipf.writestr(f"{arrays_dir}/{name}.bin", encoded_array.data)
                
                arrays_metadata[name] = ArrayMetadata(
                    shape=encoded_array.shape,
                    dtype=str(encoded_array.dtype),
                    itemsize=encoded_array.itemsize
                )
            
            zipf.writestr(f"{arrays_dir}/metadata.json", arrays_metadata.model_dump_json(indent=2))
            
            if metadata is not None:
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
    except Exception as e:
        raise IOError(f"Error saving combined data to zip file: {str(e)}")


def get_combined_data_as_bytes(
    encoded_mesh: EncodedMesh,
    encoded_arrays: Dict[str, EncodedArray],
    metadata: Optional[Dict[str, Any]] = None,
    mesh_dir: str = "mesh",
    arrays_dir: str = "arrays"
) -> bytes:
    """
    Create a zip file in memory containing an encoded mesh and multiple encoded arrays.
    
    Args:
        encoded_mesh: EncodedMesh object to include in the zip
        encoded_arrays: Dictionary mapping names to EncodedArray objects
        metadata: Optional general metadata to include in the zip
        mesh_dir: Directory name for mesh data within the zip (default: "mesh")
        arrays_dir: Directory name for array data within the zip (default: "arrays")
        
    Returns:
        Bytes object containing the zip file data
        
    Raises:
        ValueError: If encoded_mesh is None or encoded_arrays is empty
        IOError: If there's an error creating the zip data
    """
    if encoded_mesh is None:
        raise ValueError("encoded_mesh cannot be None")
    
    if not encoded_arrays:
        raise ValueError("encoded_arrays dictionary cannot be empty")
    
    try:
        # Create an in-memory file-like object
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Save the encoded mesh
            zipf.writestr(f"{mesh_dir}/vertices.bin", encoded_mesh.vertices)
            zipf.writestr(f"{mesh_dir}/indices.bin", encoded_mesh.indices)
            
            mesh_metadata = MeshMetadata(
                vertex_count=encoded_mesh.vertex_count,
                vertex_size=encoded_mesh.vertex_size,
                index_count=encoded_mesh.index_count,
                index_size=encoded_mesh.index_size
            )
            zipf.writestr(f"{mesh_dir}/metadata.json", mesh_metadata.model_dump_json(indent=2))
            
            # Save the encoded arrays
            arrays_metadata = ArraysMetadata.create_empty()
            for name, encoded_array in encoded_arrays.items():
                zipf.writestr(f"{arrays_dir}/{name}.bin", encoded_array.data)
                
                arrays_metadata[name] = ArrayMetadata(
                    shape=encoded_array.shape,
                    dtype=str(encoded_array.dtype),
                    itemsize=encoded_array.itemsize
                )
            
            zipf.writestr(f"{arrays_dir}/metadata.json", arrays_metadata.model_dump_json(indent=2))
            
            if metadata is not None:
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        # Get the bytes from the buffer
        buffer.seek(0)
        return buffer.getvalue()
                
    except Exception as e:
        raise IOError(f"Error creating combined data zip bytes: {str(e)}")


def load_combined_data_from_zip(
    zip_path: Union[str, bytes, io.BytesIO],
    mesh_dir: str = "mesh",
    arrays_dir: str = "arrays"
) -> Tuple[EncodedMesh, Dict[str, EncodedArray], Optional[Dict[str, Any]]]:
    """
    Load an encoded mesh and multiple encoded arrays from a zip file or bytes.
    
    Args:
        zip_path: Path to the input zip file, bytes object containing zip data,
                 or BytesIO object containing zip data
        mesh_dir: Directory name for mesh data within the zip (default: "mesh")
        arrays_dir: Directory name for array data within the zip (default: "arrays")
        
    Returns:
        Tuple containing:
        - EncodedMesh object
        - Dictionary mapping array names to EncodedArray objects
        - General metadata dictionary (or None if not present)
        
    Raises:
        FileNotFoundError: If the zip file doesn't exist (when path is provided)
        ValueError: If the zip file doesn't contain the expected structure
        IOError: If there's an error reading from the zip file
        TypeError: If the zip_path is of an unsupported type
    """
    try:
        # Handle different input types, similar to how get_combined_data_as_bytes creates a buffer
        if isinstance(zip_path, str):
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            zip_source = zip_path
        elif isinstance(zip_path, bytes):
            zip_source = io.BytesIO(zip_path)
        elif isinstance(zip_path, io.BytesIO):
            zip_source = zip_path
        else:
            raise TypeError(f"Unsupported zip_path type: {type(zip_path)}")
        
        with zipfile.ZipFile(zip_source, 'r') as zipf:
            # Load mesh data
            mesh_vertices_data = zipf.read(f"{mesh_dir}/vertices.bin")
            mesh_indices_data = zipf.read(f"{mesh_dir}/indices.bin")
            mesh_metadata = MeshMetadata.model_validate_json(zipf.read(f"{mesh_dir}/metadata.json"))
            
            # Create EncodedMesh object
            encoded_mesh = EncodedMesh(
                vertices=mesh_vertices_data,
                indices=mesh_indices_data,
                vertex_count=mesh_metadata.vertex_count,
                vertex_size=mesh_metadata.vertex_size,
                index_count=mesh_metadata.index_count,
                index_size=mesh_metadata.index_size
            )
            
            # Load array metadata
            arrays_metadata = ArraysMetadata.model_validate_json(zipf.read(f"{arrays_dir}/metadata.json"))
            
            # Load array data
            encoded_arrays = {}
            for name, metadata in arrays_metadata.items():
                array_data = zipf.read(f"{arrays_dir}/{name}.bin")
                
                encoded_arrays[name] = EncodedArray(
                    data=array_data,
                    shape=metadata.shape,
                    dtype=np.dtype(metadata.dtype),
                    itemsize=metadata.itemsize
                )
            
            # Load general metadata if present
            general_metadata = None
            try:
                general_metadata = json.loads(zipf.read("metadata.json"))
            except KeyError:
                # Metadata file not found, that's okay
                pass
            
            return encoded_mesh, encoded_arrays, general_metadata
            
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid zip file: {zip_path}")
    except KeyError as e:
        raise ValueError(f"Missing expected file in zip: {str(e)}")
    except Exception as e:
        raise IOError(f"Error loading combined data from zip file: {str(e)}")