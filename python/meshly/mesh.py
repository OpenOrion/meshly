"""
High-level mesh abstraction for easier use of meshoptimizer.

This module provides:
1. Mesh class as a Pydantic base class for representing 3D meshes
2. MeshUtils class for mesh optimization and encoding/decoding operations
3. Functions for encoding and decoding meshes
"""

import json
import zipfile
from io import BytesIO
from typing import (
    Dict,
    Optional,
    Set,
    Type,
    Any,
    TypeVar,
    Union,
    List,
    get_type_hints,
)
import numpy as np
from pydantic import BaseModel, Field, model_validator

# Use meshoptimizer directly
from meshoptimizer import (
    # Encoder functions
    encode_vertex_buffer,
    encode_index_sequence,
    decode_vertex_buffer,
    decode_index_sequence,
    optimize_vertex_cache,
    optimize_overdraw,
    optimize_vertex_fetch,
    simplify,
)

from .array import ArrayMetadata, EncodedArray, ArrayUtils
from .common import PathLike

# Type variable for the Mesh class
T = TypeVar("T", bound="Mesh")

# Removed ARRAY_NAME_SEPERATOR - now using nested directories


class EncodedMesh(BaseModel):
    """
    Pydantic model representing an encoded mesh with its vertices and indices.

    This is a Pydantic version of the EncodedMesh class in mesh.py.
    """

    vertices: bytes = Field(..., description="Encoded vertex buffer")
    indices: Optional[bytes] = Field(
        None, description="Encoded index buffer (optional)"
    )
    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")
    index_sizes: Optional[bytes] = Field(
        None, description="Encoded polygon sizes (optional)"
    )
    arrays: Dict[str, EncodedArray] = Field(
        default_factory=dict, description="Dictionary of additional encoded arrays"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class MeshSize(BaseModel):
    """
    Pydantic model representing size metadata for an encoded mesh.

    Used in the save_to_zip method to store mesh size information.
    """

    vertex_count: int = Field(..., description="Number of vertices")
    vertex_size: int = Field(..., description="Size of each vertex in bytes")
    index_count: Optional[int] = Field(None, description="Number of indices (optional)")
    index_size: int = Field(..., description="Size of each index in bytes")


class MeshMetadata(BaseModel):
    """
    Pydantic model representing general metadata for a mesh file.

    Used in the save_to_zip method to store general metadata.
    """

    class_name: str = Field(..., description="Name of the mesh class")
    module_name: str = Field(
        ..., description="Name of the module containing the mesh class"
    )
    field_data: Optional[Dict[str, Any]] = Field(
        None, description="Dictionary of model fields that aren't numpy arrays"
    )
    mesh_size: MeshSize = Field(description="Size metadata for the encoded mesh")


class Mesh(BaseModel):
    """
    A Pydantic base class representing a 3D mesh.

    Users can inherit from this class to define custom mesh types with additional
    numpy array attributes that will be automatically encoded/decoded.
    """

    # Required fields
    vertices: np.ndarray = Field(..., description="Vertex data as a numpy array")
    indices: Optional[Union[np.ndarray, List[Any]]] = Field(
        None, description="Index data as a flattened 1D numpy array or list of polygons"
    )
    index_sizes: Optional[Union[np.ndarray, List[int]]] = Field(
        None, description="Size of each polygon (number of vertices per polygon). "
        "If not provided, will be automatically inferred from indices structure: "
        "- For 2D numpy arrays: uniform polygon size from array shape "
        "- For list of lists: individual polygon sizes "
        "If explicitly provided, will be validated against inferred structure."
    )
    
    def copy(self: T) -> T:
        """
        Create a deep copy of the mesh.
        
        Returns:
            A new mesh instance with copied data
        """
        # Create a dictionary to hold the copied fields
        copied_fields = {}
        
        # Copy all fields, with special handling for numpy arrays
        for field_name in self.model_fields_set:
            value = getattr(self, field_name)
            if isinstance(value, np.ndarray):
                copied_fields[field_name] = value.copy()
            else:
                copied_fields[field_name] = value
                
        # Create a new instance of the same class
        return self.__class__(**copied_fields)

    @property
    def vertex_count(self) -> int:
        """Get the number of vertices."""
        return len(self.vertices)

    @property
    def index_count(self) -> int:
        """Get the number of indices."""
        return len(self.indices) if self.indices is not None else 0

    @property
    def polygon_count(self) -> int:
        """Get the number of polygons."""
        return len(self.index_sizes) if self.index_sizes is not None else 0

    @property
    def is_uniform_polygons(self) -> bool:
        """Check if all polygons have the same number of vertices."""
        if self.index_sizes is None:
            return True  # No polygon info means uniform (legacy)
        return len(np.unique(self.index_sizes)) == 1

    def get_polygon_indices(self) -> Union[np.ndarray, list]:
        """
        Get indices in their original polygon structure.
        
        Returns:
            For uniform polygons: 2D numpy array where each row is a polygon
            For mixed polygons: List of lists where each sublist is a polygon
        """
        if self.indices is None:
            return None
            
        if self.index_sizes is None:
            # Legacy format - assume triangles
            if len(self.indices) % 3 == 0:
                return self.indices.reshape(-1, 3)
            else:
                raise ValueError("Cannot determine polygon structure without index_sizes")
        
        if self.is_uniform_polygons:
            # Uniform case: return as 2D numpy array
            polygon_size = self.index_sizes[0]
            return self.indices.reshape(-1, polygon_size)
        else:
            # Mixed case: return as list of lists
            result = []
            offset = 0
            for size in self.index_sizes:
                result.append(self.indices[offset:offset+size].tolist())
                offset += size
            return result

    def _extract_nested_arrays(self, obj: Any, prefix: str = "") -> Dict[str, np.ndarray]:
        """
        Recursively extract numpy arrays from nested structures.
        
        Args:
            obj: The object to extract arrays from
            prefix: The current path prefix for nested keys
            
        Returns:
            Dictionary mapping dotted paths to numpy arrays
        """
        arrays = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                nested_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, np.ndarray):
                    arrays[nested_key] = value
                elif isinstance(value, dict):
                    arrays.update(self._extract_nested_arrays(value, nested_key))
        elif isinstance(obj, np.ndarray):
            arrays[prefix] = obj
            
        return arrays


    @property
    def array_fields(self) -> Set[str]:
        """Identify all numpy array fields in this class, including nested arrays in dictionaries."""
        result = set()
        type_hints = get_type_hints(self.__class__)

        # Find all fields that are numpy arrays or contain numpy arrays
        for field_name, field_type in type_hints.items():
            if field_name in self.__private_attributes__:
                continue
            try:
                value = getattr(self, field_name, None)
                if isinstance(value, np.ndarray):
                    result.add(field_name)
                elif isinstance(value, dict):
                    # Extract nested arrays and add them with dotted notation
                    nested_arrays = self._extract_nested_arrays(value, field_name)
                    result.update(nested_arrays.keys())
            except AttributeError:
                # Skip attributes that don't exist
                pass

        return result


    class Config:
        arbitrary_types_allowed = True



    @model_validator(mode="after")
    def validate_arrays(self) -> "Mesh":
        """
        Validate and convert arrays to the correct types.
        
        This method handles various input formats for indices and automatically infers
        index_sizes when not explicitly provided:
        
        - 2D numpy arrays: Assumes uniform polygons, infers size from array shape
        - List of lists: Supports mixed polygon sizes, infers from individual polygon lengths
        - Flat arrays: Requires explicit index_sizes for polygon structure
        
        When index_sizes is explicitly provided, it validates that the structure matches
        the inferred polygon sizes and that the sum equals the total number of indices.
        
        Raises:
            ValueError: If explicit index_sizes doesn't match inferred structure or
                       if sum of index_sizes doesn't match total indices count.
        """
        # Ensure vertices is a float32 array
        if self.vertices is not None:
            self.vertices = np.asarray(self.vertices, dtype=np.float32)

        # Handle indices - convert to flattened 1D array and extract size info
        if self.indices is not None:
            # Store original index_sizes if explicitly provided for validation
            explicit_index_sizes = self.index_sizes
            inferred_index_sizes = None
            
            # Convert various input formats to flattened indices
            if isinstance(self.indices, (list, tuple)):
                # Handle list of lists (mixed polygons) or flat list
                if len(self.indices) > 0 and isinstance(self.indices[0], (list, tuple, np.ndarray)):
                    # List of lists - mixed polygon format
                    flat_indices = []
                    sizes = []
                    for polygon in self.indices:
                        polygon_array = np.asarray(polygon, dtype=np.uint32)
                        flat_indices.extend(polygon_array.flatten())
                        sizes.append(len(polygon_array.flatten()))
                    self.indices = np.array(flat_indices, dtype=np.uint32)
                    inferred_index_sizes = np.array(sizes, dtype=np.uint32)
                    
                    # Set index_sizes only if not explicitly provided
                    if self.index_sizes is None:
                        self.index_sizes = inferred_index_sizes
                else:
                    # Flat list
                    self.indices = np.asarray(self.indices, dtype=np.uint32).flatten()
            else:
                # Numpy array input
                original_shape = self.indices.shape
                self.indices = np.asarray(self.indices, dtype=np.uint32).flatten()
                
                # If it was a 2D array, extract size information
                if len(original_shape) > 1:
                    # Infer uniform polygons from 2D array shape
                    inferred_index_sizes = np.full(original_shape[0], original_shape[1], dtype=np.uint32)
                    
                    # Set index_sizes only if not explicitly provided
                    if self.index_sizes is None:
                        self.index_sizes = inferred_index_sizes

            # Ensure index_sizes is uint32 array if present
            if self.index_sizes is not None:
                self.index_sizes = np.asarray(self.index_sizes, dtype=np.uint32)
                
                # Validate that explicit index_sizes matches inferred structure if both exist
                if explicit_index_sizes is not None and inferred_index_sizes is not None:
                    explicit_array = np.asarray(explicit_index_sizes, dtype=np.uint32)
                    if not np.array_equal(explicit_array, inferred_index_sizes):
                        raise ValueError(
                            f"Explicit index_sizes {explicit_array.tolist()} does not match "
                            f"inferred structure {inferred_index_sizes.tolist()}"
                        )
                
                # Validate that index_sizes sum matches total indices
                if len(self.indices) != np.sum(self.index_sizes):
                    raise ValueError(
                        f"Sum of index_sizes ({np.sum(self.index_sizes)}) does not match "
                        f"total number of indices ({len(self.indices)})"
                    )

        return self


class MeshUtils:
    """
    Utility class for mesh optimization and encoding/decoding operations.
    """

    @staticmethod
    def optimize_vertex_cache(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex cache efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()
        
        optimized_indices = np.zeros_like(result_mesh.indices)
        optimize_vertex_cache(
            optimized_indices, result_mesh.indices, result_mesh.index_count, result_mesh.vertex_count
        )

        result_mesh.indices = optimized_indices
        return result_mesh

    @staticmethod
    def optimize_overdraw(mesh: Mesh, threshold: float = 1.05) -> Mesh:
        """
        Optimize the mesh for overdraw.

        Args:
            mesh: The mesh to optimize
            threshold: threshold for optimization (default: 1.05)

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()
        
        optimized_indices = np.zeros_like(result_mesh.indices)
        optimize_overdraw(
            optimized_indices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
            threshold,
        )

        result_mesh.indices = optimized_indices
        return result_mesh

    @staticmethod
    def optimize_vertex_fetch(mesh: Mesh) -> Mesh:
        """
        Optimize the mesh for vertex fetch efficiency.

        Args:
            mesh: The mesh to optimize

        Returns:
            A new optimized mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to optimize")

        # Create a copy of the mesh
        result_mesh = mesh.copy()
        
        optimized_vertices = np.zeros_like(result_mesh.vertices)
        unique_vertex_count = optimize_vertex_fetch(
            optimized_vertices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
        )

        result_mesh.vertices = optimized_vertices[:unique_vertex_count]
        # No need to update vertex_count as it's calculated on-the-fly
        return result_mesh

    @staticmethod
    def simplify(
        mesh: Mesh,
        target_ratio: float = 0.25,
        target_error: float = 0.01,
        options: int = 0,
    ) -> Mesh:
        """
        Simplify the mesh.

        Args:
            mesh: The mesh to simplify
            target_ratio: ratio of triangles to keep (default: 0.25)
            target_error: target error (default: 0.01)
            options: simplification options (default: 0)

        Returns:
            A new simplified mesh, original mesh is unchanged
        """
        if mesh.indices is None:
            raise ValueError("Mesh has no indices to simplify")

        # Create a copy of the mesh
        result_mesh = mesh.copy()
        
        target_index_count = int(result_mesh.index_count * target_ratio)
        simplified_indices = np.zeros(result_mesh.index_count, dtype=np.uint32)

        result_error = np.array([0.0], dtype=np.float32)
        new_index_count = simplify(
            simplified_indices,
            result_mesh.indices,
            result_mesh.vertices,
            result_mesh.index_count,
            result_mesh.vertex_count,
            result_mesh.vertices.itemsize * result_mesh.vertices.shape[1],
            target_index_count,
            target_error,
            options,
            result_error,
        )

        result_mesh.indices = simplified_indices[:new_index_count]
        # No need to update index_count as it's calculated on-the-fly
        return result_mesh

    @staticmethod
    def encode(mesh: Mesh):
        """
        Encode the mesh and all numpy array fields for efficient transmission.

        Args:
            mesh: The mesh to encode

        Returns:
            EncodedMesh object with encoded vertices, indices, and arrays
        """
        # Encode vertex buffer
        encoded_vertices = encode_vertex_buffer(
            mesh.vertices,
            mesh.vertex_count,
            mesh.vertices.itemsize * mesh.vertices.shape[1],
        )

        # Encode index buffer if present
        encoded_indices = None
        if mesh.indices is not None:
            encoded_indices = encode_index_sequence(
                mesh.indices, mesh.index_count, mesh.vertex_count
            )

        # Note: index_sizes will be encoded as a regular array in the arrays dict
        encoded_index_sizes = None

        # Encode additional array fields, including nested arrays from dictionaries
        encoded_arrays = {}
        for field_name in mesh.array_fields:
            if field_name in ("vertices", "indices"):
                continue  # Skip the main vertices and indices

            # Handle nested array paths (e.g., "textures.diffuse")
            if "." in field_name:
                # Extract the nested array
                parts = field_name.split(".")
                obj = mesh
                for part in parts[:-1]:
                    if isinstance(obj, dict):
                        obj = obj[part]
                    else:
                        obj = getattr(obj, part)
                
                # Get the final array
                if isinstance(obj, dict):
                    array = obj[parts[-1]]
                else:
                    array = getattr(obj, parts[-1])
                    
                if isinstance(array, np.ndarray):
                    encoded_arrays[field_name] = ArrayUtils.encode_array(array)
            else:
                # Handle direct array fields
                try:
                    array = getattr(mesh, field_name)
                    if isinstance(array, np.ndarray):
                        encoded_arrays[field_name] = ArrayUtils.encode_array(array)
                except AttributeError:
                    # Skip attributes that don't exist
                    pass

        # Handle index_sizes as a special case - store in arrays if present
        if mesh.index_sizes is not None:
            encoded_arrays["index_sizes"] = ArrayUtils.encode_array(mesh.index_sizes)

        # Create encoded mesh
        return EncodedMesh(
            vertices=encoded_vertices,
            indices=encoded_indices,
            vertex_count=mesh.vertex_count,
            vertex_size=mesh.vertices.itemsize * mesh.vertices.shape[1],
            index_count=mesh.index_count if mesh.indices is not None else None,
            index_size=mesh.indices.itemsize if mesh.indices is not None else 4,
            index_sizes=encoded_index_sizes,
            arrays=encoded_arrays,
        )

    @staticmethod
    def save_to_zip(mesh: Mesh, source: Union[PathLike, BytesIO], date_time: Optional[tuple] = None) -> None:
        """
        Save the mesh to a zip file.

        Args:
            mesh: The mesh to save
            source: Path to the output zip file
        """
        encoded_mesh = MeshUtils.encode(mesh)

        # Add model fields that aren't numpy arrays, preserving non-array values in dicts
        model_data = {}
        
        def extract_non_arrays(obj: Any, prefix: str = "") -> Any:
            """Recursively extract non-array values from nested structures."""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    nested_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, np.ndarray):
                        # Skip arrays - they're stored separately
                        continue
                    elif isinstance(value, dict):
                        # Recursively process nested dicts
                        nested_result = extract_non_arrays(value, nested_key)
                        if nested_result:  # Only include non-empty dicts
                            result[key] = nested_result
                    else:
                        # Include non-array values
                        result[key] = value
                return result if result else None
            elif isinstance(obj, np.ndarray):
                # Skip arrays
                return None
            else:
                # Include non-array values
                return obj
        
        for field_name, field_value in mesh.model_dump().items():
            # Skip direct array fields (they're stored separately)
            if field_name in mesh.array_fields:
                continue
            
            if isinstance(field_value, dict):
                # Extract non-array values from dictionaries
                non_array_content = extract_non_arrays(field_value, field_name)
                if non_array_content:
                    model_data[field_name] = non_array_content
            else:
                # Include scalar fields directly
                model_data[field_name] = field_value


        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Create mesh size metadata
            mesh_size = MeshSize(
                vertex_count=encoded_mesh.vertex_count,
                vertex_size=encoded_mesh.vertex_size,
                index_count=encoded_mesh.index_count,
                index_size=encoded_mesh.index_size,
            )

            metadata = MeshMetadata(
                class_name=mesh.__class__.__name__,
                module_name=mesh.__class__.__module__,
                mesh_size=mesh_size,
                field_data=model_data,
            )

            # Collect all files to write in sorted order for deterministic output
            files_to_write = []
            
            # Add mesh data
            files_to_write.append(("mesh/vertices.bin", encoded_mesh.vertices))
            if encoded_mesh.indices is not None:
                files_to_write.append(("mesh/indices.bin", encoded_mesh.indices))

            # Add array data (sorted by name for deterministic order)
            for name in sorted(encoded_mesh.arrays.keys()):
                encoded_array = encoded_mesh.arrays[name]
                # Convert dots to nested directory structure, each array gets its own directory
                array_path = name.replace(".", "/")
                files_to_write.append((f"arrays/{array_path}/array.bin", encoded_array.data))

                # Save array metadata
                array_metadata = ArrayMetadata(
                    shape=list(encoded_array.shape),
                    dtype=str(encoded_array.dtype),
                    itemsize=encoded_array.itemsize,
                )
                files_to_write.append((
                    f"arrays/{array_path}/metadata.json",
                    json.dumps(array_metadata.model_dump(), indent=2, sort_keys=True)
                ))

            # Add general metadata
            files_to_write.append(("metadata.json", json.dumps(metadata.model_dump(), indent=2, sort_keys=True)))

            # Sort files by path for deterministic order and write them
            for filename, data in sorted(files_to_write):
                if date_time is not None:
                    info = zipfile.ZipInfo(filename=filename, date_time=date_time)
                else:
                    info = zipfile.ZipInfo(filename=filename)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16  # Fixed file permissions
                if isinstance(data, str):
                    data = data.encode('utf-8')
                zipf.writestr(info, data)



    @staticmethod
    def load_from_zip(cls: Type[T], destination: Union[PathLike, BytesIO]) -> T:
        """
        Load a mesh from a zip file.

        Args:
            cls: The mesh class to instantiate
            destination: Path to the input zip file

        Returns:
            Mesh object loaded from the zip file
        """
        with zipfile.ZipFile(destination, "r") as zipf:
            # Load general metadata
            with zipf.open("metadata.json") as f:
                metadata_dict = json.loads(f.read().decode("utf-8"))
                metadata = MeshMetadata(**metadata_dict)

            # Check if the class matches
            class_name = metadata.class_name
            module_name = metadata.module_name

            # If the class doesn't match, try to import it
            if class_name != cls.__name__ or module_name != cls.__module__:
                raise ValueError(
                    f"Class mismatch: expected {cls.__name__} but got {class_name} from {module_name}"
                )
            else:
                target_cls = cls

            # Get mesh size metadata from the file metadata
            mesh_size = metadata.mesh_size

            # Load mesh data
            with zipf.open("mesh/vertices.bin") as f:
                encoded_vertices = f.read()

            encoded_indices = None
            if "mesh/indices.bin" in zipf.namelist():
                with zipf.open("mesh/indices.bin") as f:
                    encoded_indices = f.read()

            # Create encoded mesh model
            encoded_mesh = EncodedMesh(
                vertices=encoded_vertices,
                indices=encoded_indices,
                vertex_count=mesh_size.vertex_count,
                vertex_size=mesh_size.vertex_size,
                index_count=mesh_size.index_count,
                index_size=mesh_size.index_size,
                index_sizes=None,  # Not used anymore
                arrays={},  # Will be populated below
            )

            # Load additional array data
            for array_file_name in [
                file_name
                for file_name in zipf.namelist()
                if file_name.startswith("arrays/") and file_name.endswith("/array.bin")
            ]:
                # Extract the array path by removing "arrays/" prefix and "/array.bin" suffix
                array_path = array_file_name[7:-10]  # Remove "arrays/" and "/array.bin"

                # Load array metadata
                with zipf.open(f"arrays/{array_path}/metadata.json") as f:
                    array_metadata_dict = json.loads(f.read().decode("utf-8"))
                    array_metadata = ArrayMetadata(**array_metadata_dict)

                # Load array data
                with zipf.open(array_file_name) as f:
                    encoded_data = f.read()

                # Create encoded array
                encoded_array = EncodedArray(
                    data=encoded_data,
                    shape=tuple(array_metadata.shape),
                    dtype=np.dtype(array_metadata.dtype),
                    itemsize=array_metadata.itemsize,
                )

                # Convert directory path back to dotted name
                original_name = array_path.replace("/", ".")

                # Add to encoded mesh arrays with original name
                encoded_mesh.arrays[original_name] = encoded_array

            # Decode the mesh using MeshUtils.decode
            decoded_mesh = MeshUtils.decode(target_cls, encoded_mesh)

            # Add any additional field data from the metadata, merging with existing dict fields
            if metadata.field_data:
                for field_name, field_value in metadata.field_data.items():
                    existing_value = getattr(decoded_mesh, field_name, None)
                    if existing_value is not None and isinstance(existing_value, dict) and isinstance(field_value, dict):
                        # Merge non-array values into existing dictionary structure
                        def merge_dicts(target: dict, source: dict) -> None:
                            """Recursively merge source dict into target dict."""
                            for key, value in source.items():
                                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                                    merge_dicts(target[key], value)
                                else:
                                    target[key] = value
                        
                        merge_dicts(existing_value, field_value)
                    else:
                        # Set scalar fields directly
                        setattr(decoded_mesh, field_name, field_value)

            return decoded_mesh

    @staticmethod
    def decode(cls: Type[T], encoded_mesh: EncodedMesh) -> T:
        """
        Decode an encoded mesh.

        Args:
            cls: The mesh class to instantiate
            encoded_mesh: EncodedMesh object to decode

        Returns:
            Decoded Mesh object
        """
        # Decode vertex buffer
        vertices = decode_vertex_buffer(
            encoded_mesh.vertex_count, encoded_mesh.vertex_size, encoded_mesh.vertices
        )

        # Decode index buffer if present
        indices = None
        if encoded_mesh.indices is not None and encoded_mesh.index_count is not None:
            indices = decode_index_sequence(
                encoded_mesh.index_count, encoded_mesh.index_size, encoded_mesh.indices
            )

        # Create the base mesh object
        mesh_args = {
            "vertices": vertices,
            "indices": indices,
        }

        # Decode index_sizes if present (stored in arrays dict)
        index_sizes = None

        # Decode additional arrays if present
        dict_fields = {}  # Will store reconstructed dictionary fields
        
        if encoded_mesh.arrays:
            for name, encoded_array in encoded_mesh.arrays.items():
                # Decode the array
                decoded_array = ArrayUtils.decode_array(encoded_array)
                
                # Check if this is a nested array (contains dots)
                if name == "index_sizes":
                    # Special handling for index_sizes
                    index_sizes = decoded_array
                elif "." in name:
                    # This is a nested array from a dictionary field
                    parts = name.split(".")
                    field_name = parts[0]
                    
                    # Initialize the dictionary field if not exists
                    if field_name not in dict_fields:
                        dict_fields[field_name] = {}
                    
                    # Navigate/create the nested structure
                    current_dict = dict_fields[field_name]
                    for part in parts[1:-1]:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]
                    
                    # Set the final array
                    current_dict[parts[-1]] = decoded_array
                else:
                    # This is a direct array field
                    mesh_args[name] = decoded_array

        # Add index_sizes to mesh args
        mesh_args["index_sizes"] = index_sizes

        # Add reconstructed dictionary fields to mesh arguments
        mesh_args.update(dict_fields)

        # Create and return the mesh object
        return cls(**mesh_args)
