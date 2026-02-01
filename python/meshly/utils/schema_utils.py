"""Schema utilities for resolving Pydantic types and merging field data."""

import types
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel

from ..array import ArrayRefMetadata, ArrayType, ArrayUtils, EncodingType
from ..data_handler import AssetProvider
from .json_schema import JsonSchema, JsonSchemaProperty
from .serialization_utils import SerializationUtils

# Array types as they appear in schema.json
ARRAY_SCHEMA_TYPES = {"array", "vertex_buffer", "index_sequence"}


class SchemaUtils:
    """Utility class for Pydantic schema operations."""

    @staticmethod
    def unwrap_optional(expected_type: Any) -> Any:
        """Unwrap Optional[X] or X | None to X.

        Args:
            expected_type: Type annotation, possibly Optional

        Returns:
            Inner type if Optional, otherwise unchanged
        """
        origin = get_origin(expected_type)
        # Handle both typing.Union and types.UnionType (X | Y syntax)
        if origin is Union or isinstance(expected_type, types.UnionType):
            args = get_args(expected_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return expected_type

    @staticmethod
    def get_field_encoding(model_class: type[BaseModel], field_name: str) -> EncodingType:
        """Get the encoding type for a field from its type annotation.
        
        Uses get_array_encoding to extract encoding from Annotated types
        (Array, VertexBuffer, IndexSequence).
        
        Args:
            model_class: Pydantic model class
            field_name: Name of the field
            
        Returns:
            Encoding type (defaults to "array")
        """
        import typing
        hints = typing.get_type_hints(model_class, include_extras=True)
        if field_name in hints:
            return ArrayUtils.get_array_encoding(hints[field_name])
        return "array"

    @staticmethod
    def resolve_refs_with_schema(
        model_class: type[BaseModel],
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: ArrayType | None,
    ) -> dict[str, Any]:
        """Resolve $ref references using Pydantic schema for type information.

        Args:
            model_class: Pydantic model class with field definitions
            data: Data dict with potential $ref values
            assets: Asset provider
            array_type: Target array type

        Returns:
            Resolved data dict
        """
        result = {}

        for field_name, field_info in model_class.model_fields.items():
            if field_name not in data:
                continue

            # Get encoding from schema for this field
            encoding = SchemaUtils.get_field_encoding(model_class, field_name)

            result[field_name] = SchemaUtils.resolve_value_with_type(
                data[field_name], field_info.annotation, assets, array_type, encoding
            )

        return result

    @staticmethod
    def resolve_value_with_type(
        value: Any,
        expected_type: Any,
        assets: AssetProvider,
        array_type: ArrayType | None,
        encoding: EncodingType = "array",
    ) -> Any:
        """Resolve a value using the expected type from Pydantic schema.

        Args:
            value: Value to resolve
            expected_type: Expected type from schema
            assets: Asset provider
            array_type: Target array type
            encoding: Encoding type from schema.json

        Returns:
            Resolved value
        """
        # Import here to avoid circular imports
        from ..packable import Packable

        if value is None:
            return None

        # Handle $ref
        if isinstance(value, dict) and "$ref" in value:
            checksum = value["$ref"]

            expected_type = SchemaUtils.unwrap_optional(expected_type)
            origin = get_origin(expected_type)

            # Check if it's a ResourceRef - don't unpack, just return the dict
            from ..resource import ResourceRef

            # Handle Annotated types (like Resource = Annotated[ResourceRef, ...])
            if origin is Annotated:
                args = get_args(expected_type)
                if args and args[0] is ResourceRef:
                    return value
            
            if expected_type is ResourceRef:
                return value

            # Get asset bytes for Packable or array
            asset_bytes = SerializationUtils.get_asset(assets, checksum)

            if isinstance(expected_type, type) and issubclass(expected_type, Packable):
                return expected_type.decode(asset_bytes, array_type)

            # Decode array using encoding from schema + metadata from $ref
            # The $ref object contains: {"$ref": checksum, "dtype": ..., "shape": ..., ...}
            metadata_dict = {k: v for k, v in value.items() if k != "$ref"}
            metadata = ArrayRefMetadata(**metadata_dict)
            return ArrayUtils.decode_with_metadata(asset_bytes, encoding, metadata, array_type)

        # Handle nested dict
        if isinstance(value, dict):
            expected_type = SchemaUtils.unwrap_optional(expected_type)
            origin = get_origin(expected_type)

            if origin is dict:
                _, value_type = get_args(expected_type)
                # Get encoding from the value type annotation (for dict[str, VertexBuffer] etc.)
                value_encoding = ArrayUtils.get_array_encoding(value_type)
                return {
                    k: SchemaUtils.resolve_value_with_type(v, value_type, assets, array_type, value_encoding)
                    for k, v in value.items()
                }

            # Handle untyped dict - recursively resolve values with Any type
            if expected_type is dict:
                return {
                    k: SchemaUtils.resolve_value_with_type(v, Any, assets, array_type, encoding)
                    for k, v in value.items()
                }

            if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                resolved = SchemaUtils.resolve_refs_with_schema(
                    expected_type, value, assets, array_type
                )
                return expected_type(**resolved)

            # Handle nested BaseModel with $module (for Dict[str, Any] cases)
            if "$module" in value:
                import importlib
                module_path, class_name = value["$module"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                if isinstance(model_class, type) and issubclass(model_class, BaseModel):
                    # Remove $module before resolving
                    value_copy = {k: v for k, v in value.items() if k != "$module"}
                    resolved = SchemaUtils.resolve_refs_with_schema(
                        model_class, value_copy, assets, array_type
                    )
                    return model_class(**resolved)

            return value

        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            expected_type = SchemaUtils.unwrap_optional(expected_type)
            origin = get_origin(expected_type)

            if origin in (list, tuple):
                args = get_args(expected_type)
                elem_type = args[0] if args else Any
            else:
                elem_type = Any

            result = [
                SchemaUtils.resolve_value_with_type(v, elem_type, assets, array_type) for v in value
            ]
            return result if isinstance(value, list) else tuple(result)

        return value

    @staticmethod
    def merge_field_data_with_schema(
        model_class: type[BaseModel],
        data: dict[str, Any],
        field_data: dict[str, Any],
    ) -> None:
        """Merge metadata field_data into data using Pydantic schema.

        Args:
            model_class: Pydantic model class
            data: Target data dict (modified in place)
            field_data: Source field data from metadata
        """
        for key, value in field_data.items():
            if key not in model_class.model_fields:
                data[key] = value
                continue

            field_type = model_class.model_fields[key].annotation
            data[key] = SchemaUtils.merge_value_with_schema(value, field_type, data.get(key))

    @staticmethod
    def merge_value_with_schema(
        metadata_value: Any,
        expected_type: Any,
        existing_value: Any,
    ) -> Any:
        """Merge a metadata value with existing data using the schema type.

        Args:
            metadata_value: Value from metadata
            expected_type: Expected type from schema
            existing_value: Existing value in data dict

        Returns:
            Merged value
        """
        if metadata_value is None:
            return existing_value

        expected_type = SchemaUtils.unwrap_optional(expected_type)
        origin = get_origin(expected_type)

        # Handle dict type
        if origin is dict:
            _, value_type = get_args(expected_type)
            if isinstance(metadata_value, dict) and isinstance(existing_value, dict):
                result = dict(existing_value)
                for k, v in metadata_value.items():
                    result[k] = SchemaUtils.merge_value_with_schema(
                        v, value_type, existing_value.get(k)
                    )
                return result
            elif isinstance(metadata_value, dict):
                return {
                    k: SchemaUtils.merge_value_with_schema(v, value_type, None)
                    for k, v in metadata_value.items()
                }
            return metadata_value

        # Handle BaseModel type
        if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
            if isinstance(metadata_value, dict):
                if isinstance(existing_value, dict):
                    merged = dict(existing_value)
                    SchemaUtils.merge_field_data_with_schema(expected_type, merged, metadata_value)
                    return expected_type(**merged)
                else:
                    data = {}
                    SchemaUtils.merge_field_data_with_schema(expected_type, data, metadata_value)
                    return expected_type(**data)
            return metadata_value

        # Handle list type
        if origin in (list, tuple):
            if isinstance(metadata_value, (list, tuple)):
                args = get_args(expected_type)
                elem_type = args[0] if args else Any
                result = [
                    SchemaUtils.merge_value_with_schema(v, elem_type, None) for v in metadata_value
                ]
                return result if origin is list else tuple(result)
            return metadata_value

        return metadata_value

    # ========================================================================
    # Schema-based resolution (using stored schema.json instead of class)
    # ========================================================================

    @staticmethod
    def resolve_refs_with_json_schema(
        schema: JsonSchema,
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: ArrayType | None,
    ) -> dict[str, Any]:
        """Resolve $ref references using validated JSON schema.
        
        This allows deserializing packables without having the original class.
        
        Args:
            schema: Validated JsonSchema instance
            data: Data dict with potential $ref values
            assets: Asset provider
            array_type: Target array type
            
        Returns:
            Resolved data dict with arrays decoded
        """
        result = {}
        
        for field_name, value in data.items():
            if field_name.startswith("$"):
                # Skip metadata fields like $module
                continue
            
            prop = schema.get_resolved_property(field_name)
            encoding = schema.get_encoding(field_name)
            
            result[field_name] = SchemaUtils._resolve_value_with_property(
                value, prop, schema, assets, array_type, encoding
            )
        
        return result

    @staticmethod
    def _resolve_value_with_property(
        value: Any,
        prop: JsonSchemaProperty | None,
        schema: JsonSchema,
        assets: AssetProvider,
        array_type: ArrayType | None,
        encoding: EncodingType = "array",
    ) -> Any:
        """Resolve a value using JsonSchemaProperty.
        
        Args:
            value: Value to resolve
            prop: Schema property for this field (may be None)
            schema: Root schema (for $defs lookups)
            assets: Asset provider
            array_type: Target array type
            encoding: Encoding type from schema
            
        Returns:
            Resolved value
        """
        if value is None:
            return None
        
        if prop is None:
            # No schema info - return as-is or try basic resolution
            if isinstance(value, dict) and "$ref" in value:
                # Try to decode as array with default encoding
                checksum = value["$ref"]
                asset_bytes = SerializationUtils.get_asset(assets, checksum)
                metadata_dict = {k: v for k, v in value.items() if k != "$ref"}
                metadata = ArrayRefMetadata(**metadata_dict)
                return ArrayUtils.decode_with_metadata(asset_bytes, encoding, metadata, array_type)
            return value
        
        # Resolve $ref in schema property
        if prop.ref:
            prop = schema.resolve_ref(prop.ref) or prop
        
        # Handle anyOf (Optional types)
        if prop.anyOf:
            inner = prop.get_inner_type()
            if inner:
                return SchemaUtils._resolve_value_with_property(
                    value, inner, schema, assets, array_type, encoding
                )
            return value
        
        # Handle $ref in data (array/packable/resource reference)
        if isinstance(value, dict) and "$ref" in value:
            checksum = value["$ref"]
            
            # Get metadata (everything except $ref)
            metadata_dict = {k: v for k, v in value.items() if k != "$ref"}
            
            # Check if this is a resource type - has ext field or gzip encoding
            if prop.is_resource_type() or "ext" in metadata_dict or metadata_dict.get("encoding") == "gzip":
                import gzip
                asset_bytes = SerializationUtils.get_asset(assets, checksum)
                ext = metadata_dict.get("ext", "")
                
                # Resource bytes are gzip compressed
                if metadata_dict.get("encoding") == "gzip":
                    resource_bytes = gzip.decompress(asset_bytes)
                else:
                    # Fallback for uncompressed resources (backwards compatibility)
                    resource_bytes = asset_bytes
                
                return {"$ref": checksum, "ext": ext, "_bytes": resource_bytes}
            
            asset_bytes = SerializationUtils.get_asset(assets, checksum)
            
            # Check if this is an array type - must have array metadata (dtype, shape)
            is_array = (
                (prop.is_array_type() or encoding in ARRAY_SCHEMA_TYPES)
                and "dtype" in metadata_dict
                and "shape" in metadata_dict
            )
            
            if is_array:
                actual_encoding = prop.type if prop.is_array_type() else encoding
                metadata = ArrayRefMetadata(**metadata_dict)
                return ArrayUtils.decode_with_metadata(asset_bytes, actual_encoding, metadata, array_type)
            
            # Not an array - could be a nested Packable
            # Recursively decode using schema
            from ..packable import Packable
            return Packable.decode_from_schema(asset_bytes, array_type)
        
        # Handle nested dict (object type)
        if isinstance(value, dict):
            if prop.type == "object":
                # Check for additionalProperties (dict[str, X] pattern)
                if prop.additionalProperties and isinstance(prop.additionalProperties, JsonSchemaProperty):
                    item_prop = prop.additionalProperties
                    # Resolve $ref if present
                    if item_prop.ref:
                        resolved = schema.resolve_ref(item_prop.ref)
                        if resolved:
                            item_prop = resolved
                    item_encoding = item_prop.type if item_prop.is_array_type() else "array"
                    return {
                        k: SchemaUtils._resolve_value_with_property(
                            v, item_prop, schema, assets, array_type, item_encoding
                        )
                        for k, v in value.items()
                    }
                
                # Regular object with properties
                if prop.properties:
                    result = {}
                    for k, v in value.items():
                        if k.startswith("$"):
                            continue
                        nested_prop = prop.properties.get(k)
                        nested_encoding = "array"
                        if nested_prop and nested_prop.is_array_type():
                            nested_encoding = nested_prop.type
                        result[k] = SchemaUtils._resolve_value_with_property(
                            v, nested_prop, schema, assets, array_type, nested_encoding
                        )
                    return result
            
            # Unknown dict - return as-is
            return value
        
        # Handle lists
        if isinstance(value, list):
            item_prop = prop.items
            item_encoding = "array"
            if item_prop and item_prop.is_array_type():
                item_encoding = item_prop.type
            return [
                SchemaUtils._resolve_value_with_property(
                    v, item_prop, schema, assets, array_type, item_encoding
                )
                for v in value
            ]
        
        # Primitive value
        return value