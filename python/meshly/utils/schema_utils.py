"""Schema utilities for resolving Pydantic types and merging field data."""

from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from ..array import ArrayType
from ..data_handler import AssetProvider
from .serialization_utils import SerializationUtils


class SchemaUtils:
    """Utility class for Pydantic schema operations."""

    @staticmethod
    def unwrap_optional(expected_type: Any) -> Any:
        """Unwrap Optional[X] to X.

        Args:
            expected_type: Type annotation, possibly Optional

        Returns:
            Inner type if Optional, otherwise unchanged
        """
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return expected_type

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

            result[field_name] = SchemaUtils.resolve_value_with_type(
                data[field_name], field_info.annotation, assets, array_type
            )

        return result

    @staticmethod
    def resolve_value_with_type(
        value: Any,
        expected_type: Any,
        assets: AssetProvider,
        array_type: ArrayType | None,
    ) -> Any:
        """Resolve a value using the expected type from Pydantic schema.

        Args:
            value: Value to resolve
            expected_type: Expected type from schema
            assets: Asset provider
            array_type: Target array type

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
            # The ResourceRef validator will handle it
            from ..resource import ResourceRef

            if expected_type is ResourceRef:
                return value

            # Get asset bytes for Packable or array
            asset_bytes = SerializationUtils.get_asset(assets, checksum)

            if isinstance(expected_type, type) and issubclass(expected_type, Packable):
                return expected_type.decode(asset_bytes, array_type)

            return SerializationUtils.unpack_array(asset_bytes, array_type)

        # Handle nested dict
        if isinstance(value, dict):
            expected_type = SchemaUtils.unwrap_optional(expected_type)
            origin = get_origin(expected_type)

            if origin is dict:
                _, value_type = get_args(expected_type)
                return {
                    k: SchemaUtils.resolve_value_with_type(v, value_type, assets, array_type)
                    for k, v in value.items()
                }

            if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                resolved = SchemaUtils.resolve_refs_with_schema(
                    expected_type, value, assets, array_type
                )
                return expected_type(**resolved)

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
