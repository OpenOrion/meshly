"""Schema utilities for resolving $ref values during deserialization."""

from __future__ import annotations

import gzip
import importlib
import types
import typing
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel

from meshly.array import ArrayMetadata, ArrayType, ArrayUtils, EncodedArray
from meshly.common import AssetProvider
from meshly.resource import ResourceRef
from meshly.utils.json_schema import JsonSchema, JsonSchemaProperty
from meshly.utils.serialization_utils import SerializationUtils

# JSON-compatible value type
JsonValue = Union[str, int, float, bool, None, dict, list]


class SchemaUtils:
    """Utilities for resolving $ref values during deserialization."""

    # -------------------------------------------------------------------------
    # Type helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _unwrap_optional(t: type) -> type:
        """Unwrap Optional[X] or X | None to get X."""
        origin = get_origin(t)
        if origin is Union or isinstance(t, types.UnionType):
            args = get_args(t)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0]
        return t

    @staticmethod
    def _is_resource_ref(t: type) -> bool:
        """Check if type is ResourceRef or Annotated[ResourceRef, ...]."""
        if t is ResourceRef:
            return True
        if get_origin(t) is Annotated:
            args = get_args(t)
            return bool(args and args[0] is ResourceRef)
        return False

    @staticmethod
    def _load_class(module_path: str) -> Union[type, None]:
        """Load a class from 'package.module.ClassName'."""
        path, name = module_path.rsplit(".", 1)
        return getattr(importlib.import_module(path), name, None)

    # -------------------------------------------------------------------------
    # Public: Resolution entry points
    # -------------------------------------------------------------------------

    @staticmethod
    def resolve_from_class(
        model_class: type[BaseModel],
        data: dict[str, JsonValue],
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
    ) -> dict[str, object]:
        """Resolve $ref values using Pydantic model type hints."""
        hints = typing.get_type_hints(model_class, include_extras=True)
        result: dict[str, object] = {}

        for field_name, field_info in model_class.model_fields.items():
            if field_name not in data:
                continue
            field_type = hints.get(field_name, field_info.annotation)
            result[field_name] = SchemaUtils._resolve_with_type(
                data[field_name], field_type, assets, array_type
            )

        return result

    @staticmethod
    def resolve_from_schema(
        schema: JsonSchema,
        data: dict[str, JsonValue],
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
    ) -> dict[str, object]:
        """Resolve $ref values using JSON schema."""
        result: dict[str, object] = {}
        for field_name, value in data.items():
            if field_name.startswith("$"):
                continue
            prop = schema.get_resolved_property(field_name)
            result[field_name] = SchemaUtils._resolve_with_prop(value, prop, schema, assets, array_type)
        return result

    # -------------------------------------------------------------------------
    # Private: Type-based resolution (Pydantic models)
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_with_type(
        value: JsonValue,
        expected_type: type,
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
    ) -> object:
        """Resolve a value using Pydantic type information."""
        from meshly.packable import Packable

        if value is None:
            return None

        expected_type = SchemaUtils._unwrap_optional(expected_type)

        # $ref in data - decode array, packable, or return as-is for resources
        if isinstance(value, dict) and "$ref" in value:
            if SchemaUtils._is_resource_ref(expected_type):
                return value
            if isinstance(expected_type, type) and issubclass(expected_type, Packable):
                return expected_type.decode(
                    SerializationUtils.get_asset(assets, value["$ref"]), array_type
                )
            # Array - get encoding from type annotation
            return ArrayUtils.decode_array(
                EncodedArray(
                    data=SerializationUtils.get_asset(assets, value["$ref"]),
                    metadata=ArrayMetadata(**{k: v for k, v in value.items() if k != "$ref"}),
                ),
                ArrayUtils.get_array_encoding(expected_type),
                array_type,
            )

        # Nested dict
        if isinstance(value, dict):
            origin = get_origin(expected_type)
            # dict[str, X]
            if origin is dict:
                _, val_type = get_args(expected_type)
                return {k: SchemaUtils._resolve_with_type(v, val_type, assets, array_type) for k, v in value.items()}
            # BaseModel
            if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                resolved = SchemaUtils.resolve_from_class(expected_type, value, assets, array_type)
                return expected_type(**resolved)
            # $module: dynamic class
            if "$module" in value:
                cls = SchemaUtils._load_class(value["$module"])
                if cls and isinstance(cls, type) and issubclass(cls, BaseModel):
                    resolved = SchemaUtils.resolve_from_class(
                        cls, {k: v for k, v in value.items() if k != "$module"}, assets, array_type
                    )
                    return cls(**resolved)
            # Untyped dict
            return {k: SchemaUtils._resolve_with_type(v, object, assets, array_type) for k, v in value.items()}

        # List/tuple
        if isinstance(value, (list, tuple)):
            origin = get_origin(expected_type)
            elem_type = get_args(expected_type)[0] if origin in (list, tuple) and get_args(expected_type) else object
            result = [SchemaUtils._resolve_with_type(v, elem_type, assets, array_type) for v in value]
            return result if isinstance(value, list) else tuple(result)

        return value

    # -------------------------------------------------------------------------
    # Private: Schema-based resolution (JSON schema)
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_with_prop(
        value: JsonValue,
        prop: Union[JsonSchemaProperty, None],
        schema: JsonSchema,
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
    ) -> object:
        """Resolve a value using JSON schema property."""
        from meshly.packable import Packable

        if value is None:
            return None

        # Resolve schema $ref and unwrap Optional
        if prop and prop.ref:
            prop = schema.resolve_ref(prop.ref) or prop
        if prop and prop.anyOf:
            prop = prop.get_inner_type() or prop

        # $ref in data
        if isinstance(value, dict) and "$ref" in value:
            checksum = value["$ref"]
            metadata = {k: v for k, v in value.items() if k != "$ref"}
            asset_bytes = SerializationUtils.get_asset(assets, checksum)

            # Resource
            is_resource = (prop and prop.is_resource_type()) or "ext" in metadata or metadata.get("encoding") == "gzip"
            if is_resource:
                data = gzip.decompress(asset_bytes) if metadata.get("encoding") == "gzip" else asset_bytes
                return {"$ref": checksum, "ext": metadata.get("ext", ""), "_bytes": data}

            # Array
            if "dtype" in metadata and "shape" in metadata:
                encoding = prop.type if prop and prop.is_array_type() else "array"
                return ArrayUtils.decode_array(
                    EncodedArray(
                        data=asset_bytes, 
                        metadata=ArrayMetadata(**{k: v for k, v in value.items() if k != "$ref"})
                    ), 
                    encoding,
                    array_type,
                )

            # Packable
            return Packable.decode(asset_bytes, array_type)

        # No schema info - fallback
        if prop is None:
            return value

        # Nested dict
        if isinstance(value, dict) and prop.type == "object":
            # dict[str, X] via additionalProperties
            if prop.additionalProperties and isinstance(prop.additionalProperties, JsonSchemaProperty):
                item_prop = prop.additionalProperties
                if item_prop.ref:
                    item_prop = schema.resolve_ref(item_prop.ref) or item_prop
                return {k: SchemaUtils._resolve_with_prop(v, item_prop, schema, assets, array_type) for k, v in value.items()}
            # Named properties
            if prop.properties:
                return {
                    k: SchemaUtils._resolve_with_prop(v, prop.properties.get(k), schema, assets, array_type)
                    for k, v in value.items() if not k.startswith("$")
                }
            return value

        # List
        if isinstance(value, list):
            return [SchemaUtils._resolve_with_prop(v, prop.items, schema, assets, array_type) for v in value]

        return value
