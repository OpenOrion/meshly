"""Schema utilities for resolving $ref values during deserialization."""

from __future__ import annotations

import gzip
import importlib
import types
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Union, get_args, get_origin

import numpy as np

from pydantic import BaseModel

from meshly.array import ArrayRefInfo, ArrayType, ArrayUtils, ExtractedArray
from meshly.common import AssetProvider
from meshly.resource import Resource
from meshly.utils.fork_pool import ForkPool
from meshly.utils.json_schema import JsonSchema, JsonSchemaProperty
from meshly.utils.serialization_utils import SerializationUtils

# JSON-compatible value type
JsonValue = Union[str, int, float, bool, None, dict, list]


# -----------------------------------------------------------------------------
# Fork-based parallel Packable resolution
# -----------------------------------------------------------------------------
# Uses module-level state because fork() copies memory (COW) without
# serialization, allowing non-picklable objects like assets to be shared.

@dataclass
class _PackableResolveContext:
    """Typed context for parallel Packable resolution."""
    values: list
    expected_type: type
    assets: AssetProvider
    array_type: ArrayType


_PACKABLE_CTX: _PackableResolveContext | None = None


@contextmanager
def _packable_context(values: list, expected_type: type, assets: AssetProvider, array_type: ArrayType):
    """Context manager for fork-based parallel Packable resolution."""
    global _PACKABLE_CTX
    _PACKABLE_CTX = _PackableResolveContext(values, expected_type, assets, array_type)
    try:
        yield
    finally:
        _PACKABLE_CTX = None


def _resolve_packable_item(idx: int) -> object:
    """Worker: reconstruct a single Packable from $ref."""
    from meshly.packable import Packable
    
    ctx = _PACKABLE_CTX
    if ctx is None:
        raise RuntimeError("_packable_context not set")
    
    checksum = ctx.values[idx]["$ref"]
    asset_bytes = SerializationUtils.get_asset(ctx.assets, checksum)
    
    if isinstance(ctx.expected_type, type) and issubclass(ctx.expected_type, Packable):
        return ctx.expected_type.decode(asset_bytes, ctx.array_type)
    return Packable.decode(asset_bytes, ctx.array_type)


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
        if t is Resource:
            return True
        if get_origin(t) is Annotated:
            args = get_args(t)
            return bool(args and args[0] is Resource)
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

        # $ref in data - decode array, packable, or resource
        if isinstance(value, dict) and "$ref" in value:
            # ResourceRef - decompress and create
            if SchemaUtils._is_resource_ref(expected_type):
                asset_bytes = SerializationUtils.get_asset(assets, value["$ref"])
                data = gzip.decompress(asset_bytes)
                return Resource(data=data, ext=value.get("ext", ""), name=value.get("name", ""))
            if isinstance(expected_type, type) and issubclass(expected_type, Packable):
                return expected_type.decode(
                    SerializationUtils.get_asset(assets, value["$ref"]), array_type
                )
            # Array - get encoding from type annotation
            encoding = ArrayUtils.get_array_encoding(expected_type)
            return ArrayUtils.reconstruct(
                ExtractedArray(
                    data=SerializationUtils.get_asset(assets, value["$ref"]),
                    info=ArrayRefInfo(**{k: v for k, v in value.items() if k != "$ref"}),
                    encoding=encoding,
                ),
                array_type,
            )

        # Nested dict
        if isinstance(value, dict):
            origin = get_origin(expected_type)
            # dict[str, X]
            if origin is dict:
                _, val_type = get_args(expected_type)
                keys = list(value.keys())
                resolved = SchemaUtils._resolve_list_items(list(value.values()), val_type, assets, array_type)
                return dict(zip(keys, resolved))
            # BaseModel
            if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                resolved = SchemaUtils.resolve_from_class(expected_type, value, assets, array_type)
                return expected_type(**resolved)
            # Union types - match via Literal discriminator fields
            if origin is Union or isinstance(expected_type, types.UnionType):
                union_args = [a for a in get_args(expected_type) if a is not type(None)]
                # Auto-discover: find any Literal field whose value matches a key in the data
                for arg_type in union_args:
                    if not (isinstance(arg_type, type) and issubclass(arg_type, BaseModel)):
                        continue
                    for fname, finfo in arg_type.model_fields.items():
                        if fname not in value or get_origin(finfo.annotation) is not Literal:
                            continue
                        if value[fname] in get_args(finfo.annotation):
                            resolved = SchemaUtils.resolve_from_class(arg_type, value, assets, array_type)
                            return arg_type(**resolved)
                # Fallback: try each BaseModel type
                for arg_type in union_args:
                    if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
                        try:
                            resolved = SchemaUtils.resolve_from_class(arg_type, value, assets, array_type)
                            return arg_type(**resolved)
                        except Exception:
                            continue
            # Untyped dict
            return {k: SchemaUtils._resolve_with_type(v, object, assets, array_type) for k, v in value.items()}

        # List annotation → reconstruct numpy array from inline JSON list
        if isinstance(value, list) and ArrayUtils.is_list_annotation(expected_type):
            return ArrayUtils.convert_array(np.array(value), array_type)

        # List/tuple
        if isinstance(value, (list, tuple)):
            origin = get_origin(expected_type)
            elem_type = get_args(expected_type)[0] if origin in (list, tuple) and get_args(expected_type) else object
            result = SchemaUtils._resolve_list_items(value, elem_type, assets, array_type)
            return result if isinstance(value, list) else tuple(result)

        return value

    @staticmethod
    def _resolve_list_items(
        items: list,
        elem_type: type,
        assets: AssetProvider,
        array_type: ArrayType,
    ) -> list:
        """Resolve list items, parallelizing when items are Packable $refs.
        
        Uses fork-based parallelism for lists of Packable references.
        Falls back to sequential for mixed types.
        """
        from meshly.packable import Packable
        
        if not items:
            return []
        
        # Check if all items are Packable $refs and type is Packable
        MIN_ITEMS_FOR_PARALLEL = 50
        is_packable_type = isinstance(elem_type, type) and issubclass(elem_type, Packable)
        all_packable_refs = (
            len(items) >= MIN_ITEMS_FOR_PARALLEL
            and is_packable_type
            and all(isinstance(v, dict) and "$ref" in v for v in items)
        )
        
        if all_packable_refs:
            with _packable_context(items, elem_type, assets, array_type):
                return ForkPool.map(
                    _resolve_packable_item,
                    range(len(items)),
                    min_items_for_parallel=MIN_ITEMS_FOR_PARALLEL,
                )
        
        # Sequential fallback
        return [SchemaUtils._resolve_with_type(v, elem_type, assets, array_type) for v in items]

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

            # Use schema to determine type
            if prop and prop.is_resource_type():
                # Resource - assets from _extract_resource are always gzip compressed
                data = gzip.decompress(asset_bytes)
                return Resource(data=data, ext=metadata.get("ext", ""), name=metadata.get("name", ""))
            
            if prop and prop.is_array_type():
                # Array
                encoding = prop.type
                return ArrayUtils.reconstruct(
                    ExtractedArray(
                        data=asset_bytes, 
                        info=ArrayRefInfo(**{k: v for k, v in value.items() if k != "$ref"}),
                        encoding=encoding,
                    ), 
                    array_type,
                )
            
            # Packable (default)
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
            # Named properties - reconstruct as a nested model
            if prop.properties:
                from meshly.utils.dynamic_model import DynamicModelBuilder
                from meshly.utils.json_schema import JsonSchema as _JsonSchema
                nested_schema = _JsonSchema(
                    title=prop.title or "NestedModel",
                    properties=prop.properties,
                    required=prop.required or [],
                    x_base=prop.x_base,
                    x_module=prop.x_module,
                    **{"$defs": dict(schema.defs)},
                )
                return DynamicModelBuilder.instantiate(nested_schema, value, assets, array_type)
            return value

        # List annotation → reconstruct numpy array from inline JSON list
        if isinstance(value, list) and prop and prop.type == "list":
            return ArrayUtils.convert_array(np.array(value), array_type)

        # List
        if isinstance(value, list):
            return [SchemaUtils._resolve_with_prop(v, prop.items, schema, assets, array_type) for v in value]

        return value
