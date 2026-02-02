"""
Dynamic Pydantic model building from JSON schema.

Provides utilities for:
- Building Pydantic models from validated JSON schemas
- Instantiating models with resolved data (eager or lazy)
- Caching dynamically created models
"""

from __future__ import annotations

from typing import Any, Type, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

from meshly.array import ArrayType
from meshly.common import AssetProvider
from meshly.utils.json_schema import JsonSchema, JsonSchemaProperty
from meshly.utils.schema_utils import SchemaUtils


class DynamicModelBuilder:
    """
    Builds Pydantic models dynamically from validated JSON schemas.
    
    Handles meshly's custom array types (array, vertex_buffer, index_sequence)
    and caches models for reuse.
    
    Usage:
        # Build a model from validated schema
        schema = JsonSchema.model_validate(schema_dict)
        ModelClass = DynamicModelBuilder.build_model(schema)
        
        # Instantiate with resolved data (eager)
        instance = DynamicModelBuilder.instantiate(schema, data, assets)
        
        # Instantiate with lazy loading (assets fetched on field access)
        lazy = DynamicModelBuilder.instantiate(schema, data, assets, is_lazy=True)
        value = lazy.some_field  # Asset fetched NOW
    """
    
    _model_cache: dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def build_model(
        cls,
        schema: JsonSchema,
        model_name: Union[str, None] = None,
    ) -> Type[BaseModel]:
        """
        Build a Pydantic model from a validated JSON schema.
        
        Args:
            schema: Validated JsonSchema instance
            model_name: Optional name for the model (defaults to schema title)
            
        Returns:
            Dynamically created Pydantic model class
        """
        title = model_name or schema.title or "DynamicModel"
        
        # Check cache
        cache_key = cls._schema_cache_key(schema)
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]
        
        # Build field definitions
        field_definitions = {}
        required = set(schema.required)
        
        for field_name in schema.field_names():
            prop = schema.get_property(field_name)
            if not prop:
                continue
                
            py_type, default = cls._property_to_type(prop, schema, required, field_name)
            
            # Get description for Field
            if prop.description:
                field_definitions[field_name] = (py_type, Field(default=default, description=prop.description))
            else:
                field_definitions[field_name] = (py_type, default)
        
        # Create model with config for arbitrary types (numpy arrays)
        model = create_model(
            title,
            __module__="meshly.dynamic",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **field_definitions,
        )
        
        cls._model_cache[cache_key] = model
        return model
    
    @classmethod
    def instantiate(
        cls,
        schema: JsonSchema,
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
        is_lazy: bool = False,
    ) -> Union[BaseModel, "LazyDynamicModel"]:
        """
        Build a model from schema and instantiate it with resolved data.
        
        Args:
            schema: Validated JsonSchema instance
            data: Data dict with $ref references
            assets: Asset provider (dict or callable)
            array_type: Target array type ("numpy" or "jax")
            is_lazy: If True, return a lazy proxy that resolves fields on access
            
        Returns:
            Instance of dynamically created Pydantic model, or LazyDynamicModel if lazy
        """
        
        if is_lazy:
            return LazyDynamicModel(schema, data, assets, array_type)
        
        ModelClass = cls.build_model(schema)
        resolved_data = SchemaUtils.resolve_from_schema(schema, data, assets, array_type)
        return ModelClass(**resolved_data)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the model cache."""
        cls._model_cache.clear()
    
    # =========================================================================
    # Type Conversion
    # =========================================================================
    
    @classmethod
    def _property_to_type(
        cls,
        prop: JsonSchemaProperty,
        schema: JsonSchema,
        required: set[str],
        field_name: str,
    ) -> tuple[type, Any]:
        """
        Convert a JsonSchemaProperty to Python type and default.
        
        Returns:
            Tuple of (python_type, default_value)
        """
        is_required = field_name in required
        
        # Resolve $ref
        if prop.ref:
            resolved = schema.resolve_ref(prop.ref)
            if resolved:
                prop = resolved
        
        # Handle anyOf (Optional types)
        if prop.anyOf:
            inner = prop.get_inner_type()
            if inner:
                inner_type, _ = cls._property_to_type(inner, schema, required, field_name)
                return (Union[inner_type, None], None)
            return (Any, None)
        
        # Handle array types (our custom types)
        if prop.is_array_type():
            if is_required:
                return (np.ndarray, ...)
            return (Union[np.ndarray, None], None)
        
        # Handle standard JSON schema types
        if prop.type == "string":
            return (str, ...) if is_required else (Union[str, None], None)
        
        if prop.type == "integer":
            return (int, ...) if is_required else (Union[int, None], None)
        
        if prop.type == "number":
            return (float, ...) if is_required else (Union[float, None], None)
        
        if prop.type == "boolean":
            return (bool, ...) if is_required else (Union[bool, None], None)
        
        if prop.type == "null":
            return (None, None)
        
        if prop.type == "array":
            if prop.items:
                item_type, _ = cls._property_to_type(prop.items, schema, set(), "")
                list_type = list[item_type]
            else:
                list_type = list
            default = prop.default if prop.default is not None else (... if is_required else None)
            return (list_type, default) if is_required else (Union[list_type, None], default)
        
        if prop.type == "object":
            # Check for additionalProperties (dict pattern)
            if prop.additionalProperties and isinstance(prop.additionalProperties, JsonSchemaProperty):
                value_type, _ = cls._property_to_type(prop.additionalProperties, schema, set(), "")
                dict_type = dict[str, value_type]
                default = prop.default if prop.default is not None else (... if is_required else {})
                if default is ...:
                    return (dict_type, ...)
                return (dict_type, default)
            
            # Nested object with properties - create nested model
            if prop.properties:
                nested_name = prop.title or f"{field_name.title()}Model"
                # Create a nested JsonSchema for the property
                nested_schema = JsonSchema(
                    title=nested_name,
                    properties=prop.properties,
                    required=prop.required or [],
                )
                nested_model = cls.build_model(nested_schema, nested_name)
                return (nested_model, ...) if is_required else (Union[nested_model, None], None)
            
            # Generic dict
            return (dict, ...) if is_required else (Union[dict, None], None)
        
        # Fallback
        return (Any, ...) if is_required else (Union[Any, None], None)
    
    @staticmethod
    def _schema_cache_key(schema: JsonSchema) -> str:
        """Generate a cache key for a schema."""
        title = schema.title or ""
        props = sorted(schema.field_names())
        return f"{title}:{','.join(props)}"


# =============================================================================
# LazyDynamicModel - Lazy proxy for dynamically built models
# =============================================================================


class LazyDynamicModel:
    """Lazy proxy for a Pydantic BaseModel that defers asset loading until field access.

    This class provides on-demand resolution of fields using JSON schema information.
    Assets are only fetched when their corresponding fields are accessed.
    The actual model class is only built when resolve() is called.

    Example:
        def fetch_asset(checksum: str) -> bytes:
            return cloud_storage.download(checksum)

        schema = JsonSchema.model_validate(schema_dict)
        lazy = DynamicModelBuilder.instantiate(schema, data, fetch_asset, is_lazy=True)
        # No assets loaded yet, no model built yet

        temp = lazy.temperature  # NOW the temperature asset is fetched
        vel = lazy.velocity      # NOW the velocity asset is fetched
        
        # Get the fully resolved model (model class built here)
        model = lazy.resolve()
    """

    __slots__ = ("_schema", "_data", "_assets", "_array_type", "_cache", "_resolved")

    def __init__(
        self,
        schema: JsonSchema,
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: ArrayType = "numpy",
    ):
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_assets", assets)
        object.__setattr__(self, "_array_type", array_type)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_resolved", None)

    def __getattr__(self, name: str) -> Any:
        # Check cache first
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]

        schema = object.__getattribute__(self, "_schema")
        data = object.__getattribute__(self, "_data")
        assets = object.__getattribute__(self, "_assets")
        array_type = object.__getattribute__(self, "_array_type")

        # Validate field exists in schema
        if name not in schema.field_names():
            raise AttributeError(f"'{schema.title or 'DynamicModel'}' has no attribute '{name}'")

        if name not in data:
            return None

        # Get property schema and resolve
        field_value = data[name]
        prop = schema.get_resolved_property(name)
        
        resolved = SchemaUtils._resolve_with_prop(
            field_value, prop, schema, assets, array_type
        )

        cache[name] = resolved
        return resolved

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LazyDynamicModel is read-only. Use resolve() to get a mutable model.")

    def resolve(self) -> BaseModel:
        """Fully resolve all fields and return the actual Pydantic model."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved

        schema = object.__getattribute__(self, "_schema")
        data = object.__getattribute__(self, "_data")
        assets = object.__getattribute__(self, "_assets")
        array_type = object.__getattribute__(self, "_array_type")
        cache = object.__getattribute__(self, "_cache")

        # Build model class only when resolving
        model_class = DynamicModelBuilder.build_model(schema)

        resolved_data = {}
        for field_name in schema.field_names():
            if field_name in cache:
                resolved_data[field_name] = cache[field_name]
            elif field_name in data:
                prop = schema.get_resolved_property(field_name)
                resolved_data[field_name] = SchemaUtils._resolve_with_prop(
                    data[field_name], prop, schema, assets, array_type
                )

        result = model_class(**resolved_data)
        object.__setattr__(self, "_resolved", result)
        return result

    def __repr__(self) -> str:
        schema = object.__getattribute__(self, "_schema")
        cache = object.__getattribute__(self, "_cache")
        data = object.__getattribute__(self, "_data")
        title = schema.title or "DynamicModel"
        loaded = list(cache.keys())
        pending = [k for k in data.keys() if k not in cache and not k.startswith("$")]
        return f"LazyDynamicModel[{title}](loaded={loaded}, pending={pending})"