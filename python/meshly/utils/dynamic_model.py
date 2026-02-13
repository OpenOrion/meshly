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
from meshly.resource import Resource
from meshly.utils.json_schema import JsonSchema, JsonSchemaProperty
from meshly.utils.schema_utils import SchemaUtils
from meshly.lazy_model import LazyModel


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
        
        # Determine base class from x-base hint
        base_class = BaseModel
        x_base = schema.x_base
        if x_base == "Mesh":
            from meshly.mesh import Mesh
            base_class = Mesh
        elif x_base == "Packable":
            from meshly.packable import Packable
            base_class = Packable
        
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
        
        # Create model with base class
        # Note: When using __base__, config is inherited from base class
        # Only set __config__ explicitly when base_class is BaseModel
        if base_class is BaseModel:
            model = create_model(
                title,
                __module__="meshly.dynamic",
                __config__=ConfigDict(arbitrary_types_allowed=True),
                **field_definitions,
            )
        else:
            model = create_model(
                title,
                __module__="meshly.dynamic",
                __base__=base_class,
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
    ) -> Union[BaseModel, "LazyModel"]:
        """
        Build a model from schema and instantiate it with resolved data.
        
        Args:
            schema: Validated JsonSchema instance
            data: Data dict with $ref references
            assets: Asset provider (dict or callable)
            array_type: Target array type ("numpy" or "jax")
            is_lazy: If True, return a lazy proxy that resolves fields on access
            
        Returns:
            Instance of dynamically created Pydantic model, or LazyModel if lazy
        """
        
        if is_lazy:
            return LazyModel(schema, data, assets, array_type)
        
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
    ) -> tuple[object, Any]:
        """
        Convert a JsonSchemaProperty to Python type and default.
        
        Returns:
            Tuple of (python_type, default_value)
            Note: python_type can be a type, Union type, or None
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

        # Handle resource types
        if prop.is_resource_type():
            if is_required:
                return (Resource, ...)
            return (Union[Resource, None], None)

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
            # Check for x-base hints (Mesh, Packable)
            if prop.x_base == "Mesh":
                from meshly.mesh import Mesh
                return (Mesh, ...) if is_required else (Union[Mesh, None], None)
            if prop.x_base == "Packable":
                from meshly.packable import Packable
                return (Packable, ...) if is_required else (Union[Packable, None], None)
            
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
        x_base = schema.x_base or "basemodel"
        return f"{x_base}:{title}:{','.join(props)}"
