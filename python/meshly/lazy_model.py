
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from meshly.array import ArrayType
from meshly.common import AssetProvider
from meshly.utils.json_schema import JsonSchema
from meshly.utils.schema_utils import SchemaUtils


class LazyModel:
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
        raise AttributeError("LazyModel is read-only. Use resolve() to get a mutable model.")

    def resolve(self, model_class: type[BaseModel]) -> BaseModel:
        """Fully resolve all fields and return the actual Pydantic model."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved

        schema = object.__getattribute__(self, "_schema")
        data = object.__getattribute__(self, "_data")
        assets = object.__getattribute__(self, "_assets")
        array_type = object.__getattribute__(self, "_array_type")
        cache = object.__getattribute__(self, "_cache")


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
        return f"LazyModel[{title}](loaded={loaded}, pending={pending})"