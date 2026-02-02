import asyncio
import inspect
from typing import Any, Generic, TypeVar, Union
from pydantic import BaseModel
from meshly.common import AssetProvider
from meshly.array import ArrayType
from meshly.utils.schema_utils import SchemaUtils


TModel = TypeVar("TModel", bound=BaseModel)


class LazyModel(Generic[TModel]):
    """Lazy proxy for a Pydantic BaseModel that defers asset loading until field access.

    Example:
        def fetch_asset(checksum: str) -> bytes:
            return cloud_storage.download(checksum)

        lazy_model = Packable.reconstruct(SimulationCase, data, fetch_asset, is_lazy=True)
        # No assets loaded yet

        temp = lazy_model.temperature  # NOW the temperature asset is fetched
        vel = lazy_model.velocity      # NOW the velocity asset is fetched
    """

    __slots__ = ("_model_class", "_data", "_assets", "_array_type", "_cache", "_resolved")

    def __init__(
        self,
        model_class: type[TModel],
        data: dict[str, Any],
        assets: AssetProvider,
        array_type: Union[ArrayType, None] = None,
    ):
        object.__setattr__(self, "_model_class", model_class)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_assets", assets)
        object.__setattr__(self, "_array_type", array_type)
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_resolved", None)

    def _get_cached_asset(self, checksum: str) -> bytes:
        """Get asset bytes."""
        
        assets = object.__getattribute__(self, "_assets")

        if callable(assets):
            result = assets(checksum)
            if inspect.isawaitable(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            if result is None:
                raise KeyError(f"Asset fetcher returned None for checksum '{checksum}'")
            return result

        if checksum not in assets:
            raise KeyError(f"Missing asset with checksum '{checksum}'")
        return assets[checksum]

    def __getattr__(self, name: str) -> Any:
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]

        model_class = object.__getattribute__(self, "_model_class")
        data = object.__getattribute__(self, "_data")
        array_type = object.__getattribute__(self, "_array_type")

        if name not in model_class.model_fields:
            raise AttributeError(f"'{model_class.__name__}' has no attribute '{name}'")

        if name not in data:
            return None

        field_value = data[name]
        field_type = model_class.model_fields[name].annotation

        resolved = SchemaUtils._resolve_with_type(
            field_value, field_type, self._get_cached_asset, array_type
        )

        cache[name] = resolved
        return resolved

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("LazyModel is read-only. Use resolve() to get a mutable model.")

    def resolve(self) -> TModel:
        """Fully resolve all fields and return the actual Pydantic model."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved

        model_class = object.__getattribute__(self, "_model_class")
        data = object.__getattribute__(self, "_data")
        array_type = object.__getattribute__(self, "_array_type")
        cache = object.__getattribute__(self, "_cache")

        resolved_data = {}
        for field_name, field_info in model_class.model_fields.items():
            if field_name in cache:
                resolved_data[field_name] = cache[field_name]
            elif field_name in data:
                resolved_data[field_name] = SchemaUtils._resolve_with_type(
                    data[field_name], field_info.annotation, self._get_cached_asset, array_type
                )

        result = model_class(**resolved_data)
        object.__setattr__(self, "_resolved", result)
        return result

    def __repr__(self) -> str:
        model_class = object.__getattribute__(self, "_model_class")
        cache = object.__getattribute__(self, "_cache")
        data = object.__getattribute__(self, "_data")
        loaded = list(cache.keys())
        pending = [k for k in data.keys() if k not in cache]
        return f"LazyModel[{model_class.__name__}](loaded={loaded}, pending={pending})"

