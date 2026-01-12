"""Utilities for extracting arrays and reconstructing BaseModel instances."""

from typing import Any, Dict, Set, Union
import numpy as np
from pydantic import BaseModel

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False

if HAS_JAX:
    Array = Union[np.ndarray, jnp.ndarray]
else:
    Array = np.ndarray


class PackableUtils:
    """Static utilities for array extraction and BaseModel reconstruction."""

    @staticmethod
    def is_array(obj: Any) -> bool:
        """Check if obj is a numpy or JAX array."""
        return isinstance(obj, np.ndarray) or (HAS_JAX and isinstance(obj, jnp.ndarray))

    @staticmethod
    def extract_nested_arrays(obj: Any, prefix: str = "") -> Dict[str, Array]:
        """Recursively extract arrays from nested dicts and BaseModel instances."""
        arrays = {}
        if PackableUtils.is_array(obj):
            arrays[prefix] = obj
        elif isinstance(obj, BaseModel):
            for name in type(obj).model_fields:
                value = getattr(obj, name, None)
                if value is not None:
                    key = f"{prefix}.{name}" if prefix else name
                    arrays.update(
                        PackableUtils.extract_nested_arrays(value, key))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                arrays.update(PackableUtils.extract_nested_arrays(v, key))
        return arrays

    @staticmethod
    def extract_non_arrays(obj: Any) -> Any:
        """Extract non-array values, preserving BaseModel type info for reconstruction."""
        if PackableUtils.is_array(obj):
            return None
        if isinstance(obj, BaseModel):
            result = {"__model_class__": obj.__class__.__name__,
                      "__model_module__": obj.__class__.__module__}
            for name in type(obj).model_fields:
                val = getattr(obj, name, None)
                if not PackableUtils.is_array(val):
                    extracted = PackableUtils.extract_non_arrays(val)
                    if extracted is not None:
                        result[name] = extracted
            return result if len(result) > 2 else None
        if isinstance(obj, dict):
            result = {k: PackableUtils.extract_non_arrays(v) for k, v in obj.items()
                      if not PackableUtils.is_array(v)}
            result = {k: v for k, v in result.items() if v is not None}
            return result or None
        return obj

    @staticmethod
    def reconstruct_model(data: Dict[str, Any]) -> Any:
        """Reconstruct BaseModel from serialized dict with __model_class__/__model_module__."""
        if not isinstance(data, dict):
            return data

        # Recursively process nested dicts first
        processed = {k: PackableUtils.reconstruct_model(v) if isinstance(v, dict) else v
                     for k, v in data.items() if k not in ("__model_class__", "__model_module__")}

        if "__model_class__" not in data:
            return processed

        try:
            import importlib
            module = importlib.import_module(data["__model_module__"])
            model_class = getattr(module, data["__model_class__"])
            return model_class(**processed)
        except (ImportError, AttributeError):
            return processed

    @staticmethod
    def merge_field_data(data: Dict[str, Any], field_data: Dict[str, Any]) -> None:
        """Merge metadata fields into data, reconstructing BaseModel instances."""
        for key, value in field_data.items():
            existing = data.get(key)
            if not isinstance(value, dict):
                data[key] = value
            elif "__model_class__" in value:
                # Single BaseModel: merge arrays then reconstruct
                merged = {**value, **
                          (existing if isinstance(existing, dict) else {})}
                data[key] = PackableUtils.reconstruct_model(merged)
            elif isinstance(existing, dict):
                # Check if dict of BaseModels
                for subkey, subval in value.items():
                    if isinstance(subval, dict) and "__model_class__" in subval:
                        merged = {**subval, **existing.get(subkey, {})}
                        existing[subkey] = PackableUtils.reconstruct_model(
                            merged)
                    elif isinstance(subval, dict) and isinstance(existing.get(subkey), dict):
                        PackableUtils.merge_field_data(
                            existing[subkey], subval)
                    else:
                        existing[subkey] = subval
            else:
                data[key] = PackableUtils.reconstruct_model(value)
