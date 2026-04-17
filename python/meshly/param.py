"""Unit-aware parameter field for Pydantic models.

Provides Param(), a drop-in replacement for pydantic.Field() that adds
units, shape, and example metadata to the JSON schema. Works on any
Pydantic BaseModel, including fields typed as InlineArray.
"""

from typing import Any, Callable

from pydantic.fields import FieldInfo


class ParamInfo(FieldInfo):  # type: ignore[final]
    """FieldInfo subclass that adds units, shape, and example to the JSON schema.

    Works on any Pydantic BaseModel. When used with InlineArray, the units
    are preserved in the JSON schema output.
    """

    units: str
    shape: tuple[int, ...] | None
    example: Any

    def __init__(
        self,
        default: Any = ...,
        *,
        units: str,
        shape: tuple[int, ...] | None = None,
        example: Any = None,
        default_factory: Callable[[], Any] | None = None,
        alias: str | None = None,
        alias_priority: int | None = None,
        validation_alias: str | None = None,
        serialization_alias: str | None = None,
        title: str | None = None,
        description: str | None = None,
        exclude: bool = False,
        deprecated: str | bool | None = None,
        json_schema_extra: dict[str, Any] | Callable[[dict[str, Any]], None] | None = None,
        frozen: bool | None = None,
        validate_default: bool | None = None,
        repr: bool = True,
        init: bool | None = None,
        init_var: bool | None = None,
        kw_only: bool | None = None,
        pattern: str | None = None,
        strict: bool | None = None,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        multiple_of: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
    ):
        has_default = default is not ... or default_factory is not None
        has_example = example is not None
        if not has_default and not has_example:
            raise ValueError(
                "Param() requires either a default value or an example. "
                "Use Param(default_value, units=...) or Param(units=..., example=...)."
            )

        self.units = units
        self.shape = shape
        self.example = example

        examples: list[Any] | None = [example] if example is not None else None

        extra: dict[str, Any] = {"units": units}
        if shape is not None:
            extra["shape"] = shape
        if json_schema_extra is not None and isinstance(json_schema_extra, dict):
            extra.update(json_schema_extra)

        super().__init__(
            default=default,
            default_factory=default_factory,
            alias=alias,
            alias_priority=alias_priority,
            validation_alias=validation_alias,
            serialization_alias=serialization_alias,
            title=title,
            description=description,
            examples=examples,
            exclude=exclude,
            deprecated=deprecated,
            json_schema_extra=extra,
            frozen=frozen,
            validate_default=validate_default,
            repr=repr,
            init=init,
            init_var=init_var,
            kw_only=kw_only,
            pattern=pattern,
            strict=strict,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            multiple_of=multiple_of,
            min_length=min_length,
            max_length=max_length,
        )


def Param(
    default: Any = ...,
    *,
    units: str,
    shape: tuple[int, ...] | None = None,
    example: Any = None,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    alias_priority: int | None = None,
    validation_alias: str | None = None,
    serialization_alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    exclude: bool = False,
    deprecated: str | bool | None = None,
    json_schema_extra: dict[str, Any] | Callable[[dict[str, Any]], None] | None = None,
    frozen: bool | None = None,
    validate_default: bool | None = None,
    repr: bool = True,
    init: bool | None = None,
    init_var: bool | None = None,
    kw_only: bool | None = None,
    pattern: str | None = None,
    strict: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> Any:
    """Define a parameter with units and optional constraints.

    Drop-in replacement for pydantic.Field that adds units, shape, and
    example metadata to the JSON schema. Works on any Pydantic BaseModel.
    """
    return ParamInfo(
        default=default, units=units, shape=shape, example=example,
        default_factory=default_factory, alias=alias, alias_priority=alias_priority,
        validation_alias=validation_alias, serialization_alias=serialization_alias,
        title=title, description=description, exclude=exclude, deprecated=deprecated,
        json_schema_extra=json_schema_extra, frozen=frozen, validate_default=validate_default,
        repr=repr, init=init, init_var=init_var, kw_only=kw_only, pattern=pattern,
        strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of,
        min_length=min_length, max_length=max_length,
    )
