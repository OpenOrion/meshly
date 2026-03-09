"""Tests for schema-based reconstruction (dynamic model builder + schema_utils).

Covers the full matrix of:
- Type-based resolution (resolve_from_class / _resolve_with_type)
- Schema-based resolution (resolve_from_schema / _resolve_with_prop)
- DynamicModelBuilder.instantiate
- Packable.reconstruct (typed vs dynamic)
- Round-trip via save/load to store
- Round-trip via encode/decode (zip)

Edge cases:
- Nested BaseModel inside dict[str, BaseModel]
- Nested Packable inside dict[str, Packable]
- Deeply nested structures (3+ levels)
- Union / discriminated union types
- Optional fields (None values)
- Lists of BaseModel
- Empty containers
- $defs propagation to nested schemas
"""

import json
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from meshly.array import Array
from meshly.packable import ExtractedPackable, Packable, PackableStore
from meshly.utils.dynamic_model import DynamicModelBuilder
from meshly.utils.json_schema import JsonSchema
from meshly.utils.schema_utils import SchemaUtils


# ============================================================================
# Test Models
# ============================================================================

class ScalarField(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    data: Array
    units: Optional[str] = None


class VectorField(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    data: Array
    units: Optional[str] = None


class Snapshot(Packable):
    time: float
    fields: dict[str, ScalarField] = Field(default_factory=dict)
    status: str = "running"


class ResultWithSnapshot(Packable):
    snapshot: Snapshot
    name: str


class Metadata(BaseModel):
    author: str
    version: int = 1


class NestedMetadata(BaseModel):
    meta: Metadata
    tags: List[str] = Field(default_factory=list)


class ResultWithNestedMeta(Packable):
    data: Array
    info: NestedMetadata


class DiscriminatedA(BaseModel):
    kind: Literal["a"] = "a"
    value_a: int


class DiscriminatedB(BaseModel):
    kind: Literal["b"] = "b"
    value_b: str


class ResultWithUnion(Packable):
    item: Union[DiscriminatedA, DiscriminatedB]
    label: str


class ResultWithOptionalArray(Packable):
    required_data: Array
    optional_data: Optional[Array] = None
    label: str = "default"


class InnerPackable(Packable):
    label: str
    data: Array


class OuterPackable(Packable):
    name: str
    inner: InnerPackable


class DictOfPackables(Packable):
    name: str
    items: Dict[str, InnerPackable] = Field(default_factory=dict)


class ListOfModels(Packable):
    entries: List[Metadata]


class DeeplyNested(Packable):
    """3-level nesting: Packable -> BaseModel -> BaseModel -> Array."""
    top_field: str
    nested: NestedMetadata
    snapshot: Snapshot


class EmptyFieldsResult(Packable):
    name: str
    fields: dict[str, ScalarField] = Field(default_factory=dict)


class MultipleArrayFields(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    primary: Array
    secondary: Optional[Array] = None


class MultiArraySnapshot(Packable):
    time: float
    fields: dict[str, MultipleArrayFields] = Field(default_factory=dict)


# ============================================================================
# Helpers
# ============================================================================

def _round_trip_typed(packable: Packable, cls: type):
    """Extract + reconstruct with the concrete class."""
    extracted = packable.extract()
    return cls.reconstruct(extracted)


def _round_trip_dynamic(packable: Packable):
    """Extract + reconstruct via base Packable (schema-based / dynamic)."""
    extracted = packable.extract()
    return Packable.reconstruct(extracted)


def _round_trip_zip(packable: Packable, cls: type):
    """Encode to zip bytes and decode back."""
    encoded = packable.encode()
    return cls.decode(encoded)


def _round_trip_zip_dynamic(packable: Packable):
    """Encode to zip bytes and decode via base Packable."""
    encoded = packable.encode()
    return Packable.decode(encoded)


def _round_trip_store(packable: Packable, cls: type):
    """Save/load via PackableStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PackableStore(root_dir=Path(tmpdir), extracted_dir="extracted")
        key = packable.save(store, "test_key")
        return cls.load(store, key)


def _round_trip_store_dynamic(packable: Packable):
    """Save/load via PackableStore with base Packable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PackableStore(root_dir=Path(tmpdir), extracted_dir="extracted")
        key = packable.save(store, "test_key")
        return Packable.load(store, key)


# ============================================================================
# Test: dict[str, BaseModel] with arrays -- the bug that was just fixed
# ============================================================================

class TestDictOfBaseModelWithArrays:
    """The core bug: dict[str, BaseModel] where BaseModel has array fields."""

    def _make_snapshot(self):
        return Snapshot(
            time=0.5,
            fields={
                "temperature": ScalarField(type="scalar", data=np.array([300.0, 301.0], dtype=np.float32), units="K"),
                "pressure": ScalarField(type="scalar", data=np.ones(3, dtype=np.float64)),
            },
        )

    def test_typed_reconstruct(self):
        snap = self._make_snapshot()
        restored = _round_trip_typed(snap, Snapshot)
        assert isinstance(restored.fields["temperature"], ScalarField)
        assert restored.fields["temperature"].type == "scalar"
        assert restored.fields["temperature"].units == "K"
        np.testing.assert_array_almost_equal(restored.fields["temperature"].data, [300.0, 301.0])

    def test_dynamic_reconstruct(self):
        snap = self._make_snapshot()
        restored = _round_trip_dynamic(snap)
        temp = restored.fields["temperature"]
        assert hasattr(temp, "type") and hasattr(temp, "data")
        assert temp.type == "scalar"
        assert temp.units == "K"
        np.testing.assert_array_almost_equal(temp.data, [300.0, 301.0])

    def test_zip_round_trip_typed(self):
        snap = self._make_snapshot()
        restored = _round_trip_zip(snap, Snapshot)
        assert isinstance(restored.fields["pressure"], ScalarField)
        np.testing.assert_array_almost_equal(restored.fields["pressure"].data, np.ones(3))

    def test_zip_round_trip_dynamic(self):
        snap = self._make_snapshot()
        restored = _round_trip_zip_dynamic(snap)
        assert restored.fields["pressure"].type == "scalar"
        np.testing.assert_array_almost_equal(restored.fields["pressure"].data, np.ones(3))

    def test_store_round_trip_typed(self):
        snap = self._make_snapshot()
        restored = _round_trip_store(snap, Snapshot)
        assert isinstance(restored.fields["temperature"], ScalarField)
        assert restored.fields["temperature"].units == "K"

    def test_store_round_trip_dynamic(self):
        snap = self._make_snapshot()
        restored = _round_trip_store_dynamic(snap)
        assert restored.fields["temperature"].units == "K"

    def test_empty_dict(self):
        snap = EmptyFieldsResult(name="empty", fields={})
        restored = _round_trip_dynamic(snap)
        assert restored.fields == {}

    def test_single_entry(self):
        snap = Snapshot(
            time=0.0,
            fields={"only": ScalarField(type="scalar", data=np.array([42.0]))},
        )
        restored = _round_trip_dynamic(snap)
        assert restored.fields["only"].type == "scalar"
        np.testing.assert_array_almost_equal(restored.fields["only"].data, [42.0])


# ============================================================================
# Test: dict[str, BaseModel] where BaseModel has multiple array fields
# ============================================================================

class TestDictOfBaseModelMultipleArrays:

    def test_multiple_arrays_in_dict_value(self):
        snap = MultiArraySnapshot(
            time=1.0,
            fields={
                "field_a": MultipleArrayFields(
                    type="vector",
                    primary=np.array([[1.0, 2.0], [3.0, 4.0]]),
                    secondary=np.array([10.0, 20.0]),
                ),
                "field_b": MultipleArrayFields(
                    type="scalar",
                    primary=np.array([5.0]),
                    secondary=None,
                ),
            },
        )
        restored = _round_trip_dynamic(snap)
        assert restored.fields["field_a"].type == "vector"
        np.testing.assert_array_almost_equal(restored.fields["field_a"].primary, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(restored.fields["field_a"].secondary, [10.0, 20.0])
        assert restored.fields["field_b"].secondary is None

    def test_typed_multiple_arrays(self):
        snap = MultiArraySnapshot(
            time=0.0,
            fields={"x": MultipleArrayFields(type="t", primary=np.zeros(2), secondary=np.ones(3))},
        )
        restored = _round_trip_typed(snap, MultiArraySnapshot)
        assert isinstance(restored.fields["x"], MultipleArrayFields)
        np.testing.assert_array_almost_equal(restored.fields["x"].secondary, np.ones(3))


# ============================================================================
# Test: Nested Packable inside another Packable
# ============================================================================

class TestNestedPackable:

    def test_typed_nested_packable(self):
        inner = InnerPackable(label="inner", data=np.array([1.0, 2.0]))
        outer = OuterPackable(name="outer", inner=inner)
        restored = _round_trip_typed(outer, OuterPackable)
        assert isinstance(restored.inner, InnerPackable)
        assert restored.inner.label == "inner"
        np.testing.assert_array_almost_equal(restored.inner.data, [1.0, 2.0])

    def test_dynamic_nested_packable(self):
        inner = InnerPackable(label="inner", data=np.array([3.0, 4.0]))
        outer = OuterPackable(name="outer", inner=inner)
        restored = _round_trip_dynamic(outer)
        assert restored.inner.label == "inner"
        np.testing.assert_array_almost_equal(restored.inner.data, [3.0, 4.0])

    def test_zip_nested_packable(self):
        inner = InnerPackable(label="z", data=np.array([5.0]))
        outer = OuterPackable(name="zip_test", inner=inner)
        restored = _round_trip_zip(outer, OuterPackable)
        assert isinstance(restored.inner, InnerPackable)
        np.testing.assert_array_almost_equal(restored.inner.data, [5.0])


# ============================================================================
# Test: Dict of Packables
# ============================================================================

class TestDictOfPackables:

    def test_typed_dict_of_packables(self):
        items = {
            "a": InnerPackable(label="a", data=np.array([1.0])),
            "b": InnerPackable(label="b", data=np.array([2.0, 3.0])),
        }
        container = DictOfPackables(name="test", items=items)
        restored = _round_trip_typed(container, DictOfPackables)
        assert isinstance(restored.items["a"], InnerPackable)
        assert restored.items["a"].label == "a"
        np.testing.assert_array_almost_equal(restored.items["b"].data, [2.0, 3.0])

    def test_dynamic_dict_of_packables(self):
        items = {"x": InnerPackable(label="x", data=np.array([10.0]))}
        container = DictOfPackables(name="dyn", items=items)
        restored = _round_trip_dynamic(container)
        assert restored.items["x"].label == "x"
        np.testing.assert_array_almost_equal(restored.items["x"].data, [10.0])


# ============================================================================
# Test: Packable containing Packable containing dict[str, BaseModel]
# ============================================================================

class TestPackableContainingSnapshot:

    def test_typed_result_with_snapshot(self):
        snap = Snapshot(
            time=1.0,
            fields={"vel": ScalarField(type="vector", data=np.array([1.0, 2.0, 3.0]))},
        )
        result = ResultWithSnapshot(snapshot=snap, name="test")
        restored = _round_trip_typed(result, ResultWithSnapshot)
        assert isinstance(restored.snapshot, Snapshot)
        assert isinstance(restored.snapshot.fields["vel"], ScalarField)
        np.testing.assert_array_almost_equal(restored.snapshot.fields["vel"].data, [1.0, 2.0, 3.0])

    def test_dynamic_result_with_snapshot(self):
        snap = Snapshot(
            time=2.0,
            fields={"p": ScalarField(type="scalar", data=np.array([100.0]), units="Pa")},
        )
        result = ResultWithSnapshot(snapshot=snap, name="dynamic")
        restored = _round_trip_dynamic(result)
        assert restored.snapshot.fields["p"].units == "Pa"
        assert restored.snapshot.fields["p"].type == "scalar"
        np.testing.assert_array_almost_equal(restored.snapshot.fields["p"].data, [100.0])

    def test_store_result_with_snapshot(self):
        snap = Snapshot(
            time=0.0,
            fields={"t": ScalarField(type="scalar", data=np.array([300.0]))},
        )
        result = ResultWithSnapshot(snapshot=snap, name="store")
        restored = _round_trip_store_dynamic(result)
        assert restored.snapshot.fields["t"].type == "scalar"


# ============================================================================
# Test: Nested BaseModel inside BaseModel (no arrays)
# ============================================================================

class TestNestedBaseModel:

    def test_typed_nested_basemodel(self):
        info = NestedMetadata(meta=Metadata(author="alice", version=2), tags=["a", "b"])
        result = ResultWithNestedMeta(data=np.array([1.0, 2.0]), info=info)
        restored = _round_trip_typed(result, ResultWithNestedMeta)
        assert restored.info.meta.author == "alice"
        assert restored.info.meta.version == 2
        assert restored.info.tags == ["a", "b"]

    def test_dynamic_nested_basemodel(self):
        info = NestedMetadata(meta=Metadata(author="bob"), tags=["x"])
        result = ResultWithNestedMeta(data=np.array([3.0]), info=info)
        restored = _round_trip_dynamic(result)
        assert restored.info.meta.author == "bob"
        assert restored.info.meta.version == 1
        assert restored.info.tags == ["x"]


# ============================================================================
# Test: Deeply nested (3+ levels)
# ============================================================================

class TestDeeplyNested:

    def test_typed_deeply_nested(self):
        snap = Snapshot(
            time=0.5,
            fields={"f": ScalarField(type="scalar", data=np.array([42.0]))},
        )
        nested = NestedMetadata(meta=Metadata(author="deep"), tags=["t"])
        obj = DeeplyNested(top_field="top", nested=nested, snapshot=snap)
        restored = _round_trip_typed(obj, DeeplyNested)
        assert restored.top_field == "top"
        assert restored.nested.meta.author == "deep"
        assert isinstance(restored.snapshot, Snapshot)
        assert restored.snapshot.fields["f"].type == "scalar"

    def test_dynamic_deeply_nested(self):
        snap = Snapshot(
            time=1.0,
            fields={"g": ScalarField(type="vector", data=np.zeros((2, 3)))},
        )
        nested = NestedMetadata(meta=Metadata(author="dyn", version=3), tags=[])
        obj = DeeplyNested(top_field="t", nested=nested, snapshot=snap)
        restored = _round_trip_dynamic(obj)
        assert restored.nested.meta.version == 3
        np.testing.assert_array_almost_equal(restored.snapshot.fields["g"].data, np.zeros((2, 3)))


# ============================================================================
# Test: Union / Discriminated Union types
# ============================================================================

class TestUnionTypes:

    def test_typed_discriminated_a(self):
        result = ResultWithUnion(item=DiscriminatedA(value_a=42), label="a_test")
        restored = _round_trip_typed(result, ResultWithUnion)
        assert isinstance(restored.item, DiscriminatedA)
        assert restored.item.kind == "a"
        assert restored.item.value_a == 42

    def test_typed_discriminated_b(self):
        result = ResultWithUnion(item=DiscriminatedB(value_b="hello"), label="b_test")
        restored = _round_trip_typed(result, ResultWithUnion)
        assert isinstance(restored.item, DiscriminatedB)
        assert restored.item.kind == "b"
        assert restored.item.value_b == "hello"

    def test_zip_discriminated_union(self):
        result = ResultWithUnion(item=DiscriminatedA(value_a=99), label="zip")
        restored = _round_trip_zip(result, ResultWithUnion)
        assert isinstance(restored.item, DiscriminatedA)
        assert restored.item.value_a == 99


# ============================================================================
# Test: Optional array fields
# ============================================================================

class TestOptionalFields:

    def test_optional_array_none(self):
        result = ResultWithOptionalArray(required_data=np.array([1.0, 2.0]), optional_data=None)
        restored = _round_trip_typed(result, ResultWithOptionalArray)
        assert restored.optional_data is None
        np.testing.assert_array_almost_equal(restored.required_data, [1.0, 2.0])

    def test_optional_array_present(self):
        result = ResultWithOptionalArray(
            required_data=np.array([1.0]),
            optional_data=np.array([3.0, 4.0]),
            label="with_opt",
        )
        restored = _round_trip_typed(result, ResultWithOptionalArray)
        np.testing.assert_array_almost_equal(restored.optional_data, [3.0, 4.0])
        assert restored.label == "with_opt"

    def test_dynamic_optional_none(self):
        result = ResultWithOptionalArray(required_data=np.array([5.0]))
        restored = _round_trip_dynamic(result)
        np.testing.assert_array_almost_equal(restored.required_data, [5.0])

    def test_dynamic_optional_present(self):
        result = ResultWithOptionalArray(
            required_data=np.array([1.0]),
            optional_data=np.array([2.0]),
        )
        restored = _round_trip_dynamic(result)
        np.testing.assert_array_almost_equal(restored.optional_data, [2.0])


# ============================================================================
# Test: List of BaseModel
# ============================================================================

class TestListOfModels:

    def test_typed_list_of_basemodel(self):
        entries = [Metadata(author="a"), Metadata(author="b", version=3)]
        obj = ListOfModels(entries=entries)
        restored = _round_trip_typed(obj, ListOfModels)
        assert len(restored.entries) == 2
        assert isinstance(restored.entries[0], Metadata)
        assert restored.entries[0].author == "a"
        assert restored.entries[1].version == 3

    def test_dynamic_list_of_basemodel(self):
        entries = [Metadata(author="x")]
        obj = ListOfModels(entries=entries)
        restored = _round_trip_dynamic(obj)
        assert len(restored.entries) == 1
        assert restored.entries[0].author == "x"


# ============================================================================
# Test: $defs propagation in schema-based resolution
# ============================================================================

class TestDefsPropagate:
    """The root cause of the FieldData bug: $defs not propagated to nested schemas."""

    def test_additionalproperties_ref_resolves(self):
        """dict[str, SomeModel] where SomeModel is defined in $defs."""
        snap = Snapshot(
            time=0.0,
            fields={"f": ScalarField(type="scalar", data=np.array([1.0]))},
        )
        schema_dict = snap.model_json_schema()
        schema = JsonSchema.model_validate(schema_dict)

        # The fields property should use additionalProperties with $ref
        fields_prop = schema.get_resolved_property("fields")
        assert fields_prop is not None
        assert fields_prop.additionalProperties is not None

        # $defs should have ScalarField
        assert len(schema.defs) > 0

        # Full round-trip via schema resolution
        extracted = snap.extract()
        resolved = SchemaUtils.resolve_from_schema(schema, extracted.data, extracted.assets)
        assert "fields" in resolved
        temp = resolved["fields"]["f"]
        assert hasattr(temp, "type")
        assert temp.type == "scalar"

    def test_nested_ref_in_nested_schema(self):
        """ResultWithSnapshot -> Snapshot -> dict[str, ScalarField].
        ScalarField is in $defs of ResultWithSnapshot, must be accessible
        when resolving the nested Snapshot schema."""
        snap = Snapshot(
            time=0.0,
            fields={"f": ScalarField(type="scalar", data=np.array([1.0]))},
        )
        result = ResultWithSnapshot(snapshot=snap, name="test")
        extracted = result.extract()

        schema_dict = result.model_json_schema()
        schema = JsonSchema.model_validate(schema_dict)

        # Verify $defs exist and have all needed types
        assert "ScalarField" in schema.defs
        assert "Snapshot" in schema.defs

        # Full schema-based resolution
        resolved = SchemaUtils.resolve_from_schema(schema, extracted.data, extracted.assets)
        assert resolved["snapshot"].fields["f"].type == "scalar"


# ============================================================================
# Test: Polymorphic reconstruction
# ============================================================================

class TestPolymorphicReconstruct:

    def test_reconstruct_polymorphic_matches_class(self):
        snap = Snapshot(time=0.0, fields={})
        result = ResultWithSnapshot(snapshot=snap, name="poly")
        extracted = result.extract()

        restored = Packable.reconstruct_polymorphic(
            [ResultWithSnapshot, ResultWithOptionalArray],
            extracted,
        )
        assert restored.name == "poly"

    def test_reconstruct_polymorphic_no_match_raises(self):
        snap = Snapshot(time=0.0, fields={})
        result = ResultWithSnapshot(snapshot=snap, name="nomatch")
        extracted = result.extract()

        with pytest.raises(ValueError, match="No matching model class"):
            Packable.reconstruct_polymorphic([ResultWithOptionalArray], extracted)

    def test_reconstruct_polymorphic_no_schema_raises(self):
        extracted = ExtractedPackable(data={}, json_schema=None, assets={})
        with pytest.raises(ValueError, match="json_schema is required"):
            Packable.reconstruct_polymorphic([ResultWithSnapshot], extracted)


# ============================================================================
# Test: DynamicModelBuilder edge cases
# ============================================================================

class TestDynamicModelBuilder:

    def test_build_model_caches(self):
        DynamicModelBuilder.clear_cache()
        schema = JsonSchema.model_validate({
            "title": "CacheTest",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        })
        m1 = DynamicModelBuilder.build_model(schema)
        m2 = DynamicModelBuilder.build_model(schema)
        assert m1 is m2

    def test_nested_object_properties_build(self):
        schema = JsonSchema.model_validate({
            "title": "WithNested",
            "properties": {
                "info": {
                    "type": "object",
                    "title": "Info",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["info"],
        })
        Model = DynamicModelBuilder.build_model(schema)
        instance = Model(info={"name": "test", "count": 5})
        # nested object is constructed as a model, not a plain dict
        assert hasattr(instance.info, "name")
        assert instance.info.name == "test"

    def test_instantiate_with_nested_and_assets(self):
        snap = Snapshot(
            time=0.5,
            fields={"f": ScalarField(type="scalar", data=np.array([99.0]))},
        )
        extracted = snap.extract()
        schema = JsonSchema.model_validate(snap.model_json_schema())
        instance = DynamicModelBuilder.instantiate(schema, extracted.data, extracted.assets)
        assert instance.time == 0.5
        assert instance.fields["f"].type == "scalar"
        np.testing.assert_array_almost_equal(instance.fields["f"].data, [99.0])


# ============================================================================
# Test: Checksum preservation across round-trips
# ============================================================================

class TestChecksumPreservation:

    def test_checksum_stable_after_decode(self):
        snap = Snapshot(
            time=0.0,
            fields={"f": ScalarField(type="s", data=np.array([1.0]))},
        )
        encoded = snap.encode()
        decoded = Snapshot.decode(encoded)
        assert decoded.checksum == snap.checksum

    def test_checksum_set_after_store_round_trip(self):
        snap = Snapshot(
            time=0.0,
            fields={"f": ScalarField(type="s", data=np.array([1.0]))},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PackableStore(root_dir=Path(tmpdir), extracted_dir="extracted")
            snap.save(store, "k")
            loaded = Snapshot.load(store, "k")
            assert loaded.checksum == snap.checksum


# ============================================================================
# Test: resolve_from_class type-hint resolution
# ============================================================================

class TestResolveFromClass:

    def test_dict_str_basemodel(self):
        snap = Snapshot(
            time=0.0,
            fields={"f": ScalarField(type="s", data=np.array([1.0]))},
        )
        extracted = snap.extract()
        resolved = SchemaUtils.resolve_from_class(Snapshot, extracted.data, extracted.assets)
        assert isinstance(resolved["fields"]["f"], ScalarField)

    def test_optional_none(self):
        result = ResultWithOptionalArray(required_data=np.array([1.0]))
        extracted = result.extract()
        resolved = SchemaUtils.resolve_from_class(
            ResultWithOptionalArray, extracted.data, extracted.assets
        )
        assert resolved.get("optional_data") is None

    def test_union_discriminated(self):
        result = ResultWithUnion(item=DiscriminatedB(value_b="hi"), label="u")
        extracted = result.extract()
        resolved = SchemaUtils.resolve_from_class(
            ResultWithUnion, extracted.data, extracted.assets
        )
        assert isinstance(resolved["item"], DiscriminatedB)
        assert resolved["item"].value_b == "hi"

    def test_list_of_basemodel(self):
        obj = ListOfModels(entries=[Metadata(author="a"), Metadata(author="b")])
        extracted = obj.extract()
        resolved = SchemaUtils.resolve_from_class(
            ListOfModels, extracted.data, extracted.assets
        )
        assert len(resolved["entries"]) == 2
        assert isinstance(resolved["entries"][0], Metadata)

    def test_nested_basemodel_in_basemodel(self):
        info = NestedMetadata(meta=Metadata(author="x"), tags=["t"])
        result = ResultWithNestedMeta(data=np.array([1.0]), info=info)
        extracted = result.extract()
        resolved = SchemaUtils.resolve_from_class(
            ResultWithNestedMeta, extracted.data, extracted.assets
        )
        assert isinstance(resolved["info"], NestedMetadata)
        assert isinstance(resolved["info"].meta, Metadata)
        assert resolved["info"].meta.author == "x"


# ============================================================================
# Test: Large / stress tests
# ============================================================================

class TestStress:

    def test_many_dict_entries(self):
        fields = {
            f"field_{i}": ScalarField(type="scalar", data=np.random.randn(100).astype(np.float32))
            for i in range(20)
        }
        snap = Snapshot(time=0.0, fields=fields)
        restored = _round_trip_dynamic(snap)
        for k, v in restored.fields.items():
            assert v.type == "scalar"
            assert v.data.shape == (100,)

    def test_large_array_in_nested_model(self):
        big = np.random.randn(10000, 3).astype(np.float32)
        snap = Snapshot(
            time=0.0,
            fields={"big": ScalarField(type="vector", data=big)},
        )
        restored = _round_trip_zip(snap, Snapshot)
        np.testing.assert_array_almost_equal(restored.fields["big"].data, big, decimal=5)
