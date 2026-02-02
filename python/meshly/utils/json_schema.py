"""
Pydantic models for JSON Schema validation.

Provides type-safe JSON Schema parsing and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field


class JsonSchemaProperty(BaseModel):
    """A single property in a JSON schema."""
    
    type: Union[str, None] = None
    """The type of the property (string, integer, number, boolean, array, object, null, or custom like vertex_buffer)."""
    
    title: str | None = None
    """Human-readable title."""
    
    description: str | None = None
    """Human-readable description."""
    
    default: Any = None
    """Default value."""
    
    ref: str | None = Field(default=None, alias="$ref")
    """Reference to a definition."""
    
    anyOf: list[JsonSchemaProperty] | None = None
    """Union types (e.g., Optional)."""
    
    allOf: list[JsonSchemaProperty] | None = None
    """Intersection types."""
    
    oneOf: list[JsonSchemaProperty] | None = None
    """Exclusive union types."""
    
    # Array-specific
    items: JsonSchemaProperty | None = None
    """Schema for array items."""
    
    minItems: int | None = None
    maxItems: int | None = None
    
    # Object-specific
    properties: dict[str, JsonSchemaProperty] | None = None
    """Properties for object types."""
    
    additionalProperties: Union[JsonSchemaProperty, bool, None] = None
    """Schema for additional properties (dict pattern)."""
    
    required: Union[List[str], None] = None
    """Required property names."""
    
    # String-specific
    pattern: Union[str, None] = None
    minLength: Union[int, None] = None
    maxLength: Union[int, None] = None
    format: Union[str, None] = None
    enum: Union[List[Any], None] = None
    
    # Number-specific
    minimum: Union[float, None] = None
    maximum: Union[float, None] = None
    exclusiveMinimum: Union[float, None] = None
    exclusiveMaximum: Union[float, None] = None
    multipleOf: Union[float, None] = None
    
    # Const
    const: Any = None
    
    model_config = {"extra": "allow", "populate_by_name": True}
    
    def is_array_type(self) -> bool:
        """Check if this is a meshly array type (not a JSON Schema array like list[str])."""
        # vertex_buffer and index_sequence are always meshly types
        if self.type in {"vertex_buffer", "index_sequence"}:
            return True
        # type="array" with items is a JSON Schema list, without items is a meshly array
        if self.type == "array":
            return self.items is None
        return False
    
    def is_resource_type(self) -> bool:
        """Check if this is a resource type (Resource)."""
        # Direct resource type or Resource schema
        return self.type == "resource" or self.title == "Resource"
    
    def is_optional(self) -> bool:
        """Check if this property is optional (anyOf with null)."""
        if self.anyOf:
            return any(opt.type == "null" for opt in self.anyOf)
        return False
    
    def get_inner_type(self) -> Union[JsonSchemaProperty, None]:
        """Get the non-null type from an Optional (anyOf with null)."""
        if self.anyOf:
            non_null = [opt for opt in self.anyOf if opt.type != "null"]
            return non_null[0] if non_null else None
        return self


class JsonSchema(BaseModel):
    """
    A validated JSON Schema document.
    
    This model validates that a JSON schema dict has the expected structure
    for use with DynamicModelBuilder.
    
    Usage:
        schema_dict = json.loads(schema_json)
        schema = JsonSchema.model_validate(schema_dict)
        # Now schema is validated and type-safe
    """
    
    schema_uri: Union[str, None] = Field(default=None, alias="$schema")
    """JSON Schema version URI."""
    
    id: Union[str, None] = Field(default=None, alias="$id")
    """Schema identifier."""
    
    title: Union[str, None] = None
    """Human-readable title for the schema."""
    
    description: Union[str, None] = None
    """Human-readable description."""
    
    type: Union[Literal["object"], None] = None
    """Root type (should be 'object' for Pydantic models)."""
    
    properties: Dict[str, JsonSchemaProperty] = Field(default_factory=dict)
    """Property definitions."""
    
    required: List[str] = Field(default_factory=list)
    """Required property names."""
    
    defs: Dict[str, JsonSchemaProperty] = Field(default_factory=dict, alias="$defs")
    """Reusable definitions."""
    
    additionalProperties: Union[bool, JsonSchemaProperty, None] = None
    """Whether additional properties are allowed."""
    
    model_config = {"extra": "allow", "populate_by_name": True}
    
    def get_property(self, name: str) -> Union[JsonSchemaProperty, None]:
        """Get a property by name."""
        return self.properties.get(name)
    
    def is_required(self, name: str) -> bool:
        """Check if a property is required."""
        return name in self.required
    
    def resolve_ref(self, ref: str) -> Union[JsonSchemaProperty, None]:
        """Resolve a $ref to its definition."""
        if ref.startswith("#/$defs/"):
            def_name = ref[8:]  # Remove "#/$defs/"
            return self.defs.get(def_name)
        return None
    
    def get_resolved_property(self, name: str) -> Union[JsonSchemaProperty, None]:
        """Get a property, resolving $ref if present."""
        prop = self.get_property(name)
        if prop and prop.ref:
            return self.resolve_ref(prop.ref)
        return prop
    
    def get_encoding(self, field_name: str) -> str:
        """Get the encoding type for a field (array, vertex_buffer, index_sequence)."""
        prop = self.get_resolved_property(field_name)
        if not prop:
            return "array"
        
        # Check direct type
        if prop.type and prop.is_array_type():
            return prop.type
        
        # Check anyOf (Optional types)
        if prop.anyOf:
            for opt in prop.anyOf:
                if opt.is_array_type():
                    return opt.type
                # Check $ref in anyOf
                if opt.ref:
                    resolved = self.resolve_ref(opt.ref)
                    if resolved and resolved.is_array_type():
                        return resolved.type
        
        return "array"
    
    def field_names(self) -> List[str]:
        """Get all non-metadata field names."""
        return [name for name in self.properties.keys() if not name.startswith("$")]
