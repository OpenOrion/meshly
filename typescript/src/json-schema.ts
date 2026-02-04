/**
 * TypeScript interfaces for JSON Schema validation.
 * Matches Python's json_schema.py for cross-platform compatibility.
 */

/**
 * Array encoding types for serialization.
 * - "array": Generic meshoptimizer vertex buffer (works for any array)
 * - "vertex_buffer": Optimized for 3D vertex data (N x components)
 * - "index_sequence": Optimized for mesh indices (1D array)
 */
export type ArrayEncoding = "array" | "vertex_buffer" | "index_sequence"

/**
 * A single property in a JSON schema.
 */
export interface JsonSchemaProperty {
    /** The type of the property */
    type?: string

    /** Human-readable title */
    title?: string

    /** Human-readable description */
    description?: string

    /** Default value */
    default?: unknown

    /** Reference to a definition ($ref) */
    $ref?: string

    /** Union types (e.g., Optional) */
    anyOf?: JsonSchemaProperty[]

    /** Intersection types */
    allOf?: JsonSchemaProperty[]

    /** Exclusive union types */
    oneOf?: JsonSchemaProperty[]

    /** Schema for array items */
    items?: JsonSchemaProperty

    /** Min items for array */
    minItems?: number

    /** Max items for array */
    maxItems?: number

    /** Properties for object types */
    properties?: Record<string, JsonSchemaProperty>

    /** Schema for additional properties (dict pattern) */
    additionalProperties?: JsonSchemaProperty | boolean

    /** Required property names */
    required?: string[]

    /** String pattern */
    pattern?: string

    /** Min string length */
    minLength?: number

    /** Max string length */
    maxLength?: number

    /** String format */
    format?: string

    /** Enum values */
    enum?: unknown[]

    /** Numeric minimum */
    minimum?: number

    /** Numeric maximum */
    maximum?: number

    /** Exclusive minimum */
    exclusiveMinimum?: number

    /** Exclusive maximum */
    exclusiveMaximum?: number

    /** Multiple of */
    multipleOf?: number

    /** Const value */
    const?: unknown
}

/**
 * A validated JSON Schema document.
 */
export interface JsonSchema {
    /** JSON Schema version URI */
    $schema?: string

    /** Schema identifier */
    $id?: string

    /** Human-readable title for the schema */
    title?: string

    /** Human-readable description */
    description?: string

    /** Root type (should be 'object' for Pydantic models) */
    type?: "object" | string

    /** Property definitions */
    properties?: Record<string, JsonSchemaProperty>

    /** Required property names */
    required?: string[]

    /** Reusable definitions */
    $defs?: Record<string, JsonSchemaProperty>

    /** Whether additional properties are allowed */
    additionalProperties?: boolean | JsonSchemaProperty
}

/**
 * Utility functions for working with JSON Schema.
 */
export class JsonSchemaUtils {
    /**
     * Check if a property is a meshly array type (not a JSON Schema array like list[str]).
     */
    static isArrayType(prop: JsonSchemaProperty): boolean {
        // vertex_buffer and index_sequence are always meshly types
        if (prop.type === "vertex_buffer" || prop.type === "index_sequence") {
            return true
        }
        // type="array" with items is a JSON Schema list, without items is a meshly array
        if (prop.type === "array") {
            return prop.items === undefined
        }
        return false
    }

    /**
     * Check if a property is a resource type.
     */
    static isResourceType(prop: JsonSchemaProperty): boolean {
        return prop.type === "resource" || prop.title === "Resource"
    }

    /**
     * Check if a property is optional (anyOf with null).
     */
    static isOptional(prop: JsonSchemaProperty): boolean {
        if (prop.anyOf) {
            return prop.anyOf.some(opt => opt.type === "null")
        }
        return false
    }

    /**
     * Get the non-null type from an Optional (anyOf with null).
     */
    static getInnerType(prop: JsonSchemaProperty): JsonSchemaProperty | undefined {
        if (prop.anyOf) {
            const nonNull = prop.anyOf.filter(opt => opt.type !== "null")
            return nonNull.length > 0 ? nonNull[0] : undefined
        }
        return prop
    }

    /**
     * Resolve a $ref to its definition in the schema.
     */
    static resolveRef(schema: JsonSchema, ref: string): JsonSchemaProperty | undefined {
        if (ref.startsWith("#/$defs/")) {
            const defName = ref.slice(8) // Remove "#/$defs/"
            return schema.$defs?.[defName]
        }
        return undefined
    }

    /**
     * Get a property, resolving $ref if present.
     */
    static getResolvedProperty(
        schema: JsonSchema,
        name: string
    ): JsonSchemaProperty | undefined {
        const prop = schema.properties?.[name]
        if (prop?.$ref) {
            return JsonSchemaUtils.resolveRef(schema, prop.$ref)
        }
        return prop
    }

    /**
     * Get the encoding type for a field (array, vertex_buffer, index_sequence).
     */
    static getEncoding(schema: JsonSchema, fieldName: string): ArrayEncoding {
        const prop = JsonSchemaUtils.getResolvedProperty(schema, fieldName)
        if (!prop) {
            return "array"
        }

        // Check direct type
        if (prop.type && JsonSchemaUtils.isArrayType(prop)) {
            return prop.type as ArrayEncoding
        }

        // Check anyOf (Optional types)
        if (prop.anyOf) {
            for (const opt of prop.anyOf) {
                if (JsonSchemaUtils.isArrayType(opt)) {
                    return opt.type as ArrayEncoding
                }
                // Check $ref in anyOf
                if (opt.$ref) {
                    const resolved = JsonSchemaUtils.resolveRef(schema, opt.$ref)
                    if (resolved && JsonSchemaUtils.isArrayType(resolved)) {
                        return resolved.type as ArrayEncoding
                    }
                }
            }
        }

        return "array"
    }

    /**
     * Get all non-metadata field names.
     */
    static fieldNames(schema: JsonSchema): string[] {
        return Object.keys(schema.properties ?? {}).filter(name => !name.startsWith("$"))
    }
}
