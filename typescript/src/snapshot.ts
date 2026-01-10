import JSZip from "jszip"
import { ArrayMetadata, ArrayUtils } from "./array"


/**
 * Metadata for a field (without the actual array data)
 */
export interface FieldMetadata {
    name: string
    type: string
    units: string | null
    shape: number[]
    dtype: string
    itemsize: number
}

/**
 * Metadata for a snapshot
 */
export interface SnapshotMetadata {
    time: number
    fields: Record<string, FieldMetadata>
}

/**
 * Field data with decoded array
 */
export interface FieldData {
    name: string
    type: string
    data: Float32Array | Uint32Array
    units: string | null
    shape: number[]
}

/**
 * Result of loading a snapshot
 */
export interface SnapshotResult {
    time: number
    fields: Record<string, FieldData>
}

/**
 * Utility class for loading snapshots from zip files
 */
export class SnapshotUtils {
    /**
     * Load snapshot metadata from a zip file without decoding field arrays.
     * Useful for inspecting snapshot contents before loading specific fields.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @returns Promise resolving to SnapshotMetadata
     */
    static async loadMetadata(
        zipInput: JSZip | ArrayBuffer | Uint8Array
    ): Promise<SnapshotMetadata> {
        const zip = zipInput instanceof JSZip ? zipInput : await JSZip.loadAsync(zipInput)

        // Load metadata.json
        const metadataFile = zip.file("metadata.json")
        if (!metadataFile) {
            throw new Error("metadata.json not found in snapshot zip file")
        }

        const metadataText = await metadataFile.async("text")
        return JSON.parse(metadataText) as SnapshotMetadata
    }

    /**
     * Load a specific field from a snapshot zip file.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @param fieldName The name of the field to load
     * @returns Promise resolving to FieldData
     */
    static async loadField(
        zipInput: JSZip | ArrayBuffer | Uint8Array,
        fieldName: string
    ): Promise<FieldData> {
        const zip = zipInput instanceof JSZip ? zipInput : await JSZip.loadAsync(zipInput)

        // Load metadata to get field info
        const metadata = await SnapshotUtils.loadMetadata(zip)

        const fieldMeta = metadata.fields[fieldName]
        if (!fieldMeta) {
            throw new Error(`Field "${fieldName}" not found in snapshot. Available fields: ${Object.keys(metadata.fields).join(", ")}`)
        }

        // Load the encoded field data
        const fieldPath = `fields/${fieldName}.bin`
        const fieldFile = zip.file(fieldPath)
        if (!fieldFile) {
            throw new Error(`Field data file "${fieldPath}" not found in zip`)
        }

        const encodedData = await fieldFile.async("uint8array")

        // Decode using ArrayUtils
        const arrayMetadata: ArrayMetadata = {
            shape: fieldMeta.shape,
            dtype: fieldMeta.dtype,
            itemsize: fieldMeta.itemsize
        }

        const decodedArray = ArrayUtils.decodeArray(encodedData, arrayMetadata)

        return {
            name: fieldMeta.name,
            type: fieldMeta.type,
            data: decodedArray,
            units: fieldMeta.units,
            shape: fieldMeta.shape
        }
    }

    /**
     * Load multiple fields from a snapshot zip file.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @param fieldNames Array of field names to load. If empty or undefined, loads all fields.
     * @returns Promise resolving to Record of field name to FieldData
     */
    static async loadFields(
        zipInput: JSZip | ArrayBuffer | Uint8Array,
        fieldNames?: string[]
    ): Promise<Record<string, FieldData>> {
        const zip = zipInput instanceof JSZip ? zipInput : await JSZip.loadAsync(zipInput)

        // Load metadata
        const metadata = await SnapshotUtils.loadMetadata(zip)

        // Determine which fields to load
        const fieldsToLoad = fieldNames && fieldNames.length > 0
            ? fieldNames
            : Object.keys(metadata.fields)

        // Load each field
        const fields: Record<string, FieldData> = {}
        for (const fieldName of fieldsToLoad) {
            fields[fieldName] = await SnapshotUtils.loadField(zip, fieldName)
        }

        return fields
    }

    /**
     * Load an entire snapshot from a zip file.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @returns Promise resolving to SnapshotResult containing time and all fields
     */
    static async loadFromZip(
        zipInput: JSZip | ArrayBuffer | Uint8Array
    ): Promise<SnapshotResult> {
        const zip = zipInput instanceof JSZip ? zipInput : await JSZip.loadAsync(zipInput)

        // Load metadata
        const metadata = await SnapshotUtils.loadMetadata(zip)

        // Load all fields
        const fields = await SnapshotUtils.loadFields(zip)

        return {
            time: metadata.time,
            fields
        }
    }

    /**
     * Get list of field names in a snapshot without loading the field data.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @returns Promise resolving to array of field names
     */
    static async getFieldNames(
        zipInput: JSZip | ArrayBuffer | Uint8Array
    ): Promise<string[]> {
        const metadata = await SnapshotUtils.loadMetadata(zipInput)
        return Object.keys(metadata.fields)
    }

    /**
     * Get the time value from a snapshot without loading field data.
     *
     * @param zipInput The zip file as JSZip, ArrayBuffer, or Uint8Array
     * @returns Promise resolving to the time value
     */
    static async getTime(
        zipInput: JSZip | ArrayBuffer | Uint8Array
    ): Promise<number> {
        const metadata = await SnapshotUtils.loadMetadata(zipInput)
        return metadata.time
    }
}
