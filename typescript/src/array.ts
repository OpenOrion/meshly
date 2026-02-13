import JSZip from "jszip"
import { MeshoptDecoder } from "meshoptimizer"
import { ArrayEncoding } from "./json-schema"

/**
 * TypedArray union for decoded array data
 */
export type TypedArray = Float32Array | Float64Array | Int8Array | Int16Array | Int32Array | Uint8Array | Uint16Array | Uint32Array

/**
 * Array backend type - matches Python's ArrayType for compatibility.
 * TypeScript always uses TypedArrays, but this field is stored in metadata
 * for cross-language compatibility with Python's numpy/jax arrays.
 */
export type ArrayType = "numpy" | "jax"

/**
 * Reference info for encoded arrays (stored in $ref in data.json).
 * Matches Python's ArrayRefInfo model.
 * 
 * Example $ref: {"$ref": "abc123", "dtype": "float32", "shape": [100, 3]}
 */
export interface ArrayRefInfo {
  /** Checksum reference (in serialized form as $ref) */
  $ref?: string
  /** Array shape */
  shape: number[]
  /** Data type (e.g., "float32", "uint32", "int64") */
  dtype: string
  /** Bytes per element */
  itemsize: number
  /** Padding bytes added for meshoptimizer alignment (optional) */
  pad_bytes?: number
}

/**
 * Extracted array with data and metadata - matches Python's ExtractedArray
 */
export interface ExtractedArray {
  /** Encoded data as Uint8Array (bytes in Python) */
  data: Uint8Array
  /** Array metadata */
  info: ArrayRefInfo
  /** Encoding type used */
  encoding?: ArrayEncoding
}

/**
 * Utility class for encoding and decoding arrays
 */
export class ArrayUtils {
  /**
   * Get the TypedArray constructor for a given dtype.
   */
  static getTypedArrayConstructor(dtype: string): new (buffer: ArrayBuffer) => TypedArray {
    switch (dtype) {
      case 'float32':
        return Float32Array
      case 'float64':
        return Float64Array
      case 'int8':
        return Int8Array
      case 'int16':
        return Int16Array
      case 'int32':
        return Int32Array
      case 'int64':
        // JavaScript doesn't have Int64Array, use Int32Array as fallback
        // Note: This may lose precision for large values
        return Int32Array
      case 'uint8':
        return Uint8Array
      case 'uint16':
        return Uint16Array
      case 'uint32':
        return Uint32Array
      case 'uint64':
        // JavaScript doesn't have Uint64Array, use Uint32Array as fallback
        return Uint32Array
      default:
        // Default to Float32Array for unknown types
        return Float32Array
    }
  }

  /**
   * Get the itemsize for a given dtype.
   */
  static getItemSize(dtype: string): number {
    switch (dtype) {
      case 'float32':
      case 'int32':
      case 'uint32':
        return 4
      case 'float64':
      case 'int64':
      case 'uint64':
        return 8
      case 'int16':
      case 'uint16':
        return 2
      case 'int8':
      case 'uint8':
        return 1
      default:
        return 4
    }
  }

  /**
   * Reconstruct array from ExtractedArray using meshoptimizer.
   * Supports all encoding types: array, vertex_buffer, index_sequence.
   *
   * @param extracted ExtractedArray containing data and metadata
   * @returns Decoded array as a TypedArray
   */
  static reconstruct(extracted: ExtractedArray): TypedArray {
    const { data, info } = extracted
    const encoding = extracted.encoding || "array"

    // Calculate the total number of items
    const totalItems = info.shape.reduce((acc, dim) => acc * dim, 1)
    const TypedArrayCtor = ArrayUtils.getTypedArrayConstructor(info.dtype)

    // Handle vertex_buffer encoding (optimized for 2D vertex data)
    if (encoding === "vertex_buffer") {
      const vertexCount = info.shape[0]
      const vertexSize = info.itemsize * (info.shape.length > 1 ? info.shape[1] : 1)
      const destUint8Array = new Uint8Array(vertexCount * vertexSize)

      MeshoptDecoder.decodeVertexBuffer(
        destUint8Array,
        vertexCount,
        vertexSize,
        data
      )

      return new TypedArrayCtor(destUint8Array.buffer)
    }

    // Handle index_sequence encoding (optimized for 1D indices)
    if (encoding === "index_sequence") {
      const indexCount = info.shape[0]
      const indexSize = info.itemsize
      const destUint8Array = new Uint8Array(indexCount * indexSize)

      MeshoptDecoder.decodeIndexSequence(
        destUint8Array,
        indexCount,
        indexSize,
        data
      )

      return new TypedArrayCtor(destUint8Array.buffer)
    }

    // Handle generic "array" encoding with optional padding
    const padBytes = info.pad_bytes || 0

    if (padBytes > 0) {
      // Decode with padded itemsize, then strip padding
      const paddedItemsize = info.itemsize + padBytes
      const destUint8Array = new Uint8Array(totalItems * paddedItemsize)

      MeshoptDecoder.decodeVertexBuffer(
        destUint8Array,
        totalItems,
        paddedItemsize,
        data
      )

      // Strip padding from each element
      const unpadded = new Uint8Array(totalItems * info.itemsize)
      for (let i = 0; i < totalItems; i++) {
        const srcStart = i * paddedItemsize
        unpadded.set(
          destUint8Array.subarray(srcStart, srcStart + info.itemsize),
          i * info.itemsize
        )
      }

      return new TypedArrayCtor(unpadded.buffer)
    }

    // Direct decode without padding
    const destUint8Array = new Uint8Array(totalItems * info.itemsize)

    MeshoptDecoder.decodeVertexBuffer(
      destUint8Array,
      totalItems,
      info.itemsize,
      data
    )

    return new TypedArrayCtor(destUint8Array.buffer)
  }

  /**
   * Decode a single array from a zip file.
   * Supports both old format (arrays/{name}/) and new format (uses schema for encoding).
   *
   * @param zip - JSZip instance to read from
   * @param name - Array name (e.g., "normals" or "markerIndices.boundary")
   * @param encoding - Encoding type (defaults to "array")
   * @returns Decoded typed array
   * @throws Error if array not found in zip
   */
  static async decode(
    zip: JSZip,
    name: string,
    encoding: ArrayEncoding = "array"
  ): Promise<TypedArray> {
    // Convert dotted name to path
    const arrayPath = name.replace(/\./g, "/")
    const arraysFolder = zip.folder("arrays")

    if (!arraysFolder) {
      throw new Error(`Array '${name}' not found in zip file`)
    }

    const metadataFile = arraysFolder.file(`${arrayPath}/metadata.json`)
    const arrayFile = arraysFolder.file(`${arrayPath}/array.bin`)

    if (!metadataFile || !arrayFile) {
      throw new Error(`Array '${name}' not found in zip file`)
    }

    const metadataText = await metadataFile.async("text")
    const info: ArrayRefInfo = JSON.parse(metadataText)
    const data = await arrayFile.async("uint8array")

    const extracted: ExtractedArray = { data, info, encoding }
    return ArrayUtils.reconstruct(extracted)
  }
}
