import JSZip from "jszip"
import { MeshoptDecoder } from "meshoptimizer"

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
 * Metadata for an array - matches Python's ArrayMetadata
 */
export interface ArrayMetadata {
  shape: number[]
  dtype: string
  itemsize: number
  /** Array backend type (for Python compatibility) - defaults to "numpy" */
  array_type?: ArrayType
}

/**
 * Encoded array with data and metadata - matches Python's EncodedArray
 */
export interface EncodedArray {
  /** Encoded data as Uint8Array (bytes in Python) */
  data: Uint8Array
  /** Array metadata */
  metadata: ArrayMetadata
}

/**
 * Utility class for encoding and decoding arrays
 */
export class ArrayUtils {
  /**
   * Decodes an encoded array using the meshoptimizer algorithm
   *
   * @param encodedArray EncodedArray containing data and metadata
   * @returns Decoded array as a TypedArray
   */
  static decodeArray(encodedArray: EncodedArray): TypedArray {
    const { data, metadata } = encodedArray

    // Calculate the total number of items
    const totalItems = metadata.shape.reduce((acc, dim) => acc * dim, 1)

    // Create the output buffer
    const destUint8Array = new Uint8Array(totalItems * metadata.itemsize)

    MeshoptDecoder.decodeVertexBuffer(
      destUint8Array,
      totalItems,
      metadata.itemsize,
      data
    )

    // Convert to appropriate typed array based on dtype
    if (metadata.dtype === 'uint32') {
      return new Uint32Array(destUint8Array.buffer)
    } else {
      return new Float32Array(destUint8Array.buffer)
    }
  }

  /**
   * Load and decode a single array from a zip file.
   *
   * @param zip - JSZip instance to read from
   * @param name - Array name (e.g., "normals" or "markerIndices.boundary")
   * @returns Decoded typed array
   * @throws Error if array not found in zip
   */
  static async loadArray(
    zip: JSZip,
    name: string
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
    const metadata: ArrayMetadata = JSON.parse(metadataText)
    const data = await arrayFile.async("uint8array")

    const encodedArray: EncodedArray = { data, metadata }
    return ArrayUtils.decodeArray(encodedArray)
  }
}