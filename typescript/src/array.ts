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
   *
   * @param extracted - ExtractedArray containing data and metadata
   * @param flat - If true, always return a single flat TypedArray (default: false).
   *               If false, 2-D+ arrays are split into per-column TypedArray[].
   */
  static reconstruct(extracted: ExtractedArray, flat?: boolean): TypedArray | TypedArray[] {
    const { data, info } = extracted
    const { shape } = info
    const encoding = extracted.encoding || "array"

    const totalItems = shape.reduce((acc, dim) => acc * dim, 1)
    const TypedArrayCtor = ArrayUtils.getTypedArrayConstructor(info.dtype)

    let decoded: TypedArray

    if (encoding === "vertex_buffer") {
      const vertexCount = shape[0]
      const vertexSize = info.itemsize * (shape.length > 1 ? shape[1] : 1)
      const dest = new Uint8Array(vertexCount * vertexSize)
      MeshoptDecoder.decodeVertexBuffer(dest, vertexCount, vertexSize, data)
      decoded = new TypedArrayCtor(dest.buffer)

    } else if (encoding === "index_sequence") {
      const indexCount = shape[0]
      const dest = new Uint8Array(indexCount * info.itemsize)
      MeshoptDecoder.decodeIndexSequence(dest, indexCount, info.itemsize, data)
      decoded = new TypedArrayCtor(dest.buffer)

    } else {
      const originalItemsize = info.itemsize

      if (originalItemsize % 4 !== 0) {
        const paddedSize = Math.ceil(originalItemsize / 4) * 4
        const dest = new Uint8Array(totalItems * paddedSize)
        MeshoptDecoder.decodeVertexBuffer(dest, totalItems, paddedSize, data)

        const unpadded = new Uint8Array(totalItems * originalItemsize)
        for (let i = 0; i < totalItems; i++) {
          unpadded.set(
            dest.subarray(i * paddedSize, i * paddedSize + originalItemsize),
            i * originalItemsize
          )
        }
        decoded = new TypedArrayCtor(unpadded.buffer)
      } else {
        const dest = new Uint8Array(totalItems * originalItemsize)
        MeshoptDecoder.decodeVertexBuffer(dest, totalItems, originalItemsize, data)
        decoded = new TypedArrayCtor(dest.buffer)
      }
    }

    if (flat || shape.length < 2 || shape[1] <= 1) {
      return decoded
    }

    const N = shape[0]
    const stride = shape[1]
    const Ctor = decoded.constructor as new (len: number) => TypedArray
    const columns: TypedArray[] = []
    for (let c = 0; c < stride; c++) {
      const col = new Ctor(N)
      for (let r = 0; r < N; r++) {
        col[r] = decoded[r * stride + c]
      }
      columns.push(col)
    }
    return columns
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
  ): Promise<TypedArray | TypedArray[]> {
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
