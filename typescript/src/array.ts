import { MeshoptDecoder } from "meshoptimizer"

/**
 * Metadata for an array
 */
export interface ArrayMetadata {
  shape: number[]
  dtype: string
  itemsize: number
}

/**
 * Utility class for decoding arrays
 */
export class ArrayUtils {
  /**
   * Decodes an encoded array using the meshoptimizer algorithm
   *
   * @param data Encoded array data
   * @param metadata Array metadata
   * @returns Decoded array as a Float32Array or Uint32Array
   */
  static decodeArray(data: Uint8Array, metadata: ArrayMetadata): Float32Array | Uint32Array {
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
}