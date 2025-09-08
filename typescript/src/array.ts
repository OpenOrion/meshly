import { MeshoptEncoder, MeshoptDecoder } from "meshoptimizer"


/**
 * Interface representing an encoded array with metadata
 */
export interface EncodedArray {
  /**
   * The encoded data as bytes
   */
  data: Uint8Array;
  
  /**
   * Original array shape
   */
  shape: number[];
  
  /**
   * Original array data type
   */
  dtype: string;
  
  /**
   * Size of each item in bytes
   */
  itemsize: number;
}


/**
 * Metadata for an array
 */
export interface ArrayMetadata {
  shape: number[];
  dtype: string;
  itemsize: number;
}

/**
 * Utility class for encoding and decoding arrays
 */
export class ArrayUtils {
  /**
   * Encodes a Float32Array or Uint32Array using the meshoptimizer algorithm
   *
   * @param data Float32Array or Uint32Array to encode
   * @returns EncodedArray object containing the encoded data and metadata
   */
  static encodeArray(data: Float32Array | Uint32Array): EncodedArray {
    const sourceUint8Array = new Uint8Array(data.buffer)

    if (!(data instanceof Float32Array) && !(data instanceof Uint32Array)) {
      throw new Error('Array must be a Float32Array or Uint32Array');
    }

    const metadata = {
      shape: [data.length],
      dtype: data instanceof Float32Array ? 'float32' : 'uint32',
      itemsize: 4
    }
    const destUint8Array = MeshoptEncoder.encodeVertexBuffer(
      sourceUint8Array,
      metadata.shape[0],
      metadata.itemsize,
    )

    return {
      data: destUint8Array,
      shape: metadata.shape,
      dtype: metadata.dtype,
      itemsize: metadata.itemsize
    }
  }

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