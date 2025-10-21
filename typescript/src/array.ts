import { MeshoptEncoder, MeshoptDecoder } from "meshoptimizer"
import JSZip from "jszip"


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

  /**
   * Loads an array from a zip buffer containing array.bin and metadata.json
   *
   * @param zipBuffer The zip buffer containing the encoded array and metadata
   * @returns Promise resolving to the decoded typed array (Float32Array or Uint32Array)
   */
  static async loadFromZip(zipInput: JSZip | ArrayBuffer | Uint8Array): Promise<Float32Array | Uint32Array> {
    const zip = zipInput instanceof JSZip ? zipInput : await JSZip.loadAsync(zipInput)
    
    // Load metadata.json
    const metadataFile = zip.file("metadata.json")
    if (!metadataFile) {
      throw new Error("metadata.json not found in zip file")
    }
    
    const metadataText = await metadataFile.async("text")
    const metadata: ArrayMetadata = JSON.parse(metadataText)
    
    // Load array.bin
    const arrayFile = zip.file("array.bin")
    if (!arrayFile) {
      throw new Error("array.bin not found in zip file")
    }
    
    const arrayData = await arrayFile.async("uint8array")
    
    // Decode and return the array
    return ArrayUtils.decodeArray(arrayData, metadata)
  }

}