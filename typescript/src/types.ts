/**
 * Types for the mesh decoder library
 */

/**
 * Metadata for a mesh
 */
export interface MeshMetadata {
  vertex_count: number;
  vertex_size: number;
  index_count: number | null;
  index_size: number;
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
 * General metadata for a mesh file
 */
export interface MeshFileMetadata {
  class_name: string;
  module_name: string;
  model_data?: Record<string, any>;
}

/**
 * Decoded mesh data
 */
export interface DecodedMesh {
  vertices: Float32Array;
  indices?: Uint32Array;
  normals?: Float32Array;
  colors?: Float32Array;
  uvs?: Float32Array;
  [key: string]: Float32Array | Uint32Array | undefined;
}

/**
 * Options for decoding a mesh
 */
export interface DecodeMeshOptions {
  /**
   * Whether to normalize the mesh to fit within a unit cube
   * @default false
   */
  normalize?: boolean;
  
  /**
   * Whether to compute normals if they don't exist
   * @default true
   */
  computeNormals?: boolean;
}