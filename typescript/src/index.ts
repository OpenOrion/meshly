/**
 * Mesh Decoder Library
 * 
 * A TypeScript library for decoding Python meshoptimizer zip files into THREE.js geometries.
 */

// Export types
export * from './mesh'
export * from './snapshot'

// Export utility classes
export { ArrayMetadata, ArrayUtils, EncodedArray } from './array'
export { Mesh, MeshMetadata, MeshSize, MeshUtils } from './mesh'
export { FieldData, FieldMetadata, SnapshotMetadata, SnapshotResult, SnapshotUtils } from './snapshot'

