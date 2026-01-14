/**
 * Meshly Library
 * 
 * A TypeScript library for mesh decoding with meshoptimizer decompression.
 * Compatible with the Python meshly library for cross-platform mesh serialization.
 */

// Export from packable module
export {
    Packable,
    PackableMetadata
} from './packable'

// Export from array module
export { ArrayMetadata, ArrayType, ArrayUtils, EncodedArray } from './array'

// Export from mesh module
export {
    Mesh,
    MeshData,
    MeshMetadata,
    MeshSize
} from './mesh'

