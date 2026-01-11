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
export { ArrayMetadata, ArrayUtils, EncodedArray } from './array'

// Export from mesh module
export {
    EncodedMesh, Mesh,
    MeshData,
    MeshMetadata,
    MeshSize
} from './mesh'

// Export from utils module
export { ZipUtils } from './utils'

