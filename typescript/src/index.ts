/**
 * Meshly Library
 * 
 * A TypeScript library for mesh decoding with meshoptimizer decompression.
 * Compatible with the Python meshly library for cross-platform mesh serialization.
 */

// Export constants
export { ExportConstants } from './constants'

// Export from json-schema module
export {
    ArrayEncoding,
    JsonSchema,
    JsonSchemaProperty,
    JsonSchemaUtils
} from './json-schema'

// Export from packable module
export {
    DynamicModel,
    DynamicModelBuilder, ExtractedPackable,
    FieldSchema, InstantiateOptions, isArrayRef,
    isRefObject, LazyModel,
    LazyModelProps, Packable,
    PackableDecoder,
    PackableMetadata,
    ReconstructSchema,
    RefObject,
    SerializedPackableData
} from './packable'

// Export schema utils directly as well
export { SchemaUtils } from './schema-utils'

// Export from common module
export {
    AssetFetcher,
    AssetProvider
} from './common'

// Export from array module
export {
    ArrayRefInfo,
    ArrayType,
    ArrayUtils,
    ExtractedArray,
    TypedArray
} from './array'

// Export from mesh module
export {
    Mesh,
    MeshData
} from './mesh'

// Export from cache module
export {
    AssetCache,
    AssetCacheConfig,
    createCachedProvider,
    getDefaultAssetCache
} from './cache'

