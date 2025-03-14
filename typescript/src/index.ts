/**
 * Mesh Decoder Library
 * 
 * A TypeScript library for decoding Python meshoptimizer zip files into THREE.js geometries.
 */

// Export types
export * from './types'

// Export decoder functions
export { decodeArray, decodeIndexBuffer, decodeVertexBuffer, extractMeshFromZip } from './decoder'

// Export converter functions
export { convertToBufferGeometry } from './converter'

/**
 * Main function to load a mesh from a zip file and convert it to a THREE.js BufferGeometry
 * 
 * @param zipData Zip file data as an ArrayBuffer
 * @param options Options for the conversion
 * @returns Promise that resolves to a THREE.js BufferGeometry
 */
export async function loadZipAsBufferGeometry(
  zipData: ArrayBuffer,
  options?: import('./types').DecodeMeshOptions
): Promise<import('three').BufferGeometry> {
  const { extractMeshFromZip } = await import('./decoder')
  const { convertToBufferGeometry } = await import('./converter')

  const mesh = await extractMeshFromZip(zipData)
  return convertToBufferGeometry(mesh, options)
}