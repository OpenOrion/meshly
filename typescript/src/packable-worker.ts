/**
 * General-purpose Packable Worker
 * 
 * A web worker for offloading Packable reconstruction to a background thread.
 * This handles all the CPU-intensive work of decoding nested Packables and
 * reconstructing arrays from binary data.
 * 
 * Usage from main thread:
 * ```typescript
 * import { PackableWorkerClient } from 'meshly'
 * 
 * const client = new PackableWorkerClient()
 * const reconstructed = await client.reconstruct(data, assets)
 * ```
 */

import { ArrayRefInfo, ArrayUtils } from './array'
import { JsonSchema } from './json-schema'
import { Mesh } from './mesh'
import { Packable } from './packable'

// ─── Worker Message Types ───

export interface ReconstructMessage {
  type: 'reconstruct'
  requestId: string
  /** The data object with $ref values to resolve */
  data: Record<string, unknown>
  /** Map of checksum -> binary data (pre-fetched assets) */
  assets: Record<string, ArrayBuffer>
  /** Optional JSON schema for type-aware reconstruction */
  jsonSchema?: JsonSchema
}

export interface DecodeMessage {
  type: 'decode'
  requestId: string
  /** The Packable zip data to decode */
  zipData: ArrayBuffer
}

export interface DecodeArrayMessage {
  type: 'decodeArray'
  requestId: string
  /** The binary array data */
  data: ArrayBuffer
  /** Array metadata (shape, dtype, etc.) */
  info: ArrayRefInfo
}

export interface DecodeMeshMessage {
  type: 'decodeMesh'
  requestId: string
  zipData: ArrayBuffer
}

export type PackableWorkerMessage = ReconstructMessage | DecodeMessage | DecodeArrayMessage | DecodeMeshMessage

// ─── Worker Response Types ───

export interface ReconstructResponse {
  type: 'reconstructed'
  requestId: string
  /** The fully reconstructed data with all refs resolved */
  result: Record<string, unknown>
  /** Transferable buffers that were created */
  transferables?: ArrayBuffer[]
  error?: string
}

export interface DecodeResponse {
  type: 'decoded'
  requestId: string
  result: Record<string, unknown> | null
  error?: string
}

export interface DecodeArrayResponse {
  type: 'arrayDecoded'
  requestId: string
  result: Float32Array | Int32Array | Uint32Array | null
  error?: string
}

export interface DecodeMeshResponse {
  type: 'meshDecoded'
  requestId: string
  result: Record<string, { positions: Float32Array; indices: Uint32Array; normals: Float32Array }> | null
  error?: string
}

export type PackableWorkerResponse = ReconstructResponse | DecodeResponse | DecodeArrayResponse | DecodeMeshResponse

// ─── Worker Implementation ───

/**
 * Collect all ArrayBuffers from a reconstructed object for transfer.
 */
function collectTransferables(obj: unknown, transferables: Set<ArrayBuffer>): void {
  if (obj === null || obj === undefined) return

  if (ArrayBuffer.isView(obj)) {
    transferables.add((obj as { buffer: ArrayBuffer }).buffer)
    return
  }

  if (obj instanceof ArrayBuffer) {
    transferables.add(obj)
    return
  }

  if (Array.isArray(obj)) {
    for (const item of obj) {
      collectTransferables(item, transferables)
    }
    return
  }

  if (typeof obj === 'object') {
    for (const value of Object.values(obj)) {
      collectTransferables(value, transferables)
    }
  }
}

/**
 * Convert a Mesh to raw geometry arrays (positions, indices, normals) without THREE.js.
 * Uses Mesh.triangulateIndices for cell-type-aware triangulation.
 */
function meshToRawGeometry(mesh: Mesh): { positions: Float32Array; indices: Uint32Array; normals: Float32Array } {
  const positions = mesh.vertices
  const triIndices = mesh.indices
    ? Mesh.triangulateIndices(mesh.indices, mesh.indexSizes, mesh.cellTypes, mesh.vertices)
    : new Uint32Array(0)

  const normals = new Float32Array(positions.length)

  // Accumulate face normals onto each vertex
  for (let i = 0; i < triIndices.length; i += 3) {
    const i0 = triIndices[i] * 3
    const i1 = triIndices[i + 1] * 3
    const i2 = triIndices[i + 2] * 3

    // Edge vectors from v0
    const ax = positions[i1] - positions[i0]
    const ay = positions[i1 + 1] - positions[i0 + 1]
    const az = positions[i1 + 2] - positions[i0 + 2]
    const bx = positions[i2] - positions[i0]
    const by = positions[i2 + 1] - positions[i0 + 1]
    const bz = positions[i2 + 2] - positions[i0 + 2]

    // Cross product (face normal, area-weighted)
    const nx = ay * bz - az * by
    const ny = az * bx - ax * bz
    const nz = ax * by - ay * bx

    normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz
    normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz
    normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz
  }

  // Normalize
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.sqrt(normals[i] ** 2 + normals[i + 1] ** 2 + normals[i + 2] ** 2)
    if (len > 0) {
      normals[i] /= len
      normals[i + 1] /= len
      normals[i + 2] /= len
    }
  }

  return { positions, indices: triIndices, normals }
}

/**
 * Initialize the worker message handler.
 * Call this at the top level of your worker file.
 */
export function initPackableWorker(): void {
  self.addEventListener('message', async (event: MessageEvent<PackableWorkerMessage>) => {
    const message = event.data

    if (message.type === 'reconstruct') {
      const { requestId, data, assets, jsonSchema } = message
      try {
        // Convert ArrayBuffer map to Uint8Array map for asset provider
        const assetMap = new Map<string, Uint8Array>()
        for (const [checksum, buffer] of Object.entries(assets)) {
          assetMap.set(checksum, new Uint8Array(buffer))
        }

        // Create asset provider from pre-fetched assets
        const assetProvider = async (checksum: string): Promise<Uint8Array> => {
          const asset = assetMap.get(checksum)
          if (!asset) {
            throw new Error(`Missing asset with checksum '${checksum}'`)
          }
          return asset
        }

        // Reconstruct with all refs resolved
        const result = await Packable.reconstruct(data, assetProvider, jsonSchema)

        // Note: We intentionally do NOT transfer ArrayBuffers here.
        // Transferring detaches the buffers from the TypedArrays before
        // the message is serialized, causing the main thread to receive
        // detached/empty arrays. Structured clone is used instead.
        const response: ReconstructResponse = {
          type: 'reconstructed',
          requestId,
          result,
          transferables: [],
        }

        self.postMessage(response)
      } catch (error) {
        const response: ReconstructResponse = {
          type: 'reconstructed',
          requestId,
          result: {},
          error: error instanceof Error ? error.message : String(error),
        }
        self.postMessage(response)
      }
    } else if (message.type === 'decode') {
      const { requestId, zipData } = message
      try {
        const packable = await Packable.decode(zipData)
        // Extract plain object from Packable instance
        const result = { ...packable } as Record<string, unknown>

        // Note: We intentionally do NOT transfer ArrayBuffers here.
        // Transferring detaches the buffers from the TypedArrays before
        // the message is serialized, causing the main thread to receive
        // detached/empty arrays. Structured clone is used instead.
        const response: DecodeResponse = {
          type: 'decoded',
          requestId,
          result,
        }

        self.postMessage(response)
      } catch (error) {
        const response: DecodeResponse = {
          type: 'decoded',
          requestId,
          result: null,
          error: error instanceof Error ? error.message : String(error),
        }
        self.postMessage(response)
      }
    } else if (message.type === 'decodeArray') {
      const { requestId, data, info } = message
      try {
        const bytes = new Uint8Array(data)
        const result = ArrayUtils.reconstruct({
          data: bytes,
          info,
          encoding: 'array',
        }, true) as Float32Array | Int32Array | Uint32Array

        const response: DecodeArrayResponse = {
          type: 'arrayDecoded',
          requestId,
          result,
        }

        self.postMessage(response, { transfer: [result.buffer] })
      } catch (error) {
        const response: DecodeArrayResponse = {
          type: 'arrayDecoded',
          requestId,
          result: null,
          error: error instanceof Error ? error.message : String(error),
        }
        self.postMessage(response)
      }
    } else if (message.type === 'decodeMesh') {
      const { requestId, zipData } = message
      try {
        const mesh = await Mesh.decode(zipData)
        const result: Record<string, { positions: Float32Array; indices: Uint32Array; normals: Float32Array }> = {}
        const transferables = new Set<ArrayBuffer>()

        const markerNames = mesh.markers ? Object.keys(mesh.markers) : []
        if (markerNames.length > 0) {
          for (const name of markerNames) {
            const subMesh = mesh.extractByMarker(name)
            const geo = meshToRawGeometry(subMesh)
            result[name] = geo
            transferables.add(geo.positions.buffer)
            transferables.add(geo.indices.buffer)
            transferables.add(geo.normals.buffer)
          }
        } else {
          const geo = meshToRawGeometry(mesh)
          result['default'] = geo
          transferables.add(geo.positions.buffer)
          transferables.add(geo.indices.buffer)
          transferables.add(geo.normals.buffer)
        }

        const response: DecodeMeshResponse = {
          type: 'meshDecoded',
          requestId,
          result,
        }

        self.postMessage(response, { transfer: Array.from(transferables) })
      } catch (error) {
        const response: DecodeMeshResponse = {
          type: 'meshDecoded',
          requestId,
          result: null,
          error: error instanceof Error ? error.message : String(error),
        }
        self.postMessage(response)
      }
    }
  })

  // Signal ready
  self.postMessage({ type: 'ready' })
}

// Auto-initialize when this module is loaded directly as a worker entry point.
// When imported as a regular module (e.g. by PackableWorkerClient), this is a no-op
// because WorkerGlobalScope is not defined on the main thread.
if (typeof window === 'undefined' && typeof self !== 'undefined') {
  initPackableWorker()
}

// ─── Main Thread Client ───

/**
 * Client for communicating with a Packable worker from the main thread.
 * 
 * @example
 * ```typescript
 * // Create worker (bundler handles the worker file)
 * const worker = new Worker(new URL('meshly/packable-worker', import.meta.url))
 * const client = new PackableWorkerClient(worker)
 * 
 * // Reconstruct data with pre-fetched assets
 * const assets = await fetchAllAssets(checksums)
 * const result = await client.reconstruct(data, assets)
 * ```
 */
export class PackableWorkerClient {
  private worker: Worker
  private pendingRequests = new Map<string, {
    resolve: (value: unknown) => void
    reject: (error: Error) => void
  }>()
  private requestCounter = 0
  private ready: Promise<void>

  constructor(worker: Worker) {
    this.worker = worker

    this.ready = new Promise((resolve) => {
      const onReady = (event: MessageEvent) => {
        if (event.data.type === 'ready') {
          this.worker.removeEventListener('message', onReady)
          resolve()
        }
      }
      this.worker.addEventListener('message', onReady)
    })

    this.worker.addEventListener('message', this.handleMessage.bind(this))
  }

  private handleMessage(event: MessageEvent<PackableWorkerResponse | { type: 'ready' }>) {
    const message = event.data
    if (message.type === 'ready') return

    const requestId = (message as PackableWorkerResponse).requestId
    const pending = this.pendingRequests.get(requestId)
    if (!pending) return

    this.pendingRequests.delete(requestId)

    if ('error' in message && message.error) {
      pending.reject(new Error(message.error))
    } else {
      pending.resolve((message as any).result)
    }
  }

  private generateRequestId(): string {
    return `req_${++this.requestCounter}_${Date.now()}`
  }

  /**
   * Wait for the worker to be ready.
   */
  async waitForReady(): Promise<void> {
    return this.ready
  }

  /**
   * Reconstruct a Packable data object with all $refs resolved.
   * 
   * @param data - The data object with $ref values
   * @param assets - Map of checksum -> binary data (pre-fetched)
   * @param jsonSchema - Optional JSON schema for type-aware reconstruction
   * @returns The fully reconstructed data with TypedArrays
   */
  async reconstruct<T = Record<string, unknown>>(
    data: Record<string, unknown>,
    assets: Map<string, Uint8Array> | Record<string, Uint8Array>,
    jsonSchema?: JsonSchema
  ): Promise<T> {
    await this.ready

    const requestId = this.generateRequestId()

    // Convert Map to plain object with ArrayBuffers for transfer
    const assetsObj: Record<string, ArrayBuffer> = {}
    const transferables: ArrayBuffer[] = []

    const entries = assets instanceof Map ? assets.entries() : Object.entries(assets)
    for (const [checksum, uint8] of entries) {
      // Copy to new ArrayBuffer for transfer (original may be shared)
      const copy = uint8.buffer.slice(uint8.byteOffset, uint8.byteOffset + uint8.byteLength) as ArrayBuffer
      assetsObj[checksum] = copy
      transferables.push(copy)
    }

    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      })

      const message: ReconstructMessage = {
        type: 'reconstruct',
        requestId,
        data,
        assets: assetsObj,
        jsonSchema,
      }

      this.worker.postMessage(message, { transfer: transferables })
    })
  }

  /**
   * Decode a Packable zip blob.
   * 
   * @param zipData - The zip data to decode
   * @returns The decoded data object
   */
  async decode<T = Record<string, unknown>>(zipData: ArrayBuffer): Promise<T> {
    await this.ready

    const requestId = this.generateRequestId()

    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      })

      const message: DecodeMessage = {
        type: 'decode',
        requestId,
        zipData,
      }

      this.worker.postMessage(message, { transfer: [zipData] })
    })
  }

  /**
   * Decode an array from binary data.
   * 
   * @param data - The binary array data
   * @param info - Array metadata (shape, dtype, etc.)
   * @returns The decoded TypedArray
   */
  async decodeArray(
    data: ArrayBuffer,
    info: ArrayRefInfo
  ): Promise<Float32Array | Int32Array | Uint32Array> {
    await this.ready

    const requestId = this.generateRequestId()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      })

      const message: DecodeArrayMessage = {
        type: 'decodeArray',
        requestId,
        data,
        info,
      }

      this.worker.postMessage(message, { transfer: [data] })
    })
  }

  /**
   * Decode a mesh zip, triangulate per marker, and compute normals in the worker.
   * Handles 3D volume cell types (tet, hex, wedge, pyramid) by extracting boundary faces.
   */
  async decodeMesh(zipData: ArrayBuffer): Promise<Record<string, { positions: Float32Array; indices: Uint32Array; normals: Float32Array }>> {
    await this.ready

    const requestId = this.generateRequestId()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      })

      const message: DecodeMeshMessage = {
        type: 'decodeMesh',
        requestId,
        zipData,
      }

      this.worker.postMessage(message, { transfer: [zipData] })
    })
  }

  /**
   * Terminate the worker.
   */
  terminate(): void {
    this.worker.terminate()
    // Reject all pending requests
    for (const [, pending] of this.pendingRequests) {
      pending.reject(new Error('Worker terminated'))
    }
    this.pendingRequests.clear()
  }
}
