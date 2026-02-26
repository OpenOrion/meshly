/**
 * Packable Worker Entry Point
 * 
 * Import this file as a worker to get a ready-to-use Packable reconstruction worker.
 * 
 * @example
 * ```typescript
 * // Vite/Rollup
 * const worker = new Worker(new URL('meshly/worker', import.meta.url), { type: 'module' })
 * 
 * // Or with inline worker
 * import PackableWorker from 'meshly/worker?worker'
 * const worker = new PackableWorker()
 * ```
 */

import { initPackableWorker } from './packable-worker'

// Initialize the worker
initPackableWorker()
