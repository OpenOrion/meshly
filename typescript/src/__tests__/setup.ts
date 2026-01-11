import { MeshoptDecoder } from 'meshoptimizer'
import { beforeAll } from 'vitest'

// Initialize meshoptimizer before running tests
beforeAll(async () => {
  // Wait for meshoptimizer to be ready
  await MeshoptDecoder.ready
});

// Make meshoptimizer available globally
(global as any).MeshoptDecoder = MeshoptDecoder