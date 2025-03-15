import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import { beforeAll } from 'vitest';

// Initialize meshoptimizer before running tests
beforeAll(async () => {
  // Wait for meshoptimizer to be ready
  await Promise.all([
    MeshoptEncoder.ready,
    MeshoptDecoder.ready
  ]);
  
});

// Make meshoptimizer available globally
(global as any).MeshoptEncoder = MeshoptEncoder;
(global as any).MeshoptDecoder = MeshoptDecoder;