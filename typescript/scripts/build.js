/**
 * Build script for the meshly library
 */
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create a require function to load JSON
const require = createRequire(import.meta.url);

// Clean up any existing build artifacts
console.log('Cleaning up...');
const distDir = path.join(__dirname, '..', 'dist');
if (fs.existsSync(distDir)) {
  fs.rmSync(distDir, { recursive: true, force: true });
}
fs.mkdirSync(distDir, { recursive: true });

// Run TypeScript compiler
console.log('Compiling TypeScript...');
try {
  // Use --outDir explicitly to ensure files go to the right place
  execSync('pnpm tsc --project tsconfig.json --outDir ./dist', {
    stdio: 'inherit',
    cwd: path.join(__dirname, '..')
  });
  console.log('TypeScript compilation successful!');
} catch (error) {
  console.error('TypeScript compilation failed:', error.message);
  process.exit(1);
}

// Copy package.json to dist (with modifications)
console.log('Copying package.json to dist...');
const packageJson = require('../package.json');

// Remove development-only fields
delete packageJson.devDependencies;
delete packageJson.scripts.test;

// Update paths
packageJson.main = 'index.js';
packageJson.types = 'index.d.ts';

// Write the modified package.json to dist
fs.writeFileSync(
  path.join(distDir, 'package.json'),
  JSON.stringify(packageJson, null, 2)
);

// Copy README.md to dist
console.log('Copying README.md to dist...');
const readmePath = path.join(__dirname, '..', 'README.md');
if (fs.existsSync(readmePath)) {
  fs.copyFileSync(readmePath, path.join(distDir, 'README.md'));
} else {
  console.warn('README.md not found, skipping...');
}

console.log('Build completed successfully!');