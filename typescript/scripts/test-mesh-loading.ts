#!/usr/bin/env node
/**
 * Script to test loading all mesh zip files in the examples directory
 */

import * as fs from 'fs'
import * as path from 'path'
import { Mesh, MeshUtils } from '../src/index'


// Path to the meshes directory
const MESHES_DIR = path.resolve(__dirname, '../../python/examples/meshes')

// Options for loading meshes
const options = {
  normalize: true,
  computeNormals: true
}

/**
 * Main function to test loading all mesh zip files
 */
async function testMeshLoading() {
  console.log(`Testing mesh loading from ${MESHES_DIR}`)

  // Get all zip files in the directory
  const files = fs.readdirSync(MESHES_DIR)
    .filter(file => file.endsWith('.zip'))
    .map(file => path.join(MESHES_DIR, file))

  console.log(`Found ${files.length} zip files`)

  // Results tracking
  const results = {
    success: 0,
    failure: 0,
    errors: [] as { file: string, error: string }[]
  }

  // Process each file
  for (const file of files) {
    const filename = path.basename(file)
    process.stdout.write(`Testing ${filename}... `)

    try {
      // Read the file
      const zipData = fs.readFileSync(file)

      // Load the mesh
      const mesh = await MeshUtils.loadMeshFromZip<Mesh>(zipData.buffer)
      const geometry = await MeshUtils.loadZipAsBufferGeometry(zipData.buffer)

      // Check if the geometry is valid
      if (!geometry.getAttribute('position')) {
        throw new Error('Geometry does not have position attribute')
      }
      
      // Debug geometry attributes
      console.log(`Geometry attributes: ${Object.keys(geometry.attributes).join(', ')}`)
      // Log success
      process.stdout.write('✓\n')
      results.success++
    } catch (error) {
      // Log error
      process.stdout.write('✗\n')
      const errorMessage = error instanceof Error ? error.message : String(error)
      console.error(`  Error: ${errorMessage}`)

      results.failure++
      results.errors.push({
        file: filename,
        error: errorMessage
      })
    }
  }

  // Print summary
  console.log('\nSummary:')
  console.log(`Total files: ${files.length}`)
  console.log(`Successful: ${results.success}`)
  console.log(`Failed: ${results.failure}`)

  if (results.errors.length > 0) {
    console.log('\nErrors:')
    for (const { file, error } of results.errors) {
      console.log(`  ${file}: ${error}`)
    }
  }

  // Return success if all files loaded successfully
  return results.failure === 0
}

// Run the test
testMeshLoading()
  .then(success => {
    if (success) {
      console.log('\nAll meshes loaded successfully!')
      process.exit(0)
    } else {
      console.error('\nSome meshes failed to load.')
      process.exit(1)
    }
  })
  .catch(error => {
    console.error('Error running tests:', error)
    process.exit(1)
  })