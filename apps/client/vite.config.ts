import { defineConfig } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

/**
 * Vite plugin to serve viewer.html for /view/* and /embed/* routes.
 * The viewer is a separate SPA entry point with its own React Router.
 */
function viewerFallback(): Plugin {
  return {
    name: 'viewer-fallback',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        if (req.url && (req.url.startsWith('/view/') || req.url.startsWith('/embed/'))) {
          req.url = '/viewer.html';
        }
        next();
      });
    },
  };
}

// three.js ships the Draco and Basis (KTX2) decoders as files the loaders fetch at runtime
const threeDir = path.resolve(path.dirname(createRequire(import.meta.url).resolve('three')), '..');
const DECODERS: Record<string, string> = {
  'decoders/basis/basis_transcoder.js': 'examples/jsm/libs/basis/basis_transcoder.js',
  'decoders/basis/basis_transcoder.wasm': 'examples/jsm/libs/basis/basis_transcoder.wasm',
  'decoders/draco/draco_decoder.js': 'examples/jsm/libs/draco/gltf/draco_decoder.js',
  'decoders/draco/draco_decoder.wasm': 'examples/jsm/libs/draco/gltf/draco_decoder.wasm',
  'decoders/draco/draco_wasm_wrapper.js': 'examples/jsm/libs/draco/gltf/draco_wasm_wrapper.js',
};

/**
 * Serves the decoders at /decoders/ in dev and copies them into the build, so compressed
 * models (Draco, meshopt + KTX2 from the AI pipeline) load without a CDN.
 */
function threeDecoders(): Plugin {
  return {
    name: 'three-decoders',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const file = req.url?.split('?')[0]?.replace(/^\//, '');
        const source = file ? DECODERS[file] : undefined;
        if (!file || !source) return next();
        res.setHeader(
          'Content-Type',
          file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
        );
        fs.createReadStream(path.join(threeDir, source)).pipe(res);
      });
    },
    generateBundle() {
      for (const [fileName, source] of Object.entries(DECODERS)) {
        this.emitFile({
          type: 'asset',
          fileName,
          source: fs.readFileSync(path.join(threeDir, source)),
        });
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), viewerFallback(), threeDecoders()],
  envDir: path.resolve(__dirname, '../..'),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        viewer: path.resolve(__dirname, 'viewer.html'),
      },
    },
  },
  server: {
    port: 5173,
    fs: {
      allow: [
        // Allow serving files from the entire monorepo root
        path.resolve(__dirname, '../..'),
      ],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
      },
      '/trpc': {
        target: 'http://localhost:4000',
        changeOrigin: true,
      },
      '/uploads': {
        target: 'http://localhost:4000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:4000',
        ws: true,
      },
    },
  },
});
