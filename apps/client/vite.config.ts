import { defineConfig } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
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

export default defineConfig({
  plugins: [react(), tailwindcss(), viewerFallback()],
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
