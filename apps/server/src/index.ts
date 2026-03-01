import path from 'node:path';
import Fastify from 'fastify';
import cors from '@fastify/cors';
import fastifyStatic from '@fastify/static';
import websocket from '@fastify/websocket';
import { fastifyTRPCPlugin } from '@trpc/server/adapters/fastify';
import { appRouter } from './trpc/router';
import { createContext } from './trpc/context';
import { toNodeHandler } from 'better-auth/node';
import { auth } from './auth';
import { db } from './db';
import { createHocuspocus } from './collab/hocuspocus';

const PORT = parseInt(process.env['PORT'] ?? '4000', 10);
const APP_URL = process.env['APP_URL'] ?? 'http://localhost:5173';
const UPLOAD_DIR = process.env['UPLOAD_DIR'] ?? './uploads';

async function main() {
  const server = Fastify({ logger: true });

  await server.register(cors, {
    origin: APP_URL,
    credentials: true,
  });

  // Better Auth routes — encapsulated with raw body parser to avoid Fastify JSON conflicts
  const authHandler = toNodeHandler(auth);

  await server.register(async (scope) => {
    scope.removeAllContentTypeParsers();
    scope.addContentTypeParser('*', (_req, _payload, done) => {
      done(null, undefined);
    });

    scope.all('/api/auth/*', async (req, reply) => {
      await authHandler(req.raw, reply.raw);
      reply.hijack();
    });
  });

  // Serve uploaded files (experiences, thumbnails, etc.)
  await server.register(fastifyStatic, {
    root: path.resolve(UPLOAD_DIR),
    prefix: '/uploads/',
    decorateReply: false,
  });

  // tRPC
  await server.register(fastifyTRPCPlugin, {
    prefix: '/trpc',
    trpcOptions: { router: appRouter, createContext },
  });

  // WebSocket for real-time collaboration (Hocuspocus / Yjs)
  await server.register(websocket);
  const hocuspocus = createHocuspocus(db);

  server.get('/ws/collab/:sceneId', { websocket: true }, (socket, req) => {
    const sceneId = (req.params as { sceneId: string }).sceneId;
    hocuspocus.handleConnection(socket, req.raw, sceneId);
  });

  // Health check
  server.get('/health', async () => ({ status: 'ok', timestamp: new Date().toISOString() }));

  await server.listen({ port: PORT, host: '0.0.0.0' });
  console.log(`[Forge3D] Server running on http://localhost:${PORT}`);
}

main().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
