import Fastify from 'fastify';
import cors from '@fastify/cors';
import { fastifyTRPCPlugin } from '@trpc/server/adapters/fastify';
import { appRouter } from './trpc/router';
import { createContext } from './trpc/context';
import { auth } from './auth';

const PORT = parseInt(process.env['PORT'] ?? '4000', 10);
const APP_URL = process.env['APP_URL'] ?? 'http://localhost:5173';

async function main() {
  const server = Fastify({ logger: true });

  await server.register(cors, {
    origin: APP_URL,
    credentials: true,
  });

  // Better Auth routes
  server.all('/api/auth/*', async (req, reply) => {
    const response = await auth.handler(
      new Request(new URL(req.url, `http://${req.hostname}`), {
        method: req.method,
        headers: req.headers as unknown as Record<string, string>,
        body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
      }),
    );

    reply.status(response.status);
    for (const [key, value] of response.headers.entries()) {
      reply.header(key, value);
    }
    const body = await response.text();
    reply.send(body);
  });

  // tRPC
  await server.register(fastifyTRPCPlugin, {
    prefix: '/trpc',
    trpcOptions: { router: appRouter, createContext },
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
