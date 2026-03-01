import type { CreateFastifyContextOptions } from '@trpc/server/adapters/fastify';
import { db } from '../db';
import { auth } from '../auth';

export async function createContext({ req, res }: CreateFastifyContextOptions) {
  const session = await auth.api.getSession({
    headers: req.headers as unknown as Headers,
  });

  return {
    db,
    session,
    user: session?.user ?? null,
  };
}

export type Context = Awaited<ReturnType<typeof createContext>>;
