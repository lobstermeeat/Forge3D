import { initTRPC, TRPCError } from '@trpc/server';
import { eq } from 'drizzle-orm';
import type { Context } from './context';
import { schema } from '../db';

const t = initTRPC.context<Context>().create();

export const router = t.router;
export const publicProcedure = t.procedure;

export const protectedProcedure = t.procedure.use(async ({ ctx, next }) => {
  if (!ctx.user) {
    throw new TRPCError({ code: 'UNAUTHORIZED', message: 'Not authenticated' });
  }
  return next({
    ctx: {
      ...ctx,
      user: ctx.user,
    },
  });
});

export const adminProcedure = protectedProcedure.use(async ({ ctx, next }) => {
  const [user] = await ctx.db
    .select({ role: schema.users.role })
    .from(schema.users)
    .where(eq(schema.users.id, ctx.user.id));

  if (!user || (user.role !== 'admin' && user.role !== 'moderator')) {
    throw new TRPCError({ code: 'FORBIDDEN', message: 'Admin access required' });
  }

  return next({
    ctx: { ...ctx, role: user.role },
  });
});
