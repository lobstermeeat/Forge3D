import { z } from 'zod';
import { publicProcedure, router } from '../trpc';
import { createSession, getSession } from '../../remoteControl/sessions';

export const remoteControlRouter = router({
  /** Create a new remote control session for a published experience. */
  createSession: publicProcedure
    .input(z.object({ experienceSlug: z.string().min(1) }))
    .mutation(({ input }) => {
      const sessionId = createSession(input.experienceSlug);
      return { sessionId };
    }),

  /** Get info about an active remote control session. */
  getSession: publicProcedure
    .input(z.object({ sessionId: z.string().min(1) }))
    .query(({ input }) => {
      const session = getSession(input.sessionId);
      if (!session) {
        return { found: false as const };
      }
      return {
        found: true as const,
        session,
      };
    }),
});
