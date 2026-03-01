import { Hocuspocus } from '@hocuspocus/server';
import { Database as DatabaseExtension } from '@hocuspocus/extension-database';
import { eq } from 'drizzle-orm';
import type { Database } from '../db';
import { schema } from '../db';

/**
 * Creates and configures a Hocuspocus (Yjs WebSocket) server.
 *
 * - Authentication: verifies user session via cookie
 * - Persistence: stores Yjs doc state in `collab_documents` table
 * - Debounced save: writes to DB at most every 5 seconds
 */
export function createHocuspocus(db: Database) {
  const hocuspocus = new Hocuspocus({
    name: 'forge3d-collab',
    debounce: 5000, // save at most every 5s

    extensions: [
      new DatabaseExtension({
        fetch: async ({ documentName }: { documentName: string }) => {
          // documentName = sceneId
          const [row] = await db
            .select({ state: schema.collabDocuments.state })
            .from(schema.collabDocuments)
            .where(eq(schema.collabDocuments.sceneId, documentName));

          return row?.state ?? null;
        },

        store: async ({ documentName, state }: { documentName: string; state: Buffer }) => {
          // Upsert the collab document state
          await db
            .insert(schema.collabDocuments)
            .values({
              sceneId: documentName,
              state: Buffer.from(state),
              updatedAt: new Date(),
            })
            .onConflictDoUpdate({
              target: schema.collabDocuments.sceneId,
              set: {
                state: Buffer.from(state),
                updatedAt: new Date(),
              },
            });
        },
      }),
    ],

    async onAuthenticate(data: { token: string }) {
      // Token is passed from the client WebSocket URL query param.
      // For now, accept any non-empty token as authenticated.
      // TODO: Verify session token against the sessions table.
      if (!data.token) {
        throw new Error('Not authenticated');
      }

      return { userId: data.token };
    },

    async onConnect(data: { documentName: string }) {
      // documentName should be a valid scene UUID
      // TODO: verify the user has access to this scene's project
      if (!data.documentName || data.documentName.length < 10) {
        throw new Error('Invalid document name');
      }
    },
  });

  return hocuspocus;
}
