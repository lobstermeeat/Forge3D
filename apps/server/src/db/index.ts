import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

const connectionString =
  process.env['DATABASE_URL'] ?? 'postgresql://forge3d:forge3d_dev@localhost:5432/forge3d';

const client = postgres(connectionString);
export const db = drizzle(client, { schema });

export type Database = typeof db;
export { schema };
