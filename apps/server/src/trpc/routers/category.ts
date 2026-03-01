import { eq, asc } from 'drizzle-orm';
import { router, publicProcedure, adminProcedure } from '../trpc';
import { schema } from '../../db';

const DEFAULT_CATEGORIES = [
  { name: 'Characters', slug: 'characters', sortOrder: 1 },
  { name: 'Vehicles', slug: 'vehicles', sortOrder: 2 },
  { name: 'Architecture', slug: 'architecture', sortOrder: 3 },
  { name: 'Environment', slug: 'environment', sortOrder: 4 },
  { name: 'Props', slug: 'props', sortOrder: 5 },
  { name: 'Weapons', slug: 'weapons', sortOrder: 6 },
  { name: 'Furniture', slug: 'furniture', sortOrder: 7 },
  { name: 'Animals', slug: 'animals', sortOrder: 8 },
  { name: 'Abstract', slug: 'abstract', sortOrder: 9 },
  { name: 'Other', slug: 'other', sortOrder: 10 },
];

export const categoryRouter = router({
  list: publicProcedure.query(async ({ ctx }) => {
    return ctx.db
      .select({
        id: schema.categories.id,
        name: schema.categories.name,
        slug: schema.categories.slug,
      })
      .from(schema.categories)
      .orderBy(asc(schema.categories.sortOrder));
  }),

  seed: adminProcedure.mutation(async ({ ctx }) => {
    const existing = await ctx.db
      .select({ id: schema.categories.id })
      .from(schema.categories)
      .limit(1);

    if (existing.length > 0) {
      return { seeded: false, message: 'Categories already exist' };
    }

    await ctx.db.insert(schema.categories).values(DEFAULT_CATEGORIES);
    return { seeded: true, message: `Seeded ${DEFAULT_CATEGORIES.length} categories` };
  }),
});
