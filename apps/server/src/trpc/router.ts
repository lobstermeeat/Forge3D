import { router } from './trpc';
import { projectRouter } from './routers/project';
import { sceneRouter } from './routers/scene';
import { experienceRouter } from './routers/experience';
import { profileRouter } from './routers/profile';
import { socialRouter } from './routers/social';
import { interactionRouter } from './routers/interaction';
import { categoryRouter } from './routers/category';
import { searchRouter } from './routers/search';
import { moderationRouter } from './routers/moderation';

export const appRouter = router({
  project: projectRouter,
  scene: sceneRouter,
  experience: experienceRouter,
  profile: profileRouter,
  social: socialRouter,
  interaction: interactionRouter,
  category: categoryRouter,
  search: searchRouter,
  moderation: moderationRouter,
});

export type AppRouter = typeof appRouter;
