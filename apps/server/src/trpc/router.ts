import { router } from './trpc';
import { projectRouter } from './routers/project';
import { sceneRouter } from './routers/scene';

export const appRouter = router({
  project: projectRouter,
  scene: sceneRouter,
});

export type AppRouter = typeof appRouter;
