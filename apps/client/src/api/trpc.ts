import { createTRPCReact } from '@trpc/react-query';
import { httpBatchLink } from '@trpc/client';

// AppRouter type is imported at the type level only — no server code is bundled
import type { AppRouter } from '../../../../apps/server/src/trpc/router';

export const trpc = createTRPCReact<AppRouter>();

export const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: '/trpc',
      fetch(url, options) {
        return fetch(url, { ...options, credentials: 'include' });
      },
    }),
  ],
});
