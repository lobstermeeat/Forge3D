import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { trpc, trpcClient } from '@/api/trpc';
import { ViewerPage } from './ViewerPage';
import { EmbedPage } from './EmbedPage';
import { RemotePage } from './RemotePage';

export function ViewerApp() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: false, staleTime: 30_000 },
        },
      }),
  );

  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/view/:slug" element={<ViewerPage />} />
            <Route path="/embed/:slug" element={<EmbedPage />} />
            <Route path="/remote/:sessionId" element={<RemotePage />} />
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </trpc.Provider>
  );
}
