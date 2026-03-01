import { useState, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { trpc, trpcClient } from '@/api/trpc';
import { LoginPage } from '@/pages/LoginPage';
import { DashboardPage } from '@/pages/DashboardPage';
import { EditorPage } from '@/pages/EditorPage';
import { FeedPage } from '@/pages/FeedPage';
import { ProfilePage } from '@/pages/ProfilePage';
import { EditProfilePage } from '@/pages/EditProfilePage';
import { SearchPage } from '@/pages/SearchPage';
import { AdminPage } from '@/pages/AdminPage';
import { RequireAuth } from '@/components/RequireAuth';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('React ErrorBoundary:', error, info); }
  render() {
    if (this.state.error) {
      return (
        <pre style={{ color: '#f38ba8', padding: 24, fontSize: 14, whiteSpace: 'pre-wrap', background: '#11111b', minHeight: '100vh' }}>
          React Error: {this.state.error.message}{'\n\n'}Stack:{'\n'}{this.state.error.stack}
        </pre>
      );
    }
    return this.props.children;
  }
}

export function App() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: false, staleTime: 30_000 },
        },
      }),
  );

  return (
    <ErrorBoundary>
      <trpc.Provider client={trpcClient} queryClient={queryClient}>
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                path="/dashboard"
                element={
                  <RequireAuth>
                    <DashboardPage />
                  </RequireAuth>
                }
              />
              <Route path="/editor" element={<EditorPage />} />
              <Route path="/editor/:projectId/:sceneId" element={<EditorPage />} />
              <Route path="/feed" element={<FeedPage />} />
              <Route path="/explore" element={<FeedPage />} />
              <Route path="/creator/:username" element={<ProfilePage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route
                path="/settings/profile"
                element={
                  <RequireAuth>
                    <EditProfilePage />
                  </RequireAuth>
                }
              />
              <Route
                path="/admin"
                element={
                  <RequireAuth>
                    <AdminPage />
                  </RequireAuth>
                }
              />
              <Route path="*" element={<Navigate to="/explore" replace />} />
            </Routes>
          </BrowserRouter>
        </QueryClientProvider>
      </trpc.Provider>
    </ErrorBoundary>
  );
}
