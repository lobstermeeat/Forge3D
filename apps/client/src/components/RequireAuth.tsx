import { Navigate } from 'react-router-dom';
import { useSession } from '@/auth/client';

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { data: session, isPending, error } = useSession();

  if (isPending) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          background: '#11111b',
          color: '#888',
          fontSize: 14,
        }}
      >
        Loading...
      </div>
    );
  }

  if (error || !session?.user) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
