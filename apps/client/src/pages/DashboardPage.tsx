import { useNavigate } from 'react-router-dom';
import { useSession, signOut } from '@/auth/client';
import { trpc } from '@/api/trpc';

export function DashboardPage() {
  const navigate = useNavigate();
  const { data: session } = useSession();
  const user = session?.user;

  const utils = trpc.useUtils();
  const projectsQuery = trpc.project.list.useQuery();
  const createProject = trpc.project.create.useMutation({
    onSuccess: () => projectsQuery.refetch(),
  });
  const createScene = trpc.scene.create.useMutation();
  const deleteProject = trpc.project.delete.useMutation({
    onSuccess: () => projectsQuery.refetch(),
  });

  const handleNewProject = async () => {
    const project = await createProject.mutateAsync({ name: 'Untitled Project' });
    if (!project) return;
    const scene = await createScene.mutateAsync({ projectId: project.id, name: 'Scene 1' });
    if (!scene) return;
    navigate(`/editor/${project.id}/${scene.id}`);
  };

  const handleOpenProject = async (projectId: string) => {
    const project = await utils.project.getById.fetch({ id: projectId });
    if (project?.scenes?.[0]) {
      navigate(`/editor/${projectId}/${project.scenes[0].id}`);
    }
  };

  const handleDelete = (projectId: string) => {
    if (confirm('Delete this project? This cannot be undone.')) {
      deleteProject.mutate({ id: projectId });
    }
  };

  const handleSignOut = async () => {
    await signOut();
    navigate('/login');
  };

  const projects = projectsQuery.data ?? [];

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#11111b',
        color: '#cdd6f4',
        padding: 32,
      }}
    >
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 32,
        }}
      >
        <h1 style={{ fontSize: 24 }}>Forge3D</h1>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ fontSize: 14, color: '#888' }}>{user?.name ?? user?.email ?? 'User'}</span>
          <button onClick={handleSignOut} style={headerBtn}>
            Sign Out
          </button>
        </div>
      </header>

      <section>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 16,
          }}
        >
          <h2 style={{ fontSize: 18 }}>Projects</h2>
          <button
            onClick={handleNewProject}
            disabled={createProject.isPending}
            style={{
              padding: '8px 20px',
              background: '#2563eb',
              border: 'none',
              borderRadius: 6,
              color: '#fff',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            {createProject.isPending ? 'Creating...' : 'New Project'}
          </button>
        </div>

        {projectsQuery.isLoading && (
          <p style={{ color: '#888', fontSize: 14 }}>Loading projects...</p>
        )}

        {projectsQuery.error && (
          <p style={{ color: '#f38ba8', fontSize: 14 }}>
            Failed to load projects. Is the server running?
          </p>
        )}

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 16,
          }}
        >
          {projects.length === 0 && !projectsQuery.isLoading && (
            <div style={emptyCard}>No projects yet. Create one to get started.</div>
          )}
          {projects.map((project) => (
            <div key={project.id} style={projectCard}>
              <div
                onClick={() => handleOpenProject(project.id)}
                style={{ cursor: 'pointer', flex: 1 }}
              >
                <h3 style={{ fontSize: 16, marginBottom: 4 }}>{project.name}</h3>
                {project.description && (
                  <p style={{ fontSize: 13, color: '#888', marginBottom: 8 }}>
                    {project.description}
                  </p>
                )}
                <p style={{ fontSize: 11, color: '#666' }}>
                  Updated {new Date(project.updatedAt).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(project.id);
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#f38ba8',
                  cursor: 'pointer',
                  fontSize: 12,
                  padding: '4px 8px',
                  alignSelf: 'flex-start',
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

const headerBtn: React.CSSProperties = {
  padding: '6px 16px',
  background: '#313244',
  border: 'none',
  borderRadius: 6,
  color: '#cdd6f4',
  cursor: 'pointer',
  fontSize: 13,
};

const emptyCard: React.CSSProperties = {
  background: '#1e1e2e',
  borderRadius: 12,
  border: '1px solid #313244',
  padding: 24,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 200,
  color: '#666',
  fontSize: 14,
};

const projectCard: React.CSSProperties = {
  background: '#1e1e2e',
  borderRadius: 12,
  border: '1px solid #313244',
  padding: 20,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'flex-start',
};
