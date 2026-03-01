import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { trpc } from '@/api/trpc';

export function EditProfilePage() {
  const profileQuery = trpc.profile.me.useQuery();
  const updateMutation = trpc.profile.update.useMutation({
    onSuccess: () => {
      setSaveStatus('saved');
      profileQuery.refetch();
    },
    onError: (err) => setSaveStatus(err.message),
  });
  const usernameMutation = trpc.profile.setupUsername.useMutation({
    onSuccess: () => {
      setSaveStatus('saved');
      profileQuery.refetch();
    },
    onError: (err) => setSaveStatus(err.message),
  });

  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [bio, setBio] = useState('');
  const [websiteUrl, setWebsiteUrl] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // Populate form when profile loads
  useEffect(() => {
    if (profileQuery.data) {
      setUsername(profileQuery.data.username ?? '');
      setDisplayName(profileQuery.data.displayName ?? '');
      setBio(profileQuery.data.bio ?? '');
      setWebsiteUrl(profileQuery.data.websiteUrl ?? '');
      setAvatarUrl(profileQuery.data.avatarUrl ?? '');
    }
  }, [profileQuery.data]);

  const handleSaveProfile = () => {
    setSaveStatus(null);
    updateMutation.mutate({
      displayName: displayName || undefined,
      bio: bio || undefined,
      websiteUrl: websiteUrl || '',
      avatarUrl: avatarUrl || '',
    });
  };

  const handleSetUsername = () => {
    if (!username.trim()) return;
    setSaveStatus(null);
    usernameMutation.mutate({ username: username.trim().toLowerCase() });
  };

  if (profileQuery.isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#11111b]">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* Header */}
      <header className="border-b border-[#313244] bg-[#1e1e2e]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <Link to="/explore" className="text-lg font-bold text-[#cdd6f4]">
            Forge3D
          </Link>
          <Link
            to="/dashboard"
            className="rounded-md px-3 py-1.5 text-sm text-[#a6adc8] hover:bg-[#313244] hover:text-[#cdd6f4]"
          >
            Dashboard
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-2xl px-4 py-8">
        <h1 className="mb-6 text-xl font-bold text-[#cdd6f4]">Edit Profile</h1>

        {/* Username section */}
        <div className="mb-6 rounded-lg bg-[#1e1e2e] p-6">
          <label className="mb-1 block text-sm font-medium text-[#a6adc8]">Username</label>
          <p className="mb-3 text-xs text-[#6c7086]">
            Your unique URL: forge3d.com/creator/{username || '...'}
          </p>
          <div className="flex gap-2">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ''))}
              placeholder="my-username"
              maxLength={30}
              className="flex-1 rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
            />
            <button
              onClick={handleSetUsername}
              disabled={usernameMutation.isPending || !username.trim()}
              className="rounded-md bg-[#89b4fa] px-4 py-2 text-sm font-medium text-[#11111b] hover:bg-[#74c7ec] disabled:opacity-50"
            >
              {usernameMutation.isPending ? 'Saving...' : 'Set Username'}
            </button>
          </div>
        </div>

        {/* Profile details */}
        <div className="rounded-lg bg-[#1e1e2e] p-6">
          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium text-[#a6adc8]">Display Name</label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Your display name"
                maxLength={100}
                className="w-full rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-[#a6adc8]">Bio</label>
              <textarea
                value={bio}
                onChange={(e) => setBio(e.target.value)}
                placeholder="Tell people about yourself"
                maxLength={500}
                rows={3}
                className="w-full resize-none rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
              />
              <p className="mt-1 text-right text-xs text-[#6c7086]">{bio.length}/500</p>
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-[#a6adc8]">Website</label>
              <input
                type="url"
                value={websiteUrl}
                onChange={(e) => setWebsiteUrl(e.target.value)}
                placeholder="https://example.com"
                className="w-full rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-[#a6adc8]">Avatar URL</label>
              <input
                type="url"
                value={avatarUrl}
                onChange={(e) => setAvatarUrl(e.target.value)}
                placeholder="https://example.com/avatar.jpg"
                className="w-full rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
              />
            </div>

            <div className="flex items-center justify-between pt-2">
              <div>
                {saveStatus === 'saved' && (
                  <p className="text-sm text-[#a6e3a1]">Profile saved!</p>
                )}
                {saveStatus && saveStatus !== 'saved' && (
                  <p className="text-sm text-[#f38ba8]">{saveStatus}</p>
                )}
              </div>
              <button
                onClick={handleSaveProfile}
                disabled={updateMutation.isPending}
                className="rounded-md bg-[#89b4fa] px-6 py-2 text-sm font-medium text-[#11111b] hover:bg-[#74c7ec] disabled:opacity-50"
              >
                {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
