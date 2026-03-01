import { createAuthClient } from 'better-auth/react';

export const authClient = createAuthClient({
  baseURL: window.location.origin + '/api/auth',
});

export const useSession = authClient.useSession;

export async function signIn(opts: { provider?: string; email?: string; password?: string; callbackURL?: string }) {
  if (opts.provider) {
    return authClient.signIn.social({ provider: opts.provider as 'github' | 'google', callbackURL: opts.callbackURL });
  }
  return authClient.signIn.email({ email: opts.email!, password: opts.password! });
}

export async function signUp(opts: { email: string; password: string; name: string }) {
  return authClient.signUp.email(opts);
}

export async function signOut() {
  return authClient.signOut();
}
