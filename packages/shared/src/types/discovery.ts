import type { ExperienceMeta } from './experience';

export interface SearchResult {
  items: ExperienceMeta[];
  totalHits: number;
  query: string;
}

export interface Report {
  id: string;
  reporterId: string;
  reporterName: string;
  targetType: 'experience' | 'comment' | 'user';
  targetId: string;
  targetTitle?: string;
  reason: string;
  status: 'pending' | 'reviewed' | 'resolved' | 'dismissed';
  createdAt: string;
  reviewedAt?: string;
}
