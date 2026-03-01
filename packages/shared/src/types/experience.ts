import type { SceneData } from './scene';

// ---------- Experience Format Types ----------

export type ExperienceFormatType = 'turntable' | 'animated' | 'video' | 'interactive';

export type ExperienceFormat =
  | { type: 'turntable'; speed: number; axis: 'y' | 'x' }
  | { type: 'animated'; duration: number }
  | { type: 'video'; keyframes: CameraKeyframe[] }
  | { type: 'interactive'; interactions: InteractionDef[] };

export interface CameraKeyframe {
  time: number;
  position: [number, number, number];
  target: [number, number, number];
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
}

export interface InteractionDef {
  entityId: string;
  action: 'showInfo' | 'navigate' | 'playAnimation' | 'openUrl';
  payload: Record<string, unknown>;
}

// ---------- Published Experience Data ----------

export interface ExperienceData {
  version: number;
  scene: SceneData;
  camera: {
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
  };
  environment: {
    backgroundColor: [number, number, number];
    ambientIntensity: number;
  };
  format: ExperienceFormat;
}

// ---------- Experience Status ----------

export type ExperienceStatus = 'draft' | 'processing' | 'published' | 'unpublished' | 'removed';

// ---------- API Response Types ----------

export interface ExperienceMeta {
  id: string;
  title: string;
  description: string | null;
  slug: string;
  status: ExperienceStatus;
  formatType: ExperienceFormatType;
  thumbnailUrl: string | null;
  previewUrl: string | null;
  viewCount: number;
  likeCount: number;
  commentCount: number;
  polyCount: number | null;
  fileSize: number | null;
  creator: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
  };
  categoryId: string | null;
  publishedAt: string | null;
  createdAt: string;
}

export interface ExperienceDetail extends ExperienceMeta {
  sceneUrl: string;
  isLiked?: boolean;
}

export interface CreatorProfile {
  id: string;
  username: string;
  displayName: string;
  bio: string | null;
  websiteUrl: string | null;
  avatarUrl: string | null;
  headerUrl: string | null;
  followerCount: number;
  followingCount: number;
  isFollowing?: boolean;
}

export interface Comment {
  id: string;
  body: string;
  userId: string;
  userName: string;
  userAvatar: string | null;
  parentId: string | null;
  likeCount: number;
  createdAt: string;
  replies?: Comment[];
}

export interface Category {
  id: string;
  name: string;
  slug: string;
}
