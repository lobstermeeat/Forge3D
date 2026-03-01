/**
 * Mock experience data for development without a running backend.
 * Thumbnails are inline SVG data URIs — lightweight colored gradients.
 */
import type { ExperienceMeta, ExperienceData, ExperienceFormat } from '@forge3d/shared';

// ---------------------------------------------------------------------------
// SVG placeholder thumbnails (16:9, 640x360)
// ---------------------------------------------------------------------------

function svgThumb(bg1: string, bg2: string, accent: string, label: string): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="${bg1}"/>
        <stop offset="100%" stop-color="${bg2}"/>
      </linearGradient>
    </defs>
    <rect width="640" height="360" fill="url(#bg)"/>
    <circle cx="320" cy="150" r="60" fill="${accent}" opacity="0.35"/>
    <rect x="240" y="200" width="160" height="80" rx="8" fill="${accent}" opacity="0.2"/>
    <line x1="100" y1="320" x2="540" y2="320" stroke="${accent}" stroke-width="1" opacity="0.3"/>
    <text x="320" y="345" text-anchor="middle" fill="${accent}" font-family="sans-serif" font-size="12" opacity="0.6">${label}</text>
  </svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

const thumbs = {
  turntable: svgThumb('#1a1a2e', '#16213e', '#89b4fa', 'Turntable'),
  video:     svgThumb('#1a1a2e', '#0f3460', '#f38ba8', 'Video'),
  animated:  svgThumb('#1a1a2e', '#1b2838', '#a6e3a1', 'Animated'),
  interactive: svgThumb('#1a1a2e', '#2d1b69', '#fab387', 'Interactive'),
  robot:     svgThumb('#0d1117', '#1e293b', '#74c7ec', 'Robot'),
  sword:     svgThumb('#1c1917', '#292524', '#f9e2af', 'Sword'),
  forest:    svgThumb('#052e16', '#14532d', '#a6e3a1', 'Forest'),
  city:      svgThumb('#1e1b4b', '#312e81', '#cba6f7', 'City'),
};

// ---------------------------------------------------------------------------
// Mock creators
// ---------------------------------------------------------------------------

const creators = {
  alice: { id: 'u-001', username: 'alice_3d', displayName: 'Alice Chen', avatarUrl: null },
  bob:   { id: 'u-002', username: 'bob_art', displayName: 'Bob Kowalski', avatarUrl: null },
  chara: { id: 'u-003', username: 'chara_dev', displayName: 'Chara Tanaka', avatarUrl: null },
  dave:  { id: 'u-004', username: 'dave_fx', displayName: 'Dave Müller', avatarUrl: null },
};

// ---------------------------------------------------------------------------
// Mock experiences (ExperienceMeta)
// ---------------------------------------------------------------------------

export const MOCK_EXPERIENCES: ExperienceMeta[] = [
  {
    id: 'exp-001',
    title: 'Neon Mech Turntable',
    description: 'A glowing cyberpunk mech slowly rotating under studio lights.',
    slug: 'neon-mech-turntable',
    status: 'published',
    formatType: 'turntable',
    thumbnailUrl: thumbs.robot,
    previewUrl: thumbs.robot,
    viewCount: 1284,
    likeCount: 92,
    commentCount: 7,
    polyCount: 24500,
    fileSize: 184320,
    creator: creators.alice,
    categoryId: 'cat-scifi',
    publishedAt: '2026-02-20T14:30:00.000Z',
    createdAt: '2026-02-19T10:00:00.000Z',
  },
  {
    id: 'exp-002',
    title: 'Crystal Sword Flythrough',
    description: 'Camera sweeps around an enchanted blade with ice particle effects.',
    slug: 'crystal-sword-flythrough',
    status: 'published',
    formatType: 'video',
    thumbnailUrl: thumbs.sword,
    previewUrl: thumbs.sword,
    viewCount: 876,
    likeCount: 64,
    commentCount: 3,
    polyCount: 12000,
    fileSize: 95200,
    creator: creators.bob,
    categoryId: 'cat-fantasy',
    publishedAt: '2026-02-22T09:15:00.000Z',
    createdAt: '2026-02-21T18:00:00.000Z',
  },
  {
    id: 'exp-003',
    title: 'Enchanted Forest Walk',
    description: 'An animated scene with swaying trees, fireflies, and a flowing stream.',
    slug: 'enchanted-forest-walk',
    status: 'published',
    formatType: 'animated',
    thumbnailUrl: thumbs.forest,
    previewUrl: thumbs.forest,
    viewCount: 3421,
    likeCount: 210,
    commentCount: 18,
    polyCount: 45000,
    fileSize: 512000,
    creator: creators.chara,
    categoryId: 'cat-nature',
    publishedAt: '2026-02-25T12:00:00.000Z',
    createdAt: '2026-02-24T08:30:00.000Z',
  },
  {
    id: 'exp-004',
    title: 'Cyberpunk Apartment Tour',
    description: 'Click on furniture and gadgets to learn about each piece.',
    slug: 'cyberpunk-apartment-tour',
    status: 'published',
    formatType: 'interactive',
    thumbnailUrl: thumbs.city,
    previewUrl: thumbs.city,
    viewCount: 2154,
    likeCount: 145,
    commentCount: 22,
    polyCount: 67000,
    fileSize: 1024000,
    creator: creators.dave,
    categoryId: 'cat-architecture',
    publishedAt: '2026-02-26T16:45:00.000Z',
    createdAt: '2026-02-25T20:00:00.000Z',
  },
  {
    id: 'exp-005',
    title: 'Lowpoly Fox 360',
    description: 'Cute lowpoly fox model with pastel colours.',
    slug: 'lowpoly-fox-360',
    status: 'published',
    formatType: 'turntable',
    thumbnailUrl: thumbs.turntable,
    previewUrl: thumbs.turntable,
    viewCount: 543,
    likeCount: 38,
    commentCount: 2,
    polyCount: 800,
    fileSize: 32000,
    creator: creators.alice,
    categoryId: 'cat-characters',
    publishedAt: '2026-02-27T11:00:00.000Z',
    createdAt: '2026-02-26T22:00:00.000Z',
  },
  {
    id: 'exp-006',
    title: 'Space Station Camera Path',
    description: 'Cinematic flythrough of a modular space station interior.',
    slug: 'space-station-path',
    status: 'published',
    formatType: 'video',
    thumbnailUrl: thumbs.video,
    previewUrl: thumbs.video,
    viewCount: 1899,
    likeCount: 120,
    commentCount: 9,
    polyCount: 55000,
    fileSize: 768000,
    creator: creators.bob,
    categoryId: 'cat-scifi',
    publishedAt: '2026-02-28T08:00:00.000Z',
    createdAt: '2026-02-27T15:30:00.000Z',
  },
  {
    id: 'exp-007',
    title: 'Wind-Up Toy Animation',
    description: 'A mechanical toy bot walks across a tabletop.',
    slug: 'wind-up-toy-anim',
    status: 'published',
    formatType: 'animated',
    thumbnailUrl: thumbs.animated,
    previewUrl: thumbs.animated,
    viewCount: 672,
    likeCount: 51,
    commentCount: 4,
    polyCount: 3200,
    fileSize: 48000,
    creator: creators.chara,
    categoryId: 'cat-characters',
    publishedAt: '2026-02-28T14:00:00.000Z',
    createdAt: '2026-02-28T10:00:00.000Z',
  },
  {
    id: 'exp-008',
    title: 'Museum Gallery Hotspots',
    description: 'Interactive exhibit — click paintings to reveal descriptions.',
    slug: 'museum-gallery-hotspots',
    status: 'published',
    formatType: 'interactive',
    thumbnailUrl: thumbs.interactive,
    previewUrl: thumbs.interactive,
    viewCount: 987,
    likeCount: 76,
    commentCount: 11,
    polyCount: 28000,
    fileSize: 256000,
    creator: creators.dave,
    categoryId: 'cat-architecture',
    publishedAt: '2026-03-01T10:00:00.000Z',
    createdAt: '2026-02-28T21:00:00.000Z',
  },
];

// ---------------------------------------------------------------------------
// Mock full ExperienceData for viewer testing
// ---------------------------------------------------------------------------

const SCENE_STUB = {
  entities: [
    {
      id: 'e-cube',
      name: 'Cube',
      transform: {
        position: [0, 0.5, 0] as [number, number, number],
        rotation: [0, 0, 0, 1] as [number, number, number, number],
        scale: [1, 1, 1] as [number, number, number],
      },
      components: {
        meshRenderer: { geometryType: 'box' as const, materialIndex: 0 },
      },
    },
  ],
  materials: [{ color: [0.54, 0.7, 0.98] as [number, number, number], metalness: 0, roughness: 0.5 }],
  metadata: { name: 'Mock Scene' },
};

const FORMAT_CONFIGS: Record<string, ExperienceFormat> = {
  turntable: { type: 'turntable', speed: 1, axis: 'y' },
  video: {
    type: 'video',
    keyframes: [
      { time: 0,   position: [5, 5, 5],     target: [0, 0, 0], easing: 'ease-in-out' },
      { time: 2,   position: [-3, 4, 5],    target: [0, 0.5, 0], easing: 'ease-in-out' },
      { time: 4,   position: [-5, 2, -3],   target: [0, 0, 0], easing: 'ease-out' },
      { time: 5.5, position: [5, 5, 5],     target: [0, 0, 0], easing: 'ease-in' },
    ],
  },
  animated: { type: 'animated', duration: 5 },
  interactive: {
    type: 'interactive',
    interactions: [
      { entityId: 'e-cube', action: 'showInfo', payload: { title: 'Cube', description: 'A simple box mesh.' } },
    ],
  },
};

export function getMockExperienceData(formatType: string): ExperienceData {
  return {
    version: 1,
    scene: SCENE_STUB as unknown as ExperienceData['scene'],
    camera: { position: [5, 5, 5], target: [0, 0, 0], fov: 60 },
    environment: { backgroundColor: [0.067, 0.067, 0.106], ambientIntensity: 0.4 },
    format: FORMAT_CONFIGS[formatType] ?? FORMAT_CONFIGS.turntable!,
  };
}

// ---------------------------------------------------------------------------
// Helper: check if we should use mocks (no backend available)
// ---------------------------------------------------------------------------

export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === 'true';
