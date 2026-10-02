import sharp from 'sharp';
import type {
  GenerationQuality,
  ModelOutput,
  ModelView,
  ReferencesOutput,
  StudioWorkers,
  TexturesOutput,
  ViewsOutput,
  WorkerJobState,
  WorkerKind,
} from '../types';

type Job =
  | { kind: 'references'; readyAt: number; prompt: string; count: number }
  | { kind: 'views'; readyAt: number; noViews: boolean }
  | {
      kind: 'model';
      readyAt: number;
      mode: GenerationQuality;
      seed: number;
      noObject: boolean;
      viewsUsed: number;
    }
  | {
      kind: 'textures';
      readyAt: number;
      seed: number;
      count: number;
      noTextures: boolean;
      /** Whether the judge was asked (AI_TEXTURE_JUDGE=1) */
      judge: boolean;
    };

const MOCK_CREDITS = ['Mock model for development: no AI ran'];
/** Why the mock judge picks the texture it does */
export const MOCK_JUDGE_WHY = 'The mock judge always picks the second new texture: no AI looked.';
/** Ratings like the FLUX worker's, by position, so the second picture is always the best */
const MOCK_RATINGS: { score: number; issues: string[] }[] = [
  { score: 0.64, issues: ['small in the frame'] },
  { score: 0.91, issues: [] },
  { score: 0.37, issues: ['cut off at the bottom', 'more than one object'] },
  { score: 0.55, issues: ['cut off at the top'] },
];
/** Where the multiview worker draws its 6 views from (workers/README.md), all level */
export const MOCK_VIEW_AZIMUTHS = [0, 45, 90, 180, 270, 315];

/**
 * Stand-in GPU workers for development and browser tests (AI_WORKERS_MOCK=1): pictures, views,
 * models and texture options appear after a short delay, with no GPU, network or cost. The
 * pictures are rated like the real worker rates them, with the second always the best, the views
 * are 6 cutouts of a box seen from around it, and the texture options are the final's house in
 * other colours; asked for the judge's pick, they always pick the second new texture (texture 3 in
 * the panel). A prompt containing "fail" fails its pictures, a photo under 256 px gets no
 * texture options, a photo under 128 px gets no views either (the model is then made from the
 * photo alone), and a photo under 64 px fails its model too, so the error and fallback states can
 * be tried. Never enable it in production.
 */
export class MockWorkers implements StudioWorkers {
  readonly name = 'mock';
  readonly prompts = true;
  readonly multiview = true;
  private jobs = new Map<string, Job>();
  private next = 1;

  constructor(private readonly delayMs = 1500) {}

  async startReferences(input: { prompt: string; count: number }): Promise<string> {
    return this.add({ kind: 'references', readyAt: Date.now() + this.delayMs, ...input });
  }

  async references(jobId: string): Promise<WorkerJobState<ReferencesOutput>> {
    const job = this.jobs.get(jobId);
    if (job?.kind !== 'references') return { status: 'failed', message: `Unknown job ${jobId}` };
    if (Date.now() < job.readyAt) return { status: 'running' };
    if (/\bfail\b/i.test(job.prompt)) {
      return { status: 'failed', message: 'The mock workers fail prompts that say "fail"' };
    }
    const base = 1000 + this.next * 10;
    const images = await Promise.all(
      Array.from({ length: job.count }, async (_, i) => {
        const { score, issues } = MOCK_RATINGS[i % MOCK_RATINGS.length]!;
        return {
          file: { data: await mockPicture(job.prompt, i) },
          seed: base + i,
          score,
          ...(issues.length ? { issues: [...issues] } : {}),
        };
      }),
    );
    return { status: 'done', output: { images } };
  }

  async startViews(input: { image: Buffer }): Promise<string> {
    const { width = 0 } = await sharp(input.image).metadata();
    return this.add({
      kind: 'views',
      readyAt: Date.now() + this.delayMs,
      // A small photo stands in for a multiview job that failed, so the fallback can be tried
      noViews: width < 128,
    });
  }

  async views(jobId: string): Promise<WorkerJobState<ViewsOutput>> {
    const job = this.jobs.get(jobId);
    if (job?.kind !== 'views') return { status: 'failed', message: `Unknown job ${jobId}` };
    if (Date.now() < job.readyAt) return { status: 'running' };
    if (job.noViews) {
      return {
        status: 'failed',
        message: 'The mock workers draw no views of a photo under 128 px',
      };
    }
    const views = await Promise.all(
      MOCK_VIEW_AZIMUTHS.map(async (azimuth) => ({
        file: { data: await mockView(azimuth) },
        azimuth,
        elevation: 0,
      })),
    );
    return { status: 'done', output: { views, seconds: 1 } };
  }

  async startModel(input: {
    image: Buffer;
    views?: ModelView[];
    mode: GenerationQuality;
    seed?: number;
  }): Promise<string> {
    const { width = 0 } = await sharp(input.image).metadata();
    return this.add({
      kind: 'model',
      readyAt: Date.now() + this.delayMs * (input.mode === 'final' ? 2 : 1),
      mode: input.mode,
      seed: input.seed ?? 42 + this.next,
      // A tiny photo stands in for "no object found", so the panel's error state can be tried
      noObject: width < 64,
      viewsUsed: input.views?.length ?? 0,
    });
  }

  async model(jobId: string): Promise<WorkerJobState<ModelOutput>> {
    const job = this.jobs.get(jobId);
    if (job?.kind !== 'model') return { status: 'failed', message: `Unknown job ${jobId}` };
    if (Date.now() < job.readyAt) return { status: 'running' };
    if (job.noObject) {
      return {
        status: 'failed',
        message:
          'invalid input: no object found in the image: use one object on a plain background',
      };
    }
    const glb = mockGlb(job.seed, job.mode);
    return {
      status: 'done',
      output: {
        file: { data: glb },
        seed: job.seed,
        triangles: MOCK_TRIANGLES,
        bytes: glb.length,
        seconds: job.mode === 'final' ? 3 : 1.5,
        credits: MOCK_CREDITS,
        // Like a 3D worker that takes views
        viewsUsed: job.viewsUsed,
        pipeline: job.mode === 'final' ? MOCK_FINAL_PIPELINE : '512',
      },
    };
  }

  async startTextures(input: {
    image: Buffer;
    seed: number;
    count: number;
    judge?: { prompt: string };
  }): Promise<string> {
    const { width = 0 } = await sharp(input.image).metadata();
    return this.add({
      kind: 'textures',
      readyAt: Date.now() + this.delayMs * 2,
      seed: input.seed,
      count: input.count,
      // A small photo stands in for a textures job that failed, so the panel's note can be seen
      noTextures: width < 256,
      judge: !!input.judge,
    });
  }

  /** The final's house in other colours: like texture options, the shape stays and the paint changes */
  async textures(jobId: string): Promise<WorkerJobState<TexturesOutput>> {
    const job = this.jobs.get(jobId);
    if (job?.kind !== 'textures') return { status: 'failed', message: `Unknown job ${jobId}` };
    if (Date.now() < job.readyAt) return { status: 'running' };
    if (job.noTextures) {
      return {
        status: 'failed',
        message: 'The mock workers make no texture options for a photo under 256 px',
      };
    }
    const textures = Array.from({ length: job.count }, (_, i) => {
      // The worker's seeds: the final's seed + 1000 for the first, + 2000 for the second, …
      const textureSeed = job.seed + 1000 * (i + 1);
      const glb = mockGlb(textureSeed, 'final');
      return { file: { data: glb }, textureSeed, triangles: MOCK_TRIANGLES, bytes: glb.length };
    });
    // The second new texture (or the only one), so the panel switches the model to it
    const picked = textures[Math.min(1, textures.length - 1)];
    return {
      status: 'done',
      output: {
        textures,
        errors: [],
        pipeline: MOCK_FINAL_PIPELINE,
        seconds: 1.5 * job.count,
        ...(job.judge && picked
          ? { judge: { textureSeed: picked.textureSeed, why: MOCK_JUDGE_WHY } }
          : {}),
      },
    };
  }

  /** Nothing to start: the mock has no cold start. */
  async warm(): Promise<void> {}

  /** Forgets the job, so polling it finds nothing. */
  async cancel(_kind: WorkerKind, jobId: string): Promise<void> {
    this.jobs.delete(jobId);
  }

  private add(job: Job): string {
    const id = `mock-${this.next++}`;
    this.jobs.set(id, job);
    return id;
  }
}

const PALETTE = ['#e8703a', '#3a8ee8', '#48b86a', '#b85cc9'];

/** A 512 px "reference picture": a coloured object on a light background, labelled with the prompt. */
async function mockPicture(prompt: string, index: number): Promise<Buffer> {
  const colour = PALETTE[index % PALETTE.length]!;
  const label = prompt.slice(0, 40).replace(/[<>&"']/g, '');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">
    <rect width="512" height="512" fill="#f2f1ee"/>
    <ellipse cx="256" cy="400" rx="150" ry="22" fill="#d9d6cf"/>
    <rect x="${156 + index * 6}" y="190" width="200" height="200" rx="18" fill="${colour}"/>
    <polygon points="136,200 256,${96 + index * 8} 376,200" fill="${colour}" opacity="0.8"/>
    <text x="256" y="470" font-family="sans-serif" font-size="22" text-anchor="middle" fill="#55524c">${label}</text>
  </svg>`;
  return sharp(Buffer.from(svg)).png().toBuffer();
}

/**
 * A 256 px "view" like the multiview worker's: a PNG cutout (transparent around the object) of a
 * box with a roof, 200 wide and 120 deep, seen from `azimuth` degrees round it, labelled with the
 * angle. Views of the front are orange, of the sides green and of the back blue, so it's easy to
 * tell which is which.
 */
async function mockView(azimuth: number): Promise<Buffer> {
  const angle = (azimuth * Math.PI) / 180;
  const facing = Math.cos(angle);
  const width = Math.round(200 * Math.abs(facing) + 120 * Math.abs(Math.sin(angle)));
  const left = 128 - width / 2;
  const colour = facing > 0.01 ? '#e8703a' : facing < -0.01 ? '#3a8ee8' : '#48b86a';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">
    <rect x="${left}" y="110" width="${width}" height="110" rx="8" fill="${colour}"/>
    <polygon points="${left - 8},116 128,40 ${left + width + 8},116" fill="${colour}" opacity="0.8"/>
    <text x="128" y="180" font-family="sans-serif" font-size="26" text-anchor="middle" fill="#fff">${azimuth}°</text>
  </svg>`;
  return sharp(Buffer.from(svg)).png().toBuffer();
}

const MOCK_TRIANGLES = 14;
/** The pipeline TRELLIS.2's finals report */
const MOCK_FINAL_PIPELINE = '1024_cascade';

/**
 * A small house (box and pyramid roof) as a GLB, coloured by seed. Previews are grey-blue and
 * finals take the seed's colour, so it's easy to see which one a scene shows.
 */
export function mockGlb(seed: number, mode: GenerationQuality): Buffer {
  const hue = (seed * 47) % 360;
  const colour = mode === 'final' ? hslToRgb(hue, 0.55, 0.55) : [0.55, 0.6, 0.7];

  // Flat-shaded triangles, wound outwards: body 0.8 wide and 0.5 high, roof up to 0.9
  const w = 0.4;
  const h = 0.5;
  const c: [number, number, number][] = [
    [-w, 0, -w],
    [w, 0, -w],
    [w, 0, w],
    [-w, 0, w],
    [-w, h, -w],
    [w, h, -w],
    [w, h, w],
    [-w, h, w],
  ];
  const apex: [number, number, number] = [0, 0.9, 0];
  const quads = [
    [0, 1, 2, 3], // bottom
    [3, 2, 6, 7], // front
    [1, 0, 4, 5], // back
    [0, 3, 7, 4], // left
    [2, 1, 5, 6], // right
  ];
  const triangles: [number, number, number][][] = [];
  for (const [a, b, cc, d] of quads) {
    triangles.push([c[a!]!, c[b!]!, c[cc!]!], [c[a!]!, c[cc!]!, c[d!]!]);
  }
  for (const [a, b] of [
    [7, 6],
    [6, 5],
    [5, 4],
    [4, 7],
  ] as const) {
    triangles.push([c[a]!, c[b]!, apex]);
  }

  const positions: number[] = [];
  const normals: number[] = [];
  for (const [p0, p1, p2] of triangles) {
    const n = normal(p0!, p1!, p2!);
    for (const p of [p0!, p1!, p2!]) {
      positions.push(...p);
      normals.push(...n);
    }
  }
  const vertexCount = positions.length / 3;
  const positionBytes = Buffer.from(new Float32Array(positions).buffer);
  const normalBytes = Buffer.from(new Float32Array(normals).buffer);
  const bin = Buffer.concat([positionBytes, normalBytes]);

  const json = {
    asset: { version: '2.0', generator: 'Orainge mock workers' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'Mock model' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] }],
    materials: [
      {
        pbrMetallicRoughness: {
          baseColorFactor: [...colour, 1],
          metallicFactor: 0,
          roughnessFactor: 0.7,
        },
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: vertexCount,
        type: 'VEC3',
        min: [-w, 0, -w],
        max: [w, 0.9, w],
      },
      { bufferView: 1, componentType: 5126, count: vertexCount, type: 'VEC3' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positionBytes.length, target: 34962 },
      {
        buffer: 0,
        byteOffset: positionBytes.length,
        byteLength: normalBytes.length,
        target: 34962,
      },
    ],
    buffers: [{ byteLength: bin.length }],
  };
  return glb(json, bin);
}

function normal(
  a: [number, number, number],
  b: [number, number, number],
  c: [number, number, number],
): [number, number, number] {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n = [
    u[1]! * v[2]! - u[2]! * v[1]!,
    u[2]! * v[0]! - u[0]! * v[2]!,
    u[0]! * v[1]! - u[1]! * v[0]!,
  ];
  const length = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
  return [n[0]! / length, n[1]! / length, n[2]! / length];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}

/** Packs glTF JSON and its binary buffer into a GLB (both chunks padded to 4 bytes). */
function glb(json: object, bin: Buffer): Buffer {
  const pad = (data: Buffer, fill: number) => {
    const extra = (4 - (data.length % 4)) % 4;
    return extra ? Buffer.concat([data, Buffer.alloc(extra, fill)]) : data;
  };
  const jsonChunk = pad(Buffer.from(JSON.stringify(json)), 0x20);
  const binChunk = pad(bin, 0);
  const header = Buffer.alloc(12);
  header.write('glTF', 0, 'ascii');
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + jsonChunk.length + 8 + binChunk.length, 8);
  const chunk = (data: Buffer, type: string) => {
    const head = Buffer.alloc(8);
    head.writeUInt32LE(data.length, 0);
    head.write(type, 4, 'ascii');
    return Buffer.concat([head, data]);
  };
  return Buffer.concat([header, chunk(jsonChunk, 'JSON'), chunk(binChunk, 'BIN\0')]);
}
