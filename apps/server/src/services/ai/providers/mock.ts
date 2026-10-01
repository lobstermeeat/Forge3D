import sharp from 'sharp';
import type {
  GenerationQuality,
  ModelOutput,
  ReferencesOutput,
  StudioWorkers,
  WorkerJobState,
} from '../types';

type Job =
  | { kind: 'references'; readyAt: number; prompt: string; count: number }
  | { kind: 'model'; readyAt: number; mode: GenerationQuality; seed: number; noObject: boolean };

const MOCK_CREDITS = ['Mock model for development: no AI ran'];
/** Ratings like the FLUX worker's, by position, so the second picture is always the best */
const MOCK_RATINGS: { score: number; issues: string[] }[] = [
  { score: 0.64, issues: ['small in the frame'] },
  { score: 0.91, issues: [] },
  { score: 0.37, issues: ['cut off at the bottom', 'more than one object'] },
  { score: 0.55, issues: ['cut off at the top'] },
];

/**
 * Stand-in GPU workers for development and browser tests (AI_WORKERS_MOCK=1): pictures and
 * models appear after a short delay, with no GPU, network or cost. The pictures are rated like
 * the real worker rates them, with the second always the best. A prompt containing "fail" fails
 * its pictures, and a photo run fails when the photo is very small, so error states can be tried
 * too. Never enable it in production.
 */
export class MockWorkers implements StudioWorkers {
  readonly name = 'mock';
  readonly prompts = true;
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

  async startModel(input: {
    image: Buffer;
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
      },
    };
  }

  /** Nothing to start: the mock has no cold start. */
  async warm(): Promise<void> {}

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

const MOCK_TRIANGLES = 14;

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
