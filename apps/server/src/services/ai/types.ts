export interface GenerationRequest {
  type: 'text-to-3d' | 'image-to-3d';
  prompt?: string;
  imageUrl?: string;
  userId: string;
}

export interface GenerationResult {
  modelUrl: string;
  format: 'glb' | 'gltf' | 'obj' | 'fbx';
  thumbnailUrl?: string;
  durationMs: number;
}

export interface GenerationProgress {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number; // 0-100
  message?: string;
}

export interface AIProvider {
  readonly name: string;
  readonly supportedTypes: readonly ('text-to-3d' | 'image-to-3d')[];

  generate(request: GenerationRequest): Promise<string>; // returns job ID
  pollStatus(jobId: string): Promise<GenerationProgress>;
  getResult(jobId: string): Promise<GenerationResult>;
}
