import * as THREE from 'three';

export type RendererType = 'webgpu' | 'webgl';

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  antialias?: boolean;
  preferWebGPU?: boolean;
}

export class Renderer {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private animationFrameId: number | null = null;
  private rendererType: RendererType = 'webgl';
  private resizeObserver: ResizeObserver | null = null;
  private canvas: HTMLCanvasElement;

  constructor(private options: RendererOptions) {
    this.canvas = options.canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    this.camera.position.set(5, 5, 5);
    this.camera.lookAt(0, 0, 0);
  }

  async init(): Promise<RendererType> {
    const preferWebGPU = this.options.preferWebGPU ?? true;

    if (preferWebGPU && 'gpu' in navigator) {
      try {
        // Dynamic import — three/webgpu ships separately and has no type declarations yet
        const { WebGPURenderer } = await import('three/webgpu' as string);
        const gpuRenderer = new WebGPURenderer({
          canvas: this.canvas,
          antialias: this.options.antialias ?? true,
          preserveDrawingBuffer: true,
        });
        await gpuRenderer.init();
        this.renderer = gpuRenderer;
        this.rendererType = 'webgpu';
        console.log('[Forge3D] Renderer: WebGPU');
      } catch {
        console.warn('[Forge3D] WebGPU init failed, falling back to WebGL');
        this.initWebGL();
      }
    } else {
      this.initWebGL();
    }

    this.setupResize();
    this.resize();
    this.addDefaultLights();

    return this.rendererType;
  }

  private initWebGL(): void {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: this.options.antialias ?? true,
      preserveDrawingBuffer: true,
    });
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.rendererType = 'webgl';
    console.log('[Forge3D] Renderer: WebGL');
  }

  private addDefaultLights(): void {
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 10, 5);
    directional.castShadow = true;
    this.scene.add(directional);

    // Ground grid
    const grid = new THREE.GridHelper(20, 20, 0x444466, 0x333355);
    this.scene.add(grid);
  }

  private setupResize(): void {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas.parentElement ?? this.canvas);
  }

  resize(): void {
    const parent = this.canvas.parentElement;
    if (!parent) return;

    const width = parent.clientWidth;
    const height = parent.clientHeight;
    if (width === 0 || height === 0) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  start(onFrame?: (dt: number) => void): void {
    let lastTime = performance.now();

    const loop = (now: number) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      onFrame?.(dt);
      this.renderer.render(this.scene, this.camera);
      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
  }

  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  getRendererType(): RendererType {
    return this.rendererType;
  }

  getThreeRenderer(): THREE.WebGLRenderer {
    return this.renderer;
  }

  dispose(): void {
    this.stop();
    this.resizeObserver?.disconnect();
    this.renderer?.dispose();
  }
}
