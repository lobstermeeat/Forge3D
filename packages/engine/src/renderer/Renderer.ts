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
  private backendLabel = 'WebGL 2';
  private resizeObserver: ResizeObserver | null = null;
  private canvas: HTMLCanvasElement;
  /** Editor-only helpers (grid, sky). Never serialized or published. */
  private grid: THREE.Group | null = null;
  private sky: THREE.Mesh | null = null;

  constructor(private options: RendererOptions) {
    this.canvas = options.canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1b1e24);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    this.camera.position.set(6, 5, 8);
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
        // WebGPURenderer silently falls back to a WebGL 2 backend when no adapter is available
        const onWebGPU = Boolean(
          (gpuRenderer as { backend?: { isWebGPUBackend?: boolean } }).backend?.isWebGPUBackend,
        );
        this.backendLabel = onWebGPU ? 'WebGPU' : 'WebGL 2 (WebGPU fallback)';
        console.log(`[Forge3D] Renderer: ${this.backendLabel}`);
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
    this.addEditorHelpers();

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
    this.backendLabel = 'WebGL 2';
    console.log('[Forge3D] Renderer: WebGL');
  }

  private addDefaultLights(): void {
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 10, 5);
    directional.castShadow = true;
    this.scene.add(directional);
  }

  /** Sky dome and a two-level ground grid, like an editor baseplate. */
  private addEditorHelpers(): void {
    // Sky: vertex-coloured dome (works on both WebGL and WebGPU backends)
    const skyGeo = new THREE.SphereGeometry(400, 32, 16);
    const zenith = new THREE.Color(0x283246);
    const horizon = new THREE.Color(0x5b6478);
    const ground = new THREE.Color(0x1d2026);
    const colors: number[] = [];
    const pos = skyGeo.attributes.position!;
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const h = pos.getY(i) / 400;
      if (h >= 0) c.copy(horizon).lerp(zenith, Math.min(1, Math.pow(h, 0.6)));
      else c.copy(horizon).lerp(ground, Math.min(1, -h * 6));
      colors.push(c.r, c.g, c.b);
    }
    skyGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.sky = new THREE.Mesh(
      skyGeo,
      new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.BackSide,
        depthWrite: false,
        fog: false,
      }),
    );
    this.sky.name = '__editor_sky';
    this.sky.renderOrder = -1;
    this.sky.raycast = () => {};
    this.scene.add(this.sky);

    // Grid: fine 1-unit cells plus stronger 10-unit lines
    const grid = new THREE.Group();
    grid.name = '__editor_grid';
    const fine = Renderer.gridLines(100, 1, 0x8a93a6, 0.16);
    const major = Renderer.gridLines(100, 10, 0xc3c9d4, 0.3);
    major.position.y = 0.002;
    grid.add(fine, major);
    this.grid = grid;
    this.scene.add(grid);
  }

  /**
   * Grid lines built from 1-unit segments. Long single lines that pass behind the
   * camera get dropped by some rasterizers (e.g. software GL); short ones clip cleanly.
   */
  private static gridLines(
    size: number,
    step: number,
    color: number,
    opacity: number,
  ): THREE.LineSegments {
    const half = size / 2;
    const pts: number[] = [];
    for (let i = -half; i <= half + 1e-6; i += step) {
      for (let j = -half; j < half - 1e-6; j += 1) {
        pts.push(i, 0, j, i, 0, j + 1); // along Z
        pts.push(j, 0, i, j + 1, 0, i); // along X
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const lines = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }),
    );
    lines.raycast = () => {};
    return lines;
  }

  setGridVisible(visible: boolean): void {
    if (this.grid) this.grid.visible = visible;
  }

  /** Human-readable name of the backend actually drawing frames. */
  getBackendLabel(): string {
    return this.backendLabel;
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
