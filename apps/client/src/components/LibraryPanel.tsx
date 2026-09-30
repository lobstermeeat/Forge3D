import { useEffect, useState } from 'react';
import * as THREE from 'three';
import { MeshFactory } from '@forge3d/engine';
import type { LightData } from '@forge3d/shared';
import { Icon } from '@/editor/Icon';
import { PRIMITIVES, restingTransform, type Primitive } from '@/editor/placement';
import { useAssetStore } from '@/stores/assetStore';
import { useEditorStore } from '@/stores/editorStore';
import type { EditorActions } from '@/hooks/useEditorActions';

/** MIME type used when dragging library items into the viewport. */
export const LIBRARY_DRAG_TYPE = 'application/x-forge3d-library';

export type LibraryDragPayload =
  | { kind: 'part'; type: Primitive }
  | { kind: 'light'; type: LightData['type'] }
  | { kind: 'import'; id: string };

type LibraryTab = 'parts' | 'lights' | 'imports';

const LIGHTS: { type: LightData['type']; name: string; meta: string; icon: string }[] = [
  { type: 'directional', name: 'Sun light', meta: 'Parallel rays, casts shadows', icon: 'sun' },
  { type: 'point', name: 'Point light', meta: 'Glows in every direction', icon: 'bulb' },
  { type: 'ambient', name: 'Ambient light', meta: 'Lifts every shadow evenly', icon: 'ambient' },
];

let thumbnailCache: Partial<Record<Primitive, string>> | null = null;

/** Renders each primitive once, in a throwaway WebGL context, for the Library tiles. */
function renderPartThumbnails(): Partial<Record<Primitive, string>> {
  if (thumbnailCache) return thumbnailCache;
  const out: Partial<Record<Primitive, string>> = {};
  const canvas = document.createElement('canvas');
  let renderer: THREE.WebGLRenderer | null = null;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(1);
    renderer.setSize(240, 132, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xe4ecff, 0x3a332a, 2.2));
    const key = new THREE.DirectionalLight(0xffe7d0, 2.6);
    key.position.set(3, 5, 4);
    scene.add(key);
    const camera = new THREE.PerspectiveCamera(28, 240 / 132, 0.1, 50);
    const material = new THREE.MeshStandardMaterial({ color: 0xc9ced6, roughness: 0.45 });
    const factory = new MeshFactory();

    for (const { type } of PRIMITIVES) {
      const mesh = new THREE.Mesh(
        factory.createGeometry({ geometryType: type, materialIndex: 0 }),
        material,
      );
      const t = restingTransform(type, new THREE.Vector3());
      mesh.position.set(...t.position);
      mesh.quaternion.set(...t.rotation);
      scene.add(mesh);
      const box = new THREE.Box3().setFromObject(mesh);
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      const dir = new THREE.Vector3(1, 0.75, 1.35).normalize();
      camera.position
        .copy(sphere.center)
        .addScaledVector(dir, sphere.radius / Math.sin(THREE.MathUtils.degToRad(14)));
      camera.lookAt(sphere.center);
      renderer.setClearColor(0x000000, 0);
      renderer.render(scene, camera);
      out[type] = canvas.toDataURL('image/png');
      scene.remove(mesh);
      mesh.geometry.dispose();
    }
    material.dispose();
  } catch {
    // No WebGL for thumbnails: tiles fall back to icons
  } finally {
    renderer?.dispose();
    renderer?.forceContextLoss();
  }
  thumbnailCache = out;
  return out;
}

function setDrag(e: React.DragEvent, payload: LibraryDragPayload) {
  e.dataTransfer.setData(LIBRARY_DRAG_TYPE, JSON.stringify(payload));
  e.dataTransfer.effectAllowed = 'copy';
}

interface Tile {
  key: string;
  name: string;
  meta: string;
  icon: string;
  image?: string;
  payload: LibraryDragPayload;
  onInsert: () => void;
}

export function LibraryPanel({
  actions,
  onRequestImport,
}: {
  actions: EditorActions;
  onRequestImport: () => void;
}) {
  const [tab, setTab] = useState<LibraryTab>('parts');
  const [query, setQuery] = useState('');
  const [thumbs, setThumbs] = useState<Partial<Record<Primitive, string>>>(
    () => thumbnailCache ?? {},
  );
  const imports = useAssetStore((s) => s.imports);
  const playing = useEditorStore((s) => s.isPlaying);

  useEffect(() => {
    if (!thumbnailCache) {
      const id = requestAnimationFrame(() => setThumbs(renderPartThumbnails()));
      return () => cancelAnimationFrame(id);
    }
  }, []);

  const q = query.trim().toLowerCase();
  let tiles: Tile[] = [];
  if (tab === 'parts') {
    tiles = PRIMITIVES.map((p) => ({
      key: p.type,
      name: p.label,
      meta: 'Part',
      icon: p.type,
      image: thumbs[p.type],
      payload: { kind: 'part', type: p.type },
      onInsert: () => actions.addPrimitive(p.type),
    }));
  } else if (tab === 'lights') {
    tiles = LIGHTS.map((l) => ({
      key: l.type,
      name: l.name,
      meta: l.meta,
      icon: l.icon,
      payload: { kind: 'light', type: l.type },
      onInsert: () => actions.addLight(l.type),
    }));
  } else {
    tiles = imports.map((a) => ({
      key: a.id,
      name: a.name,
      meta: a.file.name,
      icon: 'model',
      payload: { kind: 'import', id: a.id },
      onInsert: () => void actions.importModel(a.file, { remember: false }),
    }));
  }
  const shown = tiles.filter(
    (t) => !q || t.name.toLowerCase().includes(q) || t.meta.toLowerCase().includes(q),
  );

  return (
    <aside className="f3-left" aria-label="Library">
      <div className="f3-subtabs" role="tablist" aria-label="Library sections">
        {(
          [
            ['parts', 'Parts'],
            ['lights', 'Lights'],
            ['imports', 'Imports'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            className="f3-st"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
          >
            {label}
            {key === 'imports' && imports.length > 0 && (
              <span className="f3-cnt">{imports.length}</span>
            )}
          </button>
        ))}
      </div>

      <div style={{ padding: '10px 10px 8px', flexShrink: 0 }}>
        <label className="f3-fld">
          <Icon name="search" size={14} />
          <input
            aria-label="Search the library"
            placeholder={tab === 'imports' ? 'Search your imports' : `Search ${tab}`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
      </div>

      <div style={{ flexGrow: 1, overflowY: 'auto', padding: '2px 10px 10px', minHeight: 0 }}>
        {shown.length === 0 ? (
          <div className="f3-empty" style={{ margin: '8px 0 0' }}>
            <Icon name={tab === 'imports' ? 'import' : 'search'} size={22} />
            <span>
              {tab === 'imports' && !q
                ? 'Models you import show up here, so you can place them again.'
                : `Nothing matches “${query}”.`}
            </span>
          </div>
        ) : (
          <div className="f3-tiles">
            {shown.map((t) => (
              <button
                key={t.key}
                type="button"
                className="f3-tile"
                draggable={!playing}
                disabled={playing}
                onDragStart={(e) => setDrag(e, t.payload)}
                onClick={t.onInsert}
                title={`Click to insert ${t.name}, or drag it into the scene`}
              >
                <span className="f3-thumb">
                  {t.image ? (
                    <img src={t.image} alt="" draggable={false} />
                  ) : (
                    <Icon name={t.icon} size={30} style={{ strokeWidth: 1.3 }} />
                  )}
                </span>
                <span className="f3-tile-name">{t.name}</span>
                <span className="f3-tile-meta">{t.meta}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <div style={{ padding: 10, borderTop: '1px solid var(--line)', flexShrink: 0 }}>
        <button
          type="button"
          className="f3-btn ghost"
          style={{ width: '100%' }}
          disabled={playing}
          onClick={onRequestImport}
        >
          <Icon name="import" size={14} />
          Import model
        </button>
      </div>
    </aside>
  );
}
