import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import type { ModelData } from '@forge3d/shared';
import { Entity } from '../ecs/Entity';
import { SceneManager } from './SceneManager';
import { SceneBridge, type ModelLoadEvent } from './SceneBridge';

/** A loader whose loads finish when the test says so. */
function controlledLoader() {
  const pending = new Map<
    string,
    { resolve: (o: THREE.Object3D) => void; reject: (e: Error) => void }
  >();
  const loader = vi.fn(
    (url: string) =>
      new Promise<THREE.Object3D>((resolve, reject) => pending.set(url, { resolve, reject })),
  );
  return { loader, pending };
}

/** A 2 x 1 x 2 box whose base sits at y = 3, off-centre: loading must rest it on the origin. */
function boxModel(): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 1, 2), new THREE.MeshStandardMaterial());
  mesh.position.set(5, 3.5, -1);
  return mesh;
}

function setup() {
  const sceneManager = new SceneManager();
  const scene = new THREE.Scene();
  const bridge = new SceneBridge(sceneManager, scene);
  bridge.connect();
  const events: ModelLoadEvent[] = [];
  bridge.onModelLoad((event) => events.push(event));
  const entity = new Entity('Chest');
  entity.setComponent<ModelData>('model', { url: '/uploads/ai/1/preview.glb', source: 'ai' });
  return { sceneManager, scene, bridge, events, entity };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('SceneBridge model entities', () => {
  it('shows the model file resting on the entity origin once it loads', async () => {
    const { sceneManager, bridge, events, entity } = setup();
    const { loader, pending } = controlledLoader();
    bridge.setModelLoader(loader);
    sceneManager.addEntity(entity);

    const holder = bridge.getObject(entity.id)!;
    expect(holder).toBeInstanceOf(THREE.Group);
    expect(holder.children).toHaveLength(0);
    expect(loader).toHaveBeenCalledWith('/uploads/ai/1/preview.glb');

    const model = boxModel();
    pending.get('/uploads/ai/1/preview.glb')!.resolve(model);
    await flush();

    expect(holder.children).toHaveLength(1);
    expect(model.castShadow).toBe(true);
    holder.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(holder);
    expect(box.min.y).toBeCloseTo(0);
    expect(box.getCenter(new THREE.Vector3()).x).toBeCloseTo(0);
    expect(box.getCenter(new THREE.Vector3()).z).toBeCloseTo(0);
    expect(events).toEqual([
      { entityId: entity.id, url: '/uploads/ai/1/preview.glb', status: 'loaded' },
    ]);
    // Picking a mesh inside finds the entity
    expect(bridge.getEntityId(model)).toBe(entity.id);
  });

  it('loads again when the file changes, as when a preview becomes the final', async () => {
    const { sceneManager, bridge, entity } = setup();
    const { loader, pending } = controlledLoader();
    bridge.setModelLoader(loader);
    sceneManager.addEntity(entity);
    const preview = boxModel();
    pending.get('/uploads/ai/1/preview.glb')!.resolve(preview);
    await flush();
    const disposed = vi.spyOn(preview.geometry, 'dispose');

    entity.setComponent<ModelData>('model', {
      url: '/uploads/ai/1/final.glb',
      source: 'ai',
      quality: 'final',
    });
    sceneManager.notifyChange();
    // Unrelated changes don't start the same load twice
    sceneManager.notifyChange();
    expect(loader).toHaveBeenCalledTimes(2);

    pending.get('/uploads/ai/1/final.glb')!.resolve(boxModel());
    await flush();
    const holder = bridge.getObject(entity.id)!;
    expect(holder.children).toHaveLength(1);
    expect(holder.children[0]!.children[0]).not.toBe(preview);
    expect(disposed).toHaveBeenCalled();
  });

  it('drops a load that was overtaken or whose entity is gone', async () => {
    const { sceneManager, bridge, entity } = setup();
    const { loader, pending } = controlledLoader();
    bridge.setModelLoader(loader);
    sceneManager.addEntity(entity);
    entity.setComponent<ModelData>('model', { url: '/b.glb', source: 'ai' });
    sceneManager.notifyChange();

    const stale = boxModel();
    const staleDisposed = vi.spyOn(stale.geometry, 'dispose');
    pending.get('/b.glb')!.resolve(boxModel());
    pending.get('/uploads/ai/1/preview.glb')!.resolve(stale);
    await flush();
    expect(bridge.getObject(entity.id)!.children).toHaveLength(1);
    expect(staleDisposed).toHaveBeenCalled();

    const other = new Entity('Other');
    other.setComponent<ModelData>('model', { url: '/c.glb', source: 'ai' });
    sceneManager.addEntity(other);
    sceneManager.removeEntity(other.id);
    const orphan = boxModel();
    const orphanDisposed = vi.spyOn(orphan.geometry, 'dispose');
    pending.get('/c.glb')!.resolve(orphan);
    await flush();
    expect(orphanDisposed).toHaveBeenCalled();
  });

  it('reports a failed load once instead of retrying on every change', async () => {
    const { sceneManager, bridge, events, entity } = setup();
    const { loader, pending } = controlledLoader();
    bridge.setModelLoader(loader);
    sceneManager.addEntity(entity);
    pending.get('/uploads/ai/1/preview.glb')!.reject(new Error('404'));
    await flush();
    sceneManager.notifyChange();
    expect(loader).toHaveBeenCalledTimes(1);
    expect(events).toMatchObject([{ status: 'failed', entityId: entity.id }]);
  });

  it('waits for a loader, then loads every model in the scene', () => {
    const { sceneManager, bridge, entity } = setup();
    sceneManager.addEntity(entity);
    const { loader } = controlledLoader();
    bridge.setModelLoader(loader);
    expect(loader).toHaveBeenCalledWith('/uploads/ai/1/preview.glb');
  });
});

describe('Entity model component', () => {
  it('survives serializing, as scene saves need', () => {
    const entity = new Entity('Chest');
    const model: ModelData = {
      url: '/uploads/ai/1/final.glb',
      source: 'ai',
      generationId: '6c1f0c5e-58f6-4f53-9a55-5d8c1e0b9e11',
      quality: 'final',
      credits: ['Built with DINOv3'],
    };
    entity.setComponent('model', model);
    const copy = Entity.deserialize(JSON.parse(JSON.stringify(entity.serialize())));
    expect(copy.getComponent('model')).toEqual(model);
  });
});
