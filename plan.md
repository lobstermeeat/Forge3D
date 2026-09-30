# Orainge — Internal Alpha Release Plan

**Exit Criteria:** 3D viewport, basic modeling, file I/O, and user accounts functional across Chrome, Firefox, and Edge.

---

## Status: What's Done

### 3D Viewport
- [x] Three.js Renderer with WebGPU + WebGL fallback
- [x] Orbit controls (pan, rotate, zoom, double-click reset)
- [x] Transform gizmo (translate/rotate/scale with G/R/S shortcuts, Shift-snap)
- [x] Default scene (ambient + directional lights, ground grid)
- [x] Responsive canvas with ResizeObserver
- [x] Renderer type badge (shows WEBGL/WEBGPU)

### Engine Core
- [x] Entity Component System (Entity class with component Map)
- [x] SceneManager (entity registry, hierarchy, materials, pub/sub)
- [x] CommandHistory (undo/redo stacks with Command interface)
- [x] MeshFactory (box, sphere, plane, cylinder, torus)
- [x] MaterialFactory (standard/physical PBR with caching)
- [x] AssetLoader (GLTF/GLB with Draco, meshopt and KTX2 support)

### Server
- [x] Fastify server with CORS, health check
- [x] better-auth (email/password + GitHub + Google OAuth)
- [x] Drizzle ORM schema (users, sessions, accounts, projects, scenes, assets, ai_generations)
- [x] tRPC router with project CRUD + scene save/load
- [x] Protected procedures (auth middleware)
- [x] AI Orchestrator with the self-hosted TRELLIS.2 provider (Modal or RunPod serverless, see workers/)

### Client Shell
- [x] Vite + React 19 setup with path aliases
- [x] React Router (login, dashboard, editor routes)
- [x] Zustand editor store (selection, transform mode, grid, sidebar panel)
- [x] useEngine hook (renderer + controls lifecycle)
- [x] Toolbar, HierarchyPanel, Viewport components
- [x] LoginPage UI (form + social buttons, not wired to auth)
- [x] DashboardPage UI (project grid, not wired to API)

### Infrastructure
- [x] pnpm + Turborepo monorepo
- [x] TypeScript configs (base, react, node)
- [x] ESLint configs (base, react, node)
- [x] Docker Compose (PostgreSQL 17 + Redis 7)
- [x] GitHub Actions CI pipeline
- [x] Shared types, Zod schemas, constants

---

## Remaining: What Needs to Be Done

### P0 — Required for Alpha (Blocking)

#### 1. Scene ↔ Viewport Wiring
The SceneManager and Three.js scene are completely disconnected. This is the most critical gap.
- [ ] Sync SceneManager entities → Three.js Object3D (add/remove meshes in viewport when entities change)
- [ ] Sync Entity.transform ↔ THREE.Object3D.position/rotation/scale
- [ ] Wire GizmoControls to selected entity (attach gizmo, update Entity on drag)
- [ ] Create meshes from MeshFactory when entity has MeshRenderer component
- [ ] Apply materials from MaterialFactory to scene meshes
- [ ] Reflect hierarchy panel from live SceneManager state (not empty array)

#### 2. Basic Modeling Tools (Add Primitives)
Users need to add objects to the scene.
- [ ] "Add" menu/button in Toolbar (Box, Sphere, Plane, Cylinder, Torus)
- [ ] Create Entity + Three.js mesh on click
- [ ] Undo/redo for add/delete operations (wire CommandHistory)
- [ ] Delete selected entity (keyboard Delete key + hierarchy panel)
- [ ] Duplicate selected entity

#### 3. Properties / Inspector Panel
Users need to edit object properties.
- [ ] Inspector panel component (shows when entity selected)
- [ ] Transform fields (position X/Y/Z, rotation, scale) with numeric inputs
- [ ] Material assignment dropdown
- [ ] Basic material editing (color picker, metalness/roughness sliders)
- [ ] Entity rename in inspector

#### 4. File I/O
Users need to save/load their work and import/export models.
- [ ] Save scene to JSON (download as .forge3d or .json file)
- [ ] Load scene from JSON file (file picker)
- [ ] Export scene as GLTF/GLB
- [ ] Import GLTF/GLB model (drag-and-drop + file picker)
- [ ] Ctrl+S shortcut for save
- [ ] Ctrl+Z / Ctrl+Shift+Z for undo/redo

#### 5. User Accounts (Wire Auth)
Login page exists but bypasses auth entirely.
- [ ] Wire LoginPage to better-auth (signIn.email, signUp.email, signIn.social)
- [ ] Wire DashboardPage to tRPC (list projects, create project)
- [ ] Session-based route protection (redirect to /login if not authenticated)
- [ ] Wire DashboardPage sign-out to better-auth
- [ ] Show user name/email in dashboard header from session
- [ ] Database setup: run migrations (requires PostgreSQL running)

#### 6. Project Persistence (Server Integration)
- [ ] Create project on "New Project" → navigate to /editor/:projectId
- [ ] Auto-create initial scene when project is created
- [ ] Save scene to server via tRPC (POST scene data)
- [ ] Load scene from server on editor mount
- [ ] Project list in dashboard from tRPC
- [ ] Auto-save on interval or on blur

### P1 — Important for Alpha Quality

#### 7. Cross-Browser Testing
- [ ] Verify WebGL fallback works on Firefox (no WebGPU)
- [ ] Test on Chrome, Firefox, Edge
- [ ] Add browser warning if WebGL not supported
- [ ] Test orbit controls + transform gizmo across browsers
- [ ] Handle canvas context loss gracefully

#### 8. Keyboard Shortcuts
- [ ] Ctrl+Z undo, Ctrl+Shift+Z redo
- [ ] Ctrl+S save
- [ ] Delete key removes selected entity
- [ ] Ctrl+D duplicate
- [ ] F focus on selected object
- [ ] Escape deselect

#### 9. UI Polish
- [ ] Loading state while engine initializes
- [ ] Error toasts for failed operations
- [ ] Confirmation dialog for delete
- [ ] Empty state in hierarchy panel ("Add an object to start")
- [ ] Sidebar tab switching (hierarchy / materials / assets)

### P2 — Nice to Have

#### 10. AI 3D Generation Panel
- [x] AI panel component in sidebar (Library › AI, and Home › Insert › AI model)
- [x] Text prompt input for text-to-3D
- [x] Image upload for image-to-3D
- [x] Reference-image picker for text prompts (FLUX.1 [schnell] worker)
- [x] Preview (512³) → keep → final (1024³) flow, reusing the preview seed
- [x] Show the provider's credits ("Built with DINOv3") next to generated models
- [x] Progress bar during generation
- [x] Auto-import generated model into scene (saved by URL, so it survives reloads and publishing)

#### 11. Asset Library
- [ ] Asset panel in sidebar
- [ ] Upload assets (textures, models)
- [ ] Asset thumbnails
- [ ] Drag-and-drop from library to scene

#### 12. Advanced Rendering
- [x] Environment lighting (procedural studio environment, shared by editor and viewer)
- [ ] Shadows toggle
- [ ] Post-processing (SSAO, bloom)
- [ ] Camera presets (front, top, side, perspective)

#### 13. Testing
- [ ] Unit tests for SceneManager, Entity, CommandHistory
- [ ] Unit tests for MeshFactory, MaterialFactory
- [ ] Integration tests for tRPC routes
- [ ] E2E tests with Playwright (login flow, create project, add mesh, save)

---

## Implementation Order (Recommended)

```
Phase 1: Scene Integration (P0 items 1-3)
  → Makes the editor actually usable as a 3D tool

Phase 2: File I/O (P0 item 4)
  → Users can save/load work locally without server

Phase 3: Auth + Server (P0 items 5-6)
  → Requires PostgreSQL running, enables multi-user

Phase 4: Polish + Testing (P1 items 7-9)
  → Cross-browser, shortcuts, error handling

Phase 5: AI + Assets (P2 items 10-12)
  → Advanced features for post-alpha
```

---

## Architecture Notes

- **Engine ↔ React bridge** is the key integration point. SceneManager should be the single source of truth, with a React sync layer that mirrors entities into Three.js objects.
- **File I/O** should work offline first (JSON download/upload), then add server persistence on top.
- **Auth** can be deferred until scene editing works locally — the editor should function without a server for development.
- **Cross-browser**: Firefox doesn't support WebGPU yet, so WebGL fallback path must be tested.
