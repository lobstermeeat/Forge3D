"""
Job handling for the Pixal3D worker: production's job contract (forge3d_worker.service), plus views.

Input: production's ``{"image_url" | "image_base64", "mode", "seed"?, "request_id"?}`` and, optionally,
``"views": [{"image_url" | "image_base64", "azimuth", "elevation"}]`` with the views' ``"camera"``
(the multiview worker's; MV-Adapter's framing when absent). With views, both the preview and the final
are made from the picture and the views; the picture stays the main image, and the final's projection
still paints from it. The result is production's, plus ``"views_used"``: how many of the job's views
went into the model (0 without views).
"""

from __future__ import annotations

from typing import Any, Optional

from forge3d_worker import service as production
from forge3d_worker.inputs import Fetch, InputError
from forge3d_worker.service import Pack
from forge3d_worker.settings import Preset
from forge3d_worker.storage import Storage

from .views import View, ViewCamera, parse_camera, parse_views

# Attribution the DINOv3 License requires, and what made the model
CREDITS = (
    "Built with DINOv3",
    "3D generation: Pixal3D (Tencent, MIT) on TRELLIS.2 (Microsoft, MIT)",
)


class _JobRuntime:
    """
    The runtime as production's handle_job drives it (generate, then export), with this job's views
    bound to generate.
    """

    def __init__(self, runtime: Any, views: Optional[list[View]], camera: ViewCamera) -> None:
        self.runtime, self.views, self.camera = runtime, views, camera
        self.pipeline_used: Optional[str] = None
        self.last_projection: Optional[dict] = None
        self.views_used = 0

    def generate(self, image: Any, preset: Preset, seed: int) -> Any:
        if self.views:
            mesh = self.runtime.generate_views(image, self.views, preset, seed, self.camera)
        else:
            mesh = self.runtime.generate(image, preset, seed)
        self.pipeline_used = getattr(self.runtime, "pipeline_used", None)
        self.views_used = int(getattr(self.runtime, "views_used", 0) or 0)
        return mesh

    def export(self, mesh: Any, preset: Preset) -> tuple[bytes, int]:
        try:
            return self.runtime.export(mesh, preset)
        finally:
            self.last_projection = getattr(self.runtime, "last_projection", None)


def handle_job(
    job: dict,
    runtime: Any,
    storage: Storage,
    pack: Pack,
    fetch: Optional[Fetch] = None,
) -> dict:
    """Production's handle_job, with the job's views (see the module's docstring)."""
    payload = job.get("input")
    views: Optional[list[View]] = None
    camera = ViewCamera()
    if isinstance(payload, dict) and payload.get("views") is not None:
        try:
            views = parse_views(payload["views"], fetch=fetch)
            camera = parse_camera(payload.get("camera"))
        except InputError as err:
            return {"error": f"invalid input: {err}"}
    bound = _JobRuntime(runtime, views, camera)
    result = production.handle_job(job, bound, storage, pack, fetch=fetch)
    if "error" not in result:
        result["views_used"] = bound.views_used
        result["credits"] = list(CREDITS)
        camera_used = getattr(runtime, "last_camera", None)
        if isinstance(camera_used, dict) and camera_used:
            result["camera"] = camera_used
    return result
