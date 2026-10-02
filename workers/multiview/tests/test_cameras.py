"""The views' camera convention, checked against MV-Adapter's own camera code (vendored, without nvdiffrast)."""

import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from multiview_worker.cameras import AZIMUTHS, HALF_EXTENT, IMAGE_SIZE, camera_info, camera_to_world, project

WORKER = pathlib.Path(__file__).parent.parent
FRONT, RIGHT, BACK, LEFT, TOP = (0, -0.5, 0), (0.5, 0, 0), (0, 0.5, 0), (-0.5, 0, 0), (0, 0, 0.5)


def test_azimuth_zero_looks_at_the_front_from_minus_y():
    matrix = camera_to_world(0)
    np.testing.assert_allclose(matrix[:3, 3], [0, -1.8, 0], atol=1e-9)
    np.testing.assert_allclose(matrix[:3, 0], [1, 0, 0], atol=1e-9)  # image right: +X
    np.testing.assert_allclose(matrix[:3, 1], [0, 0, 1], atol=1e-9)  # image up: +Z
    # The object's right-hand side (+X) is on the image's right; the top is up
    x_right, _ = project(np.array([RIGHT]), 0)[0]
    _, y_top = project(np.array([TOP]), 0)[0]
    assert x_right > IMAGE_SIZE / 2 and y_top < IMAGE_SIZE / 2


def test_positive_azimuth_turns_the_camera_towards_the_right_of_the_front_view():
    # 90: the camera stands on the side that is on the right of the 0 view (+X) and sees the front at image-left
    np.testing.assert_allclose(camera_to_world(90)[:3, 3], [1.8, 0, 0], atol=1e-9)
    front_x, _ = project(np.array([FRONT]), 90)[0]
    back_x, _ = project(np.array([BACK]), 90)[0]
    assert front_x < IMAGE_SIZE / 2 < back_x
    # 180 sees the back with left and right swapped; 270 stands on the left
    np.testing.assert_allclose(camera_to_world(180)[:3, 3], [0, 1.8, 0], atol=1e-9)
    assert project(np.array([RIGHT]), 180)[0][0] < IMAGE_SIZE / 2
    np.testing.assert_allclose(camera_to_world(270)[:3, 3], [-1.8, 0, 0], atol=1e-9)
    # Counter-clockwise seen from above: 45 lies between the front (-Y) and the right (+X)
    position = camera_to_world(45)[:3, 3]
    assert position[0] > 0 and position[1] < 0


def test_every_view_is_level_and_orthographic_at_the_same_scale():
    for azimuth in AZIMUTHS:
        matrix = camera_to_world(azimuth)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-9)
        np.testing.assert_allclose(matrix[:3, 1], [0, 0, 1], atol=1e-9)
        # A unit-long vertical segment through the origin is 698 px tall in every view, centred
        top, bottom = project(np.array([[0, 0, 0.5], [0, 0, -0.5]]), azimuth)
        assert bottom[1] - top[1] == pytest.approx(IMAGE_SIZE / (2 * HALF_EXTENT))
        assert top[0] == pytest.approx(IMAGE_SIZE / 2)
    # The frame's edges are at +-0.55 along the camera's right axis
    np.testing.assert_allclose(project(np.array([[0.55, 0, 0], [-0.55, 0, 0]]), 0)[:, 0], [IMAGE_SIZE, 0], atol=1e-9)


def test_camera_info_is_plain_json():
    import json

    info = json.loads(json.dumps(camera_info()))
    assert info["type"] == "orthographic" and info["half_extent"] == 0.55 and info["elevation"] == 0
    assert info["pixels_per_unit"] == pytest.approx(698.182, abs=1e-3)
    assert info["up"] == [0, 0, 1] and info["front"] == [0, -1, 0]


def test_matches_mv_adapters_own_cameras():
    torch = pytest.importorskip("torch")
    from mvadapter.utils.mesh_utils import get_orthogonal_camera

    # As scripts/inference_i2mv_sdxl.py calls it
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * 6,
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
    )
    for index, azimuth in enumerate(AZIMUTHS):
        np.testing.assert_allclose(cameras.c2w[index].double().numpy(), camera_to_world(azimuth), atol=1e-6)
    assert isinstance(cameras.c2w, torch.Tensor)


def test_control_images_carry_each_views_direction():
    pytest.importorskip("torch")
    from multiview_worker.generator import control_images

    control = control_images("cpu")
    assert tuple(control.shape) == (6, 6, IMAGE_SIZE, IMAGE_SIZE)
    # Channels 0-2: the viewing direction, 3-5: the camera's direction from the origin, mapped to 0-1
    looks = {azimuth: control[index, :, 0, 0].numpy() * 2 - 1 for index, azimuth in enumerate(AZIMUTHS)}
    np.testing.assert_allclose(looks[0], [0, 1, 0, 0, -1, 0], atol=1e-6)  # from -Y, looking along +Y
    np.testing.assert_allclose(looks[90], [-1, 0, 0, 1, 0, 0], atol=1e-6)  # from +X
    np.testing.assert_allclose(looks[180], [0, -1, 0, 0, 1, 0], atol=1e-6)


def test_the_vendored_code_never_imports_nvdiffrast():
    # Statically: no vendored file names it in an import
    for path in (WORKER / "mvadapter").rglob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+nvdiffrast", path.read_text(), re.M), path
    # And at run time, with nvdiffrast and trimesh made unimportable
    code = (
        "import sys; sys.modules['nvdiffrast'] = None; sys.modules['trimesh'] = None\n"
        "import mvadapter.utils, mvadapter.utils.mesh_utils, mvadapter.utils.geometry\n"
        "import mvadapter.schedulers.scheduling_shift_snr\n"
        "assert 'nvdiffrast' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
    )
    pytest.importorskip("torch")
    subprocess.run([sys.executable, "-c", code], cwd=WORKER, check=True)


def test_the_pipeline_imports_without_nvdiffrast():
    pytest.importorskip("diffusers")
    pytest.importorskip("einops")
    code = (
        "import sys; sys.modules['nvdiffrast'] = None; sys.modules['trimesh'] = None\n"
        "from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=WORKER, check=True)
