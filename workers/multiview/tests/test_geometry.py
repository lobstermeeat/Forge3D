"""The image+geometry model's inputs: the control maps' encoding and transport, and the call into the pipeline."""

import base64
import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from multiview_worker import generator
from multiview_worker.cameras import IG2MV_AZIMUTHS, IG2MV_ELEVATIONS, IG2MV_VIEWS, camera_to_world, project
from multiview_worker.geometry import (
    CONTROL_SHAPE,
    IG2MV_FILE,
    GeometryViewGenerator,
    check_control,
    encode_control,
    gltf_to_world,
    mesh_to_world,
    pack_control,
    unpack_control,
)
from multiview_worker.inputs import InputError

SIZE = CONTROL_SHAPE[-1]


def sample_control(seed=0):
    """A control array with smooth maps (each view and channel its own ramp) and a background of 0.5."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)
    control = np.full(CONTROL_SHAPE, 0.5, dtype=np.float32)
    for view in range(CONTROL_SHAPE[0]):
        for channel in range(CONTROL_SHAPE[1]):
            a, b = rng.uniform(0, 0.5, 2)
            control[view, channel, 100:600, 150:650] = a + b * (ramp[100:600, None] + ramp[None, 150:650]) / 2
    return control


def png_bytes(image, **options):
    buffer = io.BytesIO()
    image.save(buffer, **options)
    return buffer.getvalue()


# Transport: 12 PNGs


def test_the_control_travels_as_twelve_8_bit_pngs_positions_first():
    control = sample_control()
    pngs = pack_control(control)
    assert len(pngs) == 12 and all(isinstance(png, str) for png in pngs)
    for index, png in enumerate(pngs):
        image = Image.open(io.BytesIO(base64.b64decode(png)))
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (SIZE, SIZE))
        view, first = index % 6, 3 * (index // 6)  # the six position maps, then the six normal maps
        expected = np.round(control[view, first : first + 3].transpose(1, 2, 0) * 255)
        np.testing.assert_array_equal(np.asarray(image), expected)


def test_packing_round_trips_within_8_bits():
    control = sample_control(seed=3)
    unpacked = unpack_control(pack_control(control))
    assert unpacked.shape == CONTROL_SHAPE and unpacked.dtype == np.float32
    assert np.abs(unpacked - control).max() <= 0.5 / 255 + 1e-6
    # The PNG files themselves (bytes) unpack the same
    files = [base64.b64decode(png) for png in pack_control(control)]
    np.testing.assert_array_equal(unpack_control(files), unpacked)


def bad_pngs(**change):
    pngs = pack_control(np.full(CONTROL_SHAPE, 0.5, dtype=np.float32))
    for index, value in change.items():
        pngs[int(index.lstrip("_"))] = value
    return pngs


GRAY = Image.new("RGB", (SIZE, SIZE), (127, 127, 127))


@pytest.mark.parametrize(
    "pngs, message",
    [
        ("not a list", "list of 12 PNGs"),
        (bad_pngs()[:11], "list of 12 PNGs"),
        (bad_pngs(_3="%%%"), "control PNG 3 is not valid base64"),
        (bad_pngs(_4=7), "control PNG 4 must be a base64 string"),
        (bad_pngs(_5=base64.b64encode(b"not a picture").decode()), "control PNG 5 could not be decoded"),
        (bad_pngs(_6=base64.b64encode(png_bytes(GRAY, format="JPEG")).decode()), "8-bit RGB PNG, not JPEG RGB"),
        (bad_pngs(_7=base64.b64encode(png_bytes(GRAY.convert("RGBA"), format="PNG")).decode()), "not PNG RGBA"),
        (bad_pngs(_8=base64.b64encode(png_bytes(GRAY.resize((512, 512)), format="PNG")).decode()), "not 512 x 512"),
        (bad_pngs(_11=base64.b64encode(png_bytes(GRAY, format="PNG")[:-200]).decode()), "could not be decoded"),
    ],
)
def test_bad_control_pngs_are_input_errors(pngs, message):
    with pytest.raises(InputError, match=message):
        unpack_control(pngs)


def test_the_control_array_is_checked():
    control = sample_control()
    with pytest.raises(InputError, match="shape"):
        check_control(control[:4])
    broken = control.copy()
    broken[2, 4, 300, 300] = np.nan
    with pytest.raises(InputError, match="not finite"):
        check_control(broken)
    # Raw normals in [-1, 1] mean the maps were never encoded
    with pytest.raises(InputError, match=r"must be in \[0, 1\]"):
        check_control(control * 2 - 1)
    # Rounding overshoot is clipped
    nudged = control.copy()
    nudged[0, 0, 0, 0] = 1.0004
    assert check_control(nudged).max() == 1.0


# Encoding: world positions and normals, as upstream's script makes them


def test_positions_and_normals_are_encoded_like_upstream():
    rng = np.random.default_rng(1)
    positions = rng.uniform(-0.5, 0.5, (6, 4, 5, 3))
    normals = rng.normal(size=(6, 4, 5, 3)) * 3  # not unit length: upstream normalises after interpolating
    masks = rng.uniform(size=(6, 4, 5)) > 0.3
    control = encode_control(positions, normals, masks)
    assert control.shape == (6, 6, 4, 5) and control.dtype == np.float32
    unit = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    inside = control.transpose(0, 2, 3, 1)[masks]
    np.testing.assert_allclose(inside[:, :3], positions[masks] + 0.5, atol=1e-6)
    np.testing.assert_allclose(inside[:, 3:], unit[masks] / 2 + 0.5, atol=1e-6)
    # Where no surface is, positions and normals are 0 before encoding: mid-gray in both
    np.testing.assert_array_equal(control.transpose(0, 2, 3, 1)[~masks], 0.5)
    # Positions beyond the unit cube are clamped, as upstream's (pos + 0.5).clamp(0, 1)
    far = encode_control(np.full((6, 1, 1, 3), 0.75), np.ones((6, 1, 1, 3)), np.ones((6, 1, 1), bool))
    assert far[:, :3].max() == 1.0


def test_encode_control_wants_six_views_of_matching_maps():
    with pytest.raises(ValueError, match="positions must be"):
        encode_control(np.zeros((4, 2, 2, 3)), np.zeros((4, 2, 2, 3)), np.zeros((4, 2, 2), bool))
    with pytest.raises(ValueError, match="must match"):
        encode_control(np.zeros((6, 2, 2, 3)), np.zeros((6, 2, 3, 3)), np.zeros((6, 2, 2), bool))


def upstream_load_mesh(vertices, normals):
    """What MV-Adapter's load_mesh(path, rescale=True) does to them (mesh_utils/mesh.py, its defaults)."""
    vertices = vertices / np.abs(vertices).max() * 0.5
    dir2vec = {"+x": np.array([1, 0, 0]), "+y": np.array([0, 1, 0])}
    z_, x_ = dir2vec["+y"], dir2vec["+x"]  # shape_init_mesh_up, shape_init_mesh_front
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)
    return np.dot(mesh2std, vertices.T).T, np.dot(mesh2std, normals.T).T


def test_meshes_go_into_mv_adapters_world_as_upstream_loads_them():
    rng = np.random.default_rng(2)
    vertices = rng.uniform(-0.3, 0.9, (50, 3))  # off-centre: upstream does not move the origin
    normals = rng.normal(size=(50, 3))
    expected_vertices, expected_normals = upstream_load_mesh(vertices, normals)
    np.testing.assert_allclose(mesh_to_world(vertices), expected_vertices, atol=1e-12)
    np.testing.assert_allclose(gltf_to_world(normals), expected_normals, atol=1e-12)
    assert np.abs(mesh_to_world(vertices)).max() == pytest.approx(0.5)


def test_the_gltf_front_faces_the_front_camera_and_up_is_up():
    # glTF: +Y up, the front towards +Z (TRELLIS.2's front)
    front, top = mesh_to_world(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(front, [0, -0.5, 0])
    np.testing.assert_allclose(top, [0, 0, 0.5])
    camera = camera_to_world(IG2MV_AZIMUTHS[0], IG2MV_ELEVATIONS[0])[:3, 3]
    assert np.linalg.norm(camera - front) < np.linalg.norm(camera - top)
    np.testing.assert_allclose(project(front[None], 0)[0], [SIZE / 2, SIZE / 2])
    with pytest.raises(ValueError, match="origin"):
        mesh_to_world(np.zeros((3, 3)))


# The pipeline, with stand-ins for diffusers and MV-Adapter's modules (no GPU, no weights)


class FakeModule:
    def __init__(self, keys):
        self.keys = keys
        self.moves = []

    def state_dict(self):
        return dict.fromkeys(self.keys)

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self


class FakePipeline:
    made = []

    def __init__(self, folder, **kwargs):
        self.folder, self.options = folder, kwargs
        self.scheduler = "the base scheduler"
        self.calls = []

    @classmethod
    def from_pretrained(cls, folder, **kwargs):
        pipe = cls(folder, **kwargs)
        cls.made.append(pipe)
        return pipe

    def init_custom_adapter(self, **kwargs):
        self.adapter = kwargs
        self.unet = FakeModule({"down.attn1.to_q.weight", "down.attn1.processor.to_q_mv.weight", "down.attn1.processor.to_k_ref.weight"})
        self.cond_encoder = FakeModule({"adapter.conv_in.weight"})

    def load_custom_adapter(self, weights, weight_name):
        self.loaded = (sorted(weights), weight_name)

    def to(self, **kwargs):
        self.moved = kwargs
        return self

    def enable_vae_slicing(self):
        self.sliced = True

    def set_progress_bar_config(self, **kwargs):
        self.progress = kwargs

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        count = kwargs["num_images_per_prompt"]
        return types.SimpleNamespace(images=[Image.new("RGB", (SIZE, SIZE), (20 * i, 90, 90)) for i in range(count)])


class RowProcessor:
    pass


class RowColProcessor:
    pass


def write_adapter(folder, name, keys):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    save_file({key: torch.zeros(1) for key in keys}, str(folder / name))


ADAPTER_KEYS = ("down.attn1.processor.to_q_mv.weight", "down.attn1.processor.to_k_ref.weight", "adapter.conv_in.weight")


@pytest.fixture
def models(tmp_path, monkeypatch):
    """A models folder with tiny adapter files, and MV-Adapter's pipeline modules replaced by stand-ins."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    FakePipeline.made = []
    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKL = types.SimpleNamespace(from_pretrained=lambda folder, **kw: ("vae", folder, kw))
    pipelines = types.ModuleType("mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl")
    pipelines.MVAdapterI2MVSDXLPipeline = FakePipeline
    schedulers = types.ModuleType("mvadapter.schedulers.scheduling_shift_snr")
    schedulers.ShiftSNRScheduler = types.SimpleNamespace(from_scheduler=lambda base, **kw: ("shift-snr", base, kw))
    attention = types.ModuleType("mvadapter.models.attention_processor")
    attention.DecoupledMVRowSelfAttnProcessor2_0 = RowProcessor
    attention.DecoupledMVRowColSelfAttnProcessor2_0 = RowColProcessor
    for name, module in [
        ("diffusers", diffusers),
        ("mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl", pipelines),
        ("mvadapter.schedulers.scheduling_shift_snr", schedulers),
        ("mvadapter.models.attention_processor", attention),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    for name in (generator.ADAPTER_FILE, IG2MV_FILE):
        write_adapter(tmp_path / generator.ADAPTER_DIR, name, ADAPTER_KEYS)
    return tmp_path


def remover(image):
    """BiRefNet's stand-in: the middle of the picture is the object."""
    cutout = image.convert("RGBA")
    alpha = Image.new("L", image.size, 0)
    alpha.paste(255, (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4))
    cutout.putalpha(alpha)
    return cutout


def test_the_image_geometry_model_loads_with_row_col_attention_and_its_own_adapter(models):
    import torch

    GeometryViewGenerator(str(models), device="cpu", remover=remover)
    pipe = FakePipeline.made[-1]
    assert pipe.folder == str(models / generator.SDXL_DIR)
    assert pipe.options["vae"] == ("vae", str(models / generator.VAE_DIR), {"torch_dtype": torch.float16})
    assert pipe.options["torch_dtype"] == torch.float16 and pipe.options["variant"] == "fp16"
    assert pipe.options["add_watermarker"] is False
    assert pipe.scheduler == ("shift-snr", "the base scheduler", {"shift_mode": "interpolated", "shift_scale": 8.0})
    assert pipe.adapter == {"num_views": 6, "self_attn_processor": RowColProcessor, "copy_attn_weights": False}
    assert pipe.loaded == (sorted(ADAPTER_KEYS), IG2MV_FILE)
    assert pipe.moved == {"device": "cpu", "dtype": torch.float16}
    assert pipe.cond_encoder.moves == [{"device": "cpu", "dtype": torch.float16}]
    assert pipe.sliced and pipe.progress == {"disable": True}


def test_the_image_to_multiview_model_keeps_row_attention_and_its_adapter(models):
    generator.MultiViewGenerator(str(models), device="cpu", remover=remover)
    pipe = FakePipeline.made[-1]
    assert pipe.adapter == {"num_views": 6, "self_attn_processor": RowProcessor, "copy_attn_weights": False}
    assert pipe.loaded == (sorted(ADAPTER_KEYS), generator.ADAPTER_FILE)


def test_a_missing_or_mismatched_adapter_file_is_named(models):
    (models / generator.ADAPTER_DIR / IG2MV_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="download the multiview weights again"):
        GeometryViewGenerator(str(models), device="cpu", remover=remover)
    write_adapter(models / generator.ADAPTER_DIR, IG2MV_FILE, ADAPTER_KEYS[:2])  # no condition encoder
    with pytest.raises(RuntimeError, match=f"{IG2MV_FILE} doesn't match the pipeline: 0 unused tensors"):
        GeometryViewGenerator(str(models), device="cpu", remover=remover)


def test_draw_views_calls_the_pipeline_as_upstreams_script_does(models):
    import torch

    model = GeometryViewGenerator(str(models), device="cpu", remover=remover)
    picture = Image.new("RGB", (640, 480), (200, 40, 40))
    control = sample_control()
    views = model.draw_views(picture, control, "a wooden shield with a lion", seed=11)

    assert len(views) == 6 and all(view.mode == "RGB" and view.size == (SIZE, SIZE) for view in views)
    prompt, kwargs = FakePipeline.made[-1].calls[-1]
    assert prompt == "a wooden shield with a lion"
    assert kwargs["num_images_per_prompt"] == 6 == len(IG2MV_VIEWS)
    assert (kwargs["height"], kwargs["width"]) == (SIZE, SIZE)
    assert kwargs["num_inference_steps"] == 30 and kwargs["guidance_scale"] == 3.0
    image = kwargs["control_image"]
    assert isinstance(image, torch.Tensor) and tuple(image.shape) == (6, 6, SIZE, SIZE)
    assert image.dtype == torch.float32 and image.device.type == "cpu"
    np.testing.assert_array_equal(image.numpy(), control)
    assert kwargs["control_conditioning_scale"] == 1.0 and kwargs["reference_conditioning_scale"] == 1.0
    assert kwargs["negative_prompt"] == "watermark, ugly, deformed, noisy, blurry, low contrast"
    assert kwargs["generator"].initial_seed() == 11
    # The reference: BiRefNet's cutout, framed by MV-Adapter's preprocess_image on mid-gray
    expected = generator.prepare_reference(remover(picture))
    np.testing.assert_array_equal(np.asarray(kwargs["reference_image"]), np.asarray(expected))
    assert model.last_reference is kwargs["reference_image"]
    assert set(model.last_timings) == {"cutout_s", "views_s"}


def test_draw_views_settings_reach_the_pipeline(models):
    model = GeometryViewGenerator(str(models), device="cpu", remover=remover)
    model.draw_views(
        Image.new("RGB", (64, 64), (1, 2, 3)),
        sample_control(),
        steps=12,
        guidance=5.5,
        reference_scale=0.8,
        control_scale=0.6,
    )
    prompt, kwargs = FakePipeline.made[-1].calls[-1]
    assert prompt == "high quality"
    assert kwargs["generator"].initial_seed() == 0
    assert (kwargs["num_inference_steps"], kwargs["guidance_scale"]) == (12, 5.5)
    assert (kwargs["reference_conditioning_scale"], kwargs["control_conditioning_scale"]) == (0.8, 0.6)


def test_draw_views_refuses_unencoded_maps_before_drawing(models):
    model = GeometryViewGenerator(str(models), device="cpu", remover=remover)
    with pytest.raises(InputError, match=r"in \[0, 1\]"):
        model.draw_views(Image.new("RGB", (64, 64)), sample_control() * 2 - 1)
    assert FakePipeline.made[-1].calls == []
