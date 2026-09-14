"""
Checks of the compute backend in x5_pipeline.py (section 0): the shortcuts the
CPU takes (row blocks, overlap strips) reproduce whole-image computations, and
the GPU reproduces the CPU. GPU checks are skipped without CuPy and a GPU.

Run with `python tests/test_backend.py`, or with pytest.
"""
import functools
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x5_pipeline as xp  # noqa: E402
from test_stitch import X6_LIKE, smooth_sphere_image  # noqa: E402

LENS = xp.MEILensParams(fov_deg=193.0, **X6_LIKE)
RS_ROTATIONS = [(float(f), Rotation.from_euler('XYZ', [61 + 2 * f, 7 - f, 3 * f], degrees=True))
                for f in np.linspace(0.0, 1.0, 32)]


def needs_gpu(test):
    @functools.wraps(test)
    def run():
        if not xp.gpu_available():
            print('skip', test.__name__, '(no GPU)')
            return
        test()
    return run


def coverage_masks(w=720, h=360, seed=1):
    """Front and back coverage in a seam frame: wavy edges a few degrees past the equator."""
    rng = np.random.default_rng(seed)
    cols = np.arange(w)
    wave = lambda: sum(rng.uniform(2, 6) * np.sin(2 * np.pi * k * cols / w + rng.uniform(0, 6))
                       for k in (1, 3, 7))
    rows = np.arange(h)[:, np.newaxis]
    front = rows < h // 2 + 12 + wave()
    back = rows > h // 2 - 12 + wave()
    back[:, 100:104] = False             # a hole reaching into the overlap
    return front, back


def test_row_blocks_match_one_call():
    w, h = 256, 128
    rays = xp.equirect_rays(w, h)
    R = Rotation.from_euler('XYZ', [20, -35, 10], degrees=True)
    mx, my, valid = xp.build_equirect_remap(LENS, w, h, R_stabilization=R, rays=rays)
    u, v, ref_valid = xp._project_rays(LENS, R.as_matrix() @ rays, w, h)
    assert np.array_equal(valid, ref_valid)
    assert np.abs(mx - np.where(ref_valid, u, 0)).max() < 1e-3
    assert np.abs(my - np.where(ref_valid, v, 0)).max() < 1e-3


def test_blur_rows_matches_whole_image_blur():
    img = np.zeros((900, 400), np.float32)
    img[400:500] = np.random.default_rng(2).uniform(0, 3, (100, 400))
    whole = cv2.GaussianBlur(img, (0, 0), 20)
    assert np.abs(xp.gaussian_blur_rows(img[400:500], 20) - whole[400:500]).max() < 1e-5


def test_coverage_depth_strip_matches_whole_image():
    front, back = coverage_masks()
    overlap = front & back
    rows = xp.overlap_rows(overlap)
    band = slice(*rows)
    for valid, depth in zip((front, back), xp.coverage_depth((front, back), overlap, rows)):
        whole = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)[band]
        assert np.array_equal(depth[overlap[band]], whole[overlap[band]])


@needs_gpu
def test_gpu_lanczos_matches_opencv():
    rng = np.random.default_rng(0)
    mx = rng.uniform(-12, 412, (200, 300)).astype(np.float32)
    my = rng.uniform(-12, 312, (200, 300)).astype(np.float32)
    mx[:20] = np.round(mx[:20] * 64) / 64        # ties when rounding to 1/32 px
    for dtype in (np.uint8, np.float32):
        img = rng.uniform(0, 255, (300, 400, 3)).astype(dtype)
        for replicate in (False, True):
            ref = xp.remap_lanczos(img, mx, my, replicate)
            got = xp.remap_lanczos(xp.cupy.asarray(img), xp.cupy.asarray(mx),
                                   xp.cupy.asarray(my), replicate).get()
            err = np.abs(got.astype(np.float64) - ref).max()
            assert err == 0 if dtype == np.uint8 else err < 1e-3, (dtype, replicate, err)


@needs_gpu
def test_gpu_blur_rows_matches_cpu_exactly():
    # Bit for bit: DIS flow turns a one-level change of its input into
    # displacements of tens of pixels.
    rng = np.random.default_rng(3)
    img = np.where(rng.uniform(size=(120, 700)) < 0.4, rng.uniform(0.3, 3, (120, 700)), 0)
    img = img.astype(np.float32)
    ref = xp.gaussian_blur_rows(img, 80)
    got = xp.gaussian_blur_rows(xp.cupy.asarray(img), 80).get()
    assert np.array_equal(got, ref), np.abs(got - ref).max()


@needs_gpu
def test_gpu_linear_remap_matches_opencv():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    mx = rng.uniform(-3, 403, (200, 300)).astype(np.float32)
    my = rng.uniform(-3, 303, (200, 300)).astype(np.float32)
    got = xp.remap_linear(xp.cupy.asarray(img), xp.cupy.asarray(mx), xp.cupy.asarray(my)).get()
    assert np.array_equal(got, xp.remap_linear(img, mx, my))


@needs_gpu
def test_gpu_align_band_matches_cpu():
    rng = np.random.default_rng(6)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (256, 102)).astype(np.float32), (0, 0), 2.0)
    texture = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    front = cv2.cvtColor(texture[:, 6:], cv2.COLOR_GRAY2BGR)
    back = cv2.cvtColor(texture[:, :96], cv2.COLOR_GRAY2BGR)
    valid_f = np.ones((256, 96), dtype=bool)
    valid_b = valid_f.copy()
    valid_f[:, 80:] = False
    alpha = np.tile(np.linspace(0, 1, 96, dtype=np.float32), (256, 1))

    pipe = object.__new__(xp.X5Pipeline)
    pipe.st_w, pipe.flow_width = 5760, 3840
    pipe.flow_engine = xp.FlowEngine()
    args = (front, back, valid_f, valid_b, alpha)
    ref = pipe._align_band(*args)
    got = pipe._align_band(*(xp.cupy.asarray(a) for a in args))
    for r, g in zip(ref, got):
        assert np.array_equal(g.get(), r)


@needs_gpu
def test_gpu_projection_matches_numpy_exactly():
    # The strongly cancelling X6 polynomial magnifies any fused multiply-add.
    rays = xp.equirect_rays(1024, 512)
    R = Rotation.from_euler('XYZ', [20, -35, 10], degrees=True).as_matrix()
    X, Y, Z = R @ rays
    ref = xp.mei_forward(X, Y, Z, LENS.xi, LENS.K, lens=LENS)
    got = xp.mei_forward(*(xp.cupy.asarray(a) for a in (X, Y, Z)), LENS.xi, LENS.K, lens=LENS)
    for r, g in zip(ref, got):
        assert np.array_equal(g.get(), r)


@needs_gpu
def test_gpu_remap_tables_match_cpu():
    w, h = 512, 256
    ref = xp.build_equirect_remap(LENS, w, h, rs_rotations=RS_ROTATIONS)
    got = xp.build_equirect_remap(LENS, w, h, rs_rotations=RS_ROTATIONS,
                                  rays=xp.equirect_rays(w, h, xp.cupy))
    assert np.array_equal(got[2].get(), ref[2])
    assert np.abs(got[0].get() - ref[0]).max() < 1e-3
    assert np.abs(got[1].get() - ref[1]).max() < 1e-3


@needs_gpu
def test_gpu_rotate_equirect_matches_cpu():
    img = smooth_sphere_image(512, 256).clip(0, 255).astype(np.uint8)
    R = Rotation.from_euler('XYZ', [61, 7, 20], degrees=True)
    ref = xp.rotate_equirect(img, R, 384, 192)
    # CPU rays: CPU maps, exactly
    rays = xp.equirect_rays(384, 192).astype(np.float32)
    got = xp.rotate_equirect(xp.cupy.asarray(img), R, 384, 192, rays=rays).get()
    assert np.array_equal(got, xp.rotate_equirect(img, R, 384, 192, rays=rays))
    # GPU rays: within a level
    got = xp.rotate_equirect(xp.cupy.asarray(img), R, 384, 192).get()
    assert np.abs(got.astype(int) - ref).max() <= 1


@needs_gpu
def test_gpu_blend_weights_match_cpu():
    front, back = coverage_masks()
    h, w = front.shape
    ref = xp.compute_blend_weights(front, back, w, h)
    got = xp.compute_blend_weights(xp.cupy.asarray(front), xp.cupy.asarray(back), w, h)
    for r, g in zip(ref, got):
        assert np.abs(g.get() - r).max() < 1e-6


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
