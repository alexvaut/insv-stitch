"""
Synthetic checks of the stitch geometry in x5_pipeline.py (sections 6 to 9):
the field-of-view test, the seam frame, the equirectangular rotation that
turns the stitch to the output, and the parallax alignment across the seam.

Run with `python tests/test_stitch.py`, or with pytest.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x5_pipeline as xp  # noqa: E402

# Radial terms of an X6 front lens: its projection folds back at 114 deg.
X6_LIKE = dict(xi=2.4554, fx=3603.0, fy=3603.0, cx=1920.0, cy=1920.0,
               k1=1.282, k2=-0.9048, k3=0.4568, k4=17.54, width=3840, height=3840)


def rays_at(angles_deg):
    """Unit rays (3, n) at the given angles from +Z, spread in azimuth."""
    t = np.radians(np.asarray(angles_deg, dtype=np.float64))
    a = np.linspace(0.0, 2 * np.pi, len(t), endpoint=False)
    return np.stack([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])


def valid_at(lens, angles_deg):
    rays = rays_at(angles_deg)
    return xp._project_rays(lens, rays, rays.shape[1], 1)[2].ravel()


def test_fold_is_found():
    fold = np.degrees(xp.lens_max_angle(xp.MEILensParams(**X6_LIKE)))
    assert 110.0 < fold < 118.0, fold


def test_rays_behind_the_lens_are_invalid():
    lens = xp.MEILensParams(fov_deg=193.0, **X6_LIKE)
    angles = np.repeat([0.0, 60.0, 95.0, 97.0, 120.0, 150.0, 179.0], 16)
    valid = valid_at(lens, angles)
    assert valid[angles <= 95.0].all()
    assert not valid[angles >= 97.0].any()
    # Without a calibrated FOV, the fold still rules out rays from behind.
    lens.fov_deg = None
    assert not valid_at(lens, angles)[angles >= 120.0].any()


def test_seam_frame_poles_are_the_lens_axes():
    up, forward = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert np.allclose(xp.CAMERA_FROM_SEAM.apply(up), [0.0, 0.0, 1.0])
    assert np.allclose(xp.CAMERA_FROM_SEAM.apply(-up), [0.0, 0.0, -1.0])
    # Longitude 0 on the seam is the camera's down direction.
    assert np.allclose(xp.CAMERA_FROM_SEAM.apply(forward), [0.0, 1.0, 0.0])


def test_front_lens_owns_the_upper_half():
    h, w = 64, 128
    both = np.ones((h, w), dtype=bool)
    w_front, w_back = xp.compute_blend_weights(both, both, w, h)
    assert np.all(w_front[0] == 1.0) and np.all(w_front[-1] == 0.0)
    assert np.allclose(w_front[h // 2], 0.5)
    assert np.allclose(w_front + w_back, 1.0)


def smooth_sphere_image(w, h, R=None):
    """Float image whose channels are linear in the direction R @ ray."""
    rays = xp.equirect_rays(w, h)
    if R is not None:
        rays = R.as_matrix() @ rays
    coeffs = np.array([[0.3, -0.8, 0.5], [0.9, 0.1, -0.4], [-0.2, 0.6, 0.77]])
    return (127.0 + 100.0 * coeffs @ rays).T.reshape(h, w, 3).astype(np.float32)


def test_rotate_equirect_identity():
    img = smooth_sphere_image(256, 128)
    assert np.abs(xp.rotate_equirect(img, Rotation.identity()) - img).max() < 1e-3


def test_rotate_equirect_follows_rotation_through_the_poles():
    w, h = 512, 256
    img = smooth_sphere_image(w, h)
    for R in Rotation.from_euler('XYZ', [[61, 7, 0], [90, 0, 0], [-35, 140, 20]],
                                 degrees=True):
        out = xp.rotate_equirect(img, R)
        err = np.abs(out - smooth_sphere_image(w, h, R)).max(axis=2)
        row, col = np.unravel_index(err.argmax(), err.shape)
        assert err.max() < 0.2, (R.as_euler('XYZ', degrees=True), err.max(), row, col)


def test_align_band_brings_lenses_together():
    # The back lens sees the scene 6 px further right: front(x) = back(x + 6).
    import cv2
    h, w, shift = 256, 96, 6
    noise = np.random.default_rng(4).uniform(0, 255, (h, w + shift)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.0)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    front = cv2.cvtColor(texture[:, shift:], cv2.COLOR_GRAY2BGR)
    back = cv2.cvtColor(texture[:, :w], cv2.COLOR_GRAY2BGR)

    pipe = object.__new__(xp.X5Pipeline)
    pipe.st_w = pipe.flow_width = 3840
    pipe.flow_engine = xp.FlowEngine()
    valid = np.ones((h, w), dtype=bool)
    alpha = np.full((h, w), 0.5, dtype=np.float32)
    fw, bw = pipe._align_band(front, back, valid, valid, alpha)

    inner = (slice(16, -16), slice(16, -16))
    err = lambda a, b: np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))[inner])
    assert err(fw, bw) < 0.25 * err(front, back), (err(fw, bw), err(front, back))
    # Both meet halfway: the front lens moves 3 px right.
    moved = np.roll(front, 3, axis=1)
    assert err(fw, moved) < 0.25 * err(front, moved), (err(fw, moved), err(front, moved))


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
