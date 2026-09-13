"""
Synthetic checks of the IMU convention in x5_pipeline.py (section 3).

A simulated camera rotates on all axes while its accelerometer sees gravity,
vibration and a cornering bump, and its gyro carries a bias. The tests check
that leveling, gyro integration, the complementary filter and the rolling
shutter rotations recover the simulated truth, and that a wrong sign on either
sensor is caught.

Run with `python tests/test_imu.py`, or with pytest.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x5_pipeline as xp  # noqa: E402

RATE = 1000.0
FPS = 30.0
READOUT_MS = 14.56
WORLD_DOWN = np.array([0.0, 1.0, 0.0])
Q = Rotation.from_euler('ZXY', [35, -120, 70], degrees=True).as_matrix()
CALIB = xp.ImuCalibration(Q, accel_sign=-1.0)
Y = np.array([0.0, 1.0, 0.0])


def ang(u, v):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


# Largest gyro bias perpendicular to gravity measured on X6 footage.
def simulate(gyro_bias_dps=0.2, seed=0, yaw_shake_deg=0.0):
    """Times, camera-from-world rotations B, true down, normalized_imu samples."""
    rng = np.random.default_rng(seed)
    ts = np.arange(30000) / RATE - 0.5
    euler = np.stack([
        np.radians(30) * np.sin(2 * np.pi * 0.23 * ts)
        + np.radians(3) * np.sin(2 * np.pi * 4.0 * ts),
        np.radians(20) * np.sin(2 * np.pi * 0.37 * ts + 1.0),
        np.radians(90) * np.sin(2 * np.pi * 0.05 * ts),
    ], axis=-1)
    # Optional 6 Hz shake about the world vertical, the part leveling leaves.
    shake = np.outer(np.radians(yaw_shake_deg) * np.sin(2 * np.pi * 6.0 * ts), WORLD_DOWN)
    B = (Rotation.from_euler('ZXY', euler) * Rotation.from_euler('X', 40, degrees=True)
         * Rotation.from_rotvec(shake))
    down = B.apply(WORLD_DOWN)

    # Exact discrete gyro for the integrator: B_{k+1} B_k^T = exp(-w_k dt).
    w_cam = -(B[1:] * B[:-1].inv()).as_rotvec() / np.diff(ts)[:, None]
    w_cam = np.vstack([w_cam, w_cam[-1:]])

    # Specific force: gravity reaction, vibration, and a 1 m/s^2 cornering
    # acceleration for half a second.
    accel_cam = -xp.GRAVITY * down + rng.normal(0.0, 0.5, down.shape)
    for f in (12.0, 23.0, 41.0):
        accel_cam += (rng.uniform(-1.7, 1.7, 3)
                      * np.sin(2 * np.pi * f * ts[:, None] + rng.uniform(0, 2 * np.pi, 3)))
    bump = (ts > 10.0) & (ts < 10.5)
    accel_cam[bump] += B[bump].apply([1.0, 0.0, 0.0])

    gyro_imu = np.degrees(w_cam) @ Q + gyro_bias_dps
    accel_imu = accel_cam @ Q
    samples = [dict(timestamp_ms=t * 1000.0, gyro=tuple(g), accl=tuple(a))
               for t, g, a in zip(ts, gyro_imu, accel_imu)]
    return ts, B, down, samples


def negate_gyro(samples):
    return [dict(s, gyro=tuple(-np.asarray(s['gyro']))) for s in samples]


def filter_error(samples, down, calib=CALIB):
    orientation = xp.compute_stabilization_from_imu(samples, calib)
    return ang(orientation.down, down)


def rs_error(samples, ts, B, calib=CALIB):
    """Max angle (deg) between rolling-shutter rotations and the true motion."""
    orientation = xp.compute_stabilization_from_imu(samples, calib)
    truth = Slerp(ts, B)
    worst = 0.0
    for frame in (30, 300, 600):
        t_frame = frame / FPS
        rows = xp.compute_rs_rotations(orientation, frame, FPS, READOUT_MS,
                                       apply_leveling=False)
        for frac, R in rows:
            t_row = t_frame + (frac - 0.5) * READOUT_MS / 1000.0
            expected = truth(t_row) * truth(t_frame).inv()
            worst = max(worst, np.degrees((R * expected.inv()).magnitude()))
    return worst


def upstream_leveling(accel_cam):
    """compute_gravity_orientation as it was before the filter."""
    g = accel_cam / np.linalg.norm(accel_cam)
    axis = np.cross(g, Y)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        return Rotation.identity()
    return Rotation.from_rotvec(axis / s * np.arccos(np.clip(np.dot(g, Y), -1, 1)))


def test_leveling_rotation_maps_down_onto_y():
    rng = np.random.default_rng(1)
    for d in list(rng.normal(size=(200, 3))) + [Y, -Y]:
        # arccos is too coarse near 0 for this tolerance; compare vectors
        u = d / np.linalg.norm(d)
        assert np.linalg.norm(xp.leveling_rotation(d).apply(Y) - u) < 1e-9


def test_x5_leveling_matches_upstream():
    calib = xp.IMU_CALIBRATION_BY_CAMERA['Insta360 X5']
    for a in np.random.default_rng(2).normal(size=(200, 3)):
        old = upstream_leveling(xp.IMU_TO_CAM @ a).as_matrix()
        new = xp.leveling_rotation(calib.accel_sign * calib.imu_to_cam @ a).as_matrix()
        assert np.abs(old - new).max() < 1e-9


def test_integrate_gyro_follows_camera():
    ts, B, _, samples = simulate(gyro_bias_dps=0.0)
    t, gyro, _ = xp.imu_arrays(samples, CALIB)
    C = xp.integrate_gyro(t, gyro)
    assert np.degrees((C * (B * B[0].inv()).inv()).magnitude()).max() < 0.01


def test_filter_recovers_gravity():
    _, _, down, samples = simulate()
    e = filter_error(samples, down)
    assert np.median(e) < 1.0 and e.max() < 3.0, (np.median(e), e.max())


def test_rolling_shutter_follows_camera():
    ts, B, _, samples = simulate(gyro_bias_dps=0.0)
    assert rs_error(samples, ts, B) < 0.01


def test_time_offset_realigns_imu_clock():
    ts, B, down, samples = simulate(gyro_bias_dps=0.0)
    late = [dict(s, timestamp_ms=s['timestamp_ms'] + 6.0) for s in samples]
    realigned = xp.ImuCalibration(Q, accel_sign=-1.0, time_offset=0.006)
    assert rs_error(late, ts, B, realigned) < 0.01
    # Rolling shutter is relative to the frame centre, so the offset shows in
    # the gravity estimate: 6 ms late reads where the camera was 6 ms earlier.
    inner = slice(1000, -1000)

    def tilt(calib):
        orientation = xp.compute_stabilization_from_imu(late, calib)
        return np.median(ang(orientation.down_at(ts[inner]), down[inner]))

    assert tilt(realigned) < 0.2 and tilt(CALIB) > 0.25


def test_rolling_shutter_keeps_level():
    # Each row's output +Y follows the frame's estimated gravity through the
    # camera motion since the frame centre (filter accuracy is tested above).
    ts, B, _, samples = simulate()
    orientation = xp.compute_stabilization_from_imu(samples, CALIB)
    truth = Slerp(ts, B)
    for frame in (30, 300, 600):
        t_frame = frame / FPS
        down_frame = orientation.down_at(t_frame)
        for frac, R in xp.compute_rs_rotations(orientation, frame, FPS, READOUT_MS):
            t_row = t_frame + (frac - 0.5) * READOUT_MS / 1000.0
            expected = (truth(t_row) * truth(t_frame).inv()).apply(down_frame)
            assert ang(R.apply(Y), expected) < 0.05


def test_heading_is_smoothed():
    ts, B, _, samples = simulate(yaw_shake_deg=2.0)
    orientation = xp.compute_stabilization_from_imu(samples, CALIB)
    truth = Slerp(ts, B)
    t = np.arange(60, 800) / FPS

    def heading(R):
        # Output forward axis in the world, whose down is +Y.
        f = (truth(t).inv() * R).apply([0.0, 0.0, 1.0])
        return np.unwrap(np.arctan2(f[:, 0], f[:, 2]))

    def shake(h):
        return np.std(h - uniform_filter1d(h, 15))

    smoothed = heading(orientation.leveling_at(t))
    level_only = heading(xp.leveling_rotation(orientation.down_at(t)))
    assert shake(smoothed) < 0.2 * shake(level_only), (shake(smoothed), shake(level_only))
    # Turning about gravity keeps the horizon level.
    down = orientation.down_at(t)
    assert ang(orientation.leveling_at(t).apply(Y), down).max() < 1e-6


def test_sensor_row_remap_matches_single_rotation():
    # With the same rotation at every row, the two-pass rolling shutter remap
    # must reduce to a plain rotated projection.
    lens = xp.MEILensParams(xi=1.0, fx=900.0, fy=900.0, cx=960.0, cy=960.0,
                            width=1920, height=1920)
    R = Rotation.from_euler('XYZ', [20, -35, 10], degrees=True)
    rows = [(float(f), R) for f in np.linspace(0.0, 1.0, 32)]
    by_row = xp.build_equirect_remap(lens, 128, 64, rs_rotations=rows)
    fixed = xp.build_equirect_remap(lens, 128, 64, R_stabilization=R)
    assert np.array_equal(by_row[2], fixed[2])
    assert np.abs(by_row[0] - fixed[0]).max() < 1e-3
    assert np.abs(by_row[1] - fixed[1]).max() < 1e-3


def test_wrong_signs_are_caught():
    ts, B, down, samples = simulate()
    flipped_accel = xp.ImuCalibration(Q, accel_sign=1.0)
    assert np.median(filter_error(samples, down, flipped_accel)) > 90.0
    assert np.median(filter_error(negate_gyro(samples), down)) > 5.0
    _, _, _, unbiased = simulate(gyro_bias_dps=0.0)
    assert rs_error(negate_gyro(unbiased), ts, B) > 0.1


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
