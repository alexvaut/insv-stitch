# Insta360 X5 Stitching Pipeline: Architecture and Documentation

## Overview

`x5_pipeline.py` is a from-scratch Linux implementation of Insta360 X5 dual-fisheye
stitching. It takes raw `.insv` files (two H.265 fisheye streams + embedded IMU
telemetry) and produces equirectangular 360° output.

**PSNR vs Insta360 Studio ground truth: ~22.5 dB at 1920×960, ~21.8 dB at 7680×3840**

---

## Architecture: Everything Is One Remap

Following the design principle from the Insta360 SDK and Qualcomm stabilization
patent: stabilization, rolling shutter correction, lens undistortion, and stitching
are fused into a **single backward-mapping** per output pixel. No double-resampling.

For every output equirectangular pixel `(u, v)`:

```
1. Convert (u, v) → 3D ray on unit sphere
2. Apply per-scanline stabilization rotation (rolling shutter corrected)
3. Transform ray into fisheye lens's local frame via R_extrinsic⁻¹
4. Project through MEI model with extended distortion → (src_x, src_y)
5. Sample the source fisheye image at (src_x, src_y)
```

---

## Pipeline Stages

### 1. Metadata Extraction (`extract_metadata`)

Extracts calibration and telemetry from the `.insv` file and its `.pb` sidecar:

- **Extended calibration** (preferred): 56-element string from the protobuf sidecar
  (`MISC/Camera01/*.insv.pb`), providing per-lens:
  - MEI mirror parameter (xi = 2.0)
  - Separate fx, fy, cx, cy at sensor resolution (5376×5376)
  - Per-lens yaw, pitch, roll (extrinsic rotation, small corrections)
  - Translation vector (tx, ty, tz, including the 32 mm inter-lens baseline)
  - 13 distortion coefficients: 4 radial (k1-k4) + 2 tangential (p1-p2) + 4 thin prism (s1-s4) + 2 extra

- **Fallback**: Gyroflow lens profile from telemetry-parser (5 coefficients, shared distortion)

- **IMU data**: `normalized_imu()` from telemetry-parser. Gyro (deg/s) and accel (m/s²) at ~1000 Hz (994 Hz measured on the X6).

- **Video metadata**: frame rate and frame count from the stream (`ffprobe`; the metadata rounds 29.97 fps to 30, which drifts frame times against the IMU by 33 ms every 1000 frames), rolling shutter readout time (21.24 ms on the X5, 14.56 ms on the X6)

**Resolution conversion**: Sensor coords (5376) → video coords (3840) via
`fx_video = fx_sensor × 3840/5312` and `cx_video = cx_sensor × 3840/5376`
(with cx_fix=2 for X5: cx is stored halved in Gyroflow convention).

### 2. IMU Stabilization (`compute_stabilization_from_imu`)

Horizon lock from a complementary filter: the gyroscope carries the camera orientation
over short time scales and the accelerometer pulls it slowly towards gravity. Roll and
pitch are corrected. Leveling alone passes the camera's rotation about the vertical
straight through, which on a bike is mostly vibration, so the heading follows the
camera's heading smoothed.

**Convention.** One convention serves leveling and rolling shutter. The camera frame is
X right, Y down, Z forward, and the renderer samples `rays_cam = R @ rays_out`.

- `ImuCalibration`, per camera model: a proper rotation `Q` applied to gyro and accel
  alike, the accelerometer sign, whether the gyro is fused, and the IMU time offset
  (−5 ms on the X6, measured by aligning consecutive frames of our own render with the
  gyro on three ridden segments).
- Frame times are `frame / fps` with the exact stream rate, and the frame reader passes
  frames through unchanged (`-fps_mode passthrough`): after a seek, ffmpeg would
  otherwise repeat the first frame and shift every later one by a frame against the IMU.
- `integrate_gyro`: `C_k`, the camera motion since the first sample,
  `C_{k+1} = exp(-Q ω_k dt) C_k`.
- `leveling_rotation`: the shortest arc taking +Y onto the gravity direction.

**Filter**, run offline and without lag:

1. Express the accelerometer in the gyro-integrated frame, `C_kᵀ (±Q a_k)`, where
   gravity barely moves.
2. Weight each sample by `exp(-((|a_lp| - g) / σ)²)`, floored at 0.05, with `a_lp`
   low-passed at 3 Hz. On a bike |a| is rarely close to g, so samples are down-weighted,
   never dropped.
3. Sum with exponential weights (τ = 16 s) forwards and backwards, add the two passes,
   normalize, and rotate back to each sample. A constant gyro bias tilts the two passes
   in opposite directions and cancels to first order.
4. Heading: measure the levelled output's heading in the gyro-integrated frame, smooth
   it with a Gaussian (σ = 0.5 s), and turn each frame about gravity by the difference.
   Turning about gravity leaves the horizon level.

**X6 results** against an Insta360 Studio render of a ridden clip (not used to fit the
calibration; τ was chosen on it): the horizon is within 0.96° of Studio's (median;
1.73° q90), and moves 0.67° between frames 0.17 s apart while Studio's own vertical
moves 0.81°. The previous implementation smoothed the accelerometer alone over 50
samples (50 ms at 994 Hz) and jumped 21° (median) on the same clip.

### 3. Rolling Shutter Correction (`compute_rs_rotations`)

Per-scanline orientation from the same gyro integration:

1. For each of 32 evenly-spaced scanline positions across the readout:
   - Compute capture time: `t = t_frame + (scanline_frac - 0.5) × readout_time`
   - Camera motion since the frame centre, `C(t) C(t_frame)ᵀ`, interpolated from the
     integrated gyro
   - Compose with the frame's leveling rotation (identity with `--no-stab`)
2. The remap builder interpolates between these 32 orientations per output row via SLERP

Maps are rebuilt every frame whenever the file has IMU data, including with `--no-stab`.

**Impact**: Corrects ~12-18px of displacement during typical handheld motion (+0.65 dB).

### 4. MEI Forward Projection (`mei_forward`)

Projects 3D rays to fisheye pixel coordinates through the extended MEI model:

```
1. Normalize ray to unit sphere: (Xs, Ys, Zs)
2. MEI mirror projection: x = Xs/(Zs + xi), y = Ys/(Zs + xi)
3. Extended distortion:
   radial = 1 + k1·r² + k2·r⁴ + k3·r⁶ + k4·r⁸
   xd = x·radial + 2·p1·x·y + p2·(r² + 2x²) + s1·r² + s2·r⁴
   yd = y·radial + p1·(r² + 2y²) + 2·p2·x·y + s3·r² + s4·r⁴
4. Camera matrix: u = fx·xd + cx, v = fy·yd + cy
```

xi = 2.0 for the X5 (hyperbolic mirror model, supporting >180° FOV).

### 5. Equirectangular Remap (`build_equirect_remap`)

Builds backward-mapping tables (map_x, map_y) for `cv2.remap()`:

1. For each output pixel → compute 3D ray (lon/lat → X,Y,Z with Y-down convention)
2. Apply per-scanline stabilization (RS-corrected, SLERP-interpolated)
3. Transform to lens-local frame via `R_extrinsic.T`
4. Project through MEI → source fisheye coordinates
5. Apply circular fisheye mask (5% margin from image circle edge)

### 6. Blending (`compute_blend_weights`)

**Longitude preference × coverage depth**, no hardcoded parameters:

```
w_front = longitude_pref(col) × distance_from_front_edge(row, col)
w_back  = (1 - longitude_pref(col)) × distance_from_back_edge(row, col)
normalize: w_front, w_back = w_front/(w_front+w_back), ...
```

- **Longitude preference**: Linear ramp from front-primary (|lon| < 75°) to
  back-primary (|lon| > 105°), transitioning at ±90°
- **Coverage depth**: `cv2.distanceTransform`. Pixels from the lens's
  validity boundary. A lens 24 px from its edge naturally gets less weight than
  one 162 px deep.

This handles:
- Hemisphere ownership (longitude)
- Coverage edge smoothness (no hard color steps at fisheye circle boundary)
- Close-object parallax reduction (favors the lens with more central coverage)

### 7. Color Harmonization

**Symmetric per-channel gain** in the blend zone:

1. Compute spatially-varying gain field from the overlap: `gain = front/back` per pixel, Gaussian-blurred (σ=80px)
2. Apply symmetrically: `front *= 1/√gain`, `back *= √gain`
3. Correction strength weighted by `2 × min(w_front, w_back)`. Full at seam center, zero in primary regions.

This preserves each hemisphere's natural exposure while smoothing the transition.

### 8. Optical Flow (Optional, `--flow`)

DIS optical flow for parallax correction in the stitch bands (±15° around ±90° longitude):

1. Extract stitch band crops from both gain-corrected lenses
2. Fill invalid pixels with the other lens's data
3. Compute DIS flow at full resolution
4. Partial warp: each image moves halfway (`×0.5`)
5. DP seam finding + 5-level Laplacian pyramid multi-band blending

**Impact**: Marginal improvement on close objects. The principled blending (coverage depth weighting) handles most parallax. Flow helps with fine structures like fence mesh at ~3m distance.

### 9. Denoising (Optional, `--denoise`)

Post-stitch bilateral filter (`cv2.bilateralFilter(d=9, sigmaColor=40, sigmaSpace=40)`):
- Preserves edges (ground texture sharpness matches GT)
- Smooths flat regions (sky noise matches GT)
- Chosen over NLM, which over-smooths texture

---

## Usage

```bash
# Single frame (all features)
uv run python3 x5_pipeline.py input.insv -o output.jpg -w 7680 --denoise

# Video
uv run python3 x5_pipeline.py input.insv -o output.mp4 -w 3840 --video

# Without stabilization
uv run python3 x5_pipeline.py input.insv -o output.jpg --no-stab

# With optical flow
uv run python3 x5_pipeline.py input.insv -o output.jpg --flow

# Compare to ground truth
uv run python3 x5_pipeline.py input.insv -o output.jpg --gt ground_truth.mp4
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| MEI model with xi=2.0 | Confirmed by Gyroflow source. Hyperbolic mirror handles >180° FOV |
| cx_fix=2 for X5 | Gyroflow convention halves cx; must double for actual optical center |
| Extended 13-coeff model | Protobuf sidecar has per-lens thin prism distortion that standard 5-coeff can't capture |
| Longitude × depth blending | Principled: no hardcoded feather distance, naturally favors central coverage |
| Symmetric gain correction | Prevents one-sided color step at the seam |
| Per-scanline RS correction | 12-18 px displacement during the 21 ms readout, same magnitude as parallax |
| Bilateral over NLM denoising | NLM destroys ground texture; bilateral preserves edges while matching GT noise floor |
| No per-channel CA correction | X5 has zero measurable chromatic aberration; applying CA scaling worsens PSNR |

---

## Known Limitations

1. **IMU calibration**: `IMU_CALIBRATION_BY_CAMERA` holds one calibration per camera
   model, each from a single unit.
   - X6: rotation fitted on the gyroscope against Studio renders, then aligned on
     gravity on a handheld clip, validated on a ridden clip; IMU time offset measured
     on three ridden segments.
   - X5: the upstream matrix, fitted via Wahba's method on the accelerometer alone. Its
     leveling directions are kept exactly, but its gyro axes were never checked, so its
     leveling does not fuse the gyro, and its rolling shutter follows the X6 gyro sign
     unverified.

2. **Close-object parallax**: Objects <3m at the stitch line show ~18px ghosting from
   the 30mm inter-lens baseline. DIS optical flow partially corrects this but can't
   match Insta360's neural flow model (`ai_stitch_model_v2.ins`) on repetitive patterns
   like fence mesh.

3. **Per-frame ffmpeg decode**: Each frame spawns a separate ffmpeg process (~2s overhead).
   Pipe-based batch decoding would improve video throughput.

4. **Rolling shutter timing**: scanline time is indexed by the output equirectangular
   row, not by the sensor row each ray lands on, and both lenses share it. Building one
   SLERP per output row is also slow.

---

## File Dependencies

```
input.insv                          Raw dual-fisheye video
MISC/Camera01/input.insv.pb         Protobuf sidecar (extended calibration)
```

## Python Dependencies

```
numpy, opencv-contrib-python, scipy, telemetry-parser
```
