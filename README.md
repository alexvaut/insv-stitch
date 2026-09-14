# Insta360 X5 Stitching Pipeline

Linux stitcher for raw Insta360 X5 footage. Reads `.insv` files (two H.265 fisheye streams, IMU samples, a protobuf calibration sidecar) and produces stabilized equirectangular stills or video. No Insta360 Studio required.

PSNR against Studio's own output: 22.5 to 22.9 dB at 7680×3840.

Insta360 Studio is closed source and Windows/macOS only. Reproducing its output on Linux meant working out the `.insv` container, the protobuf-encoded MEI calibration, the IMU axis convention, and the stitching and blending math. The result is a single-file pipeline of about 1,200 lines that matches the reference within a dB.

## Files

- `x5_pipeline.py`. The pipeline.
- `PIPELINE.md`. Architecture: the twelve stages, the MEI model, the blending math, the known limitations.
- `x5_pipeline.md`. Longer notes: container format, IMU axis calibration, rolling shutter, optical flow experiments.
- `old/`. First implementation, plus `FINDINGS.md` with the reverse-engineering notes the rewrite is built on. See `old/README.md`.

## Architecture

Undistortion, rolling shutter and stitching fuse into one backward remap per lens, into a seam frame whose poles are the lens axes; one rotation then turns the stitch to the output:

1. Parse the `.insv` into two H.265 streams and an IMU track.
2. Parse the `.pb` sidecar for MEI calibration (xi = 2.0, 13 distortion coefficients per lens, per-lens extrinsics).
3. Estimate gravity with a gyroscope and accelerometer complementary filter, for horizon-lock leveling with a smoothed heading.
4. Derive per-scanline rolling-shutter rotations from the same gyro integration, 32 SLERP keyframes across the sensor readout.
5. For each seam-frame pixel: ray, rolling shutter rotation, transform into the lens frame, MEI-project, distort, sample. Rays beyond half the calibrated field of view are dropped.
6. Blend on the side of the seam times coverage depth. No hardcoded feather width.
7. Symmetric per-channel gain across the seam.
8. DIS optical flow aligns the two lenses across the seam, for close-range parallax.
9. Turn the stitch to the levelled output. Optional bilateral denoise.

Full treatment in `PIPELINE.md`.

## Install

Python 3.12+, with `ffmpeg` and `ffprobe` on `PATH`.

```bash
uv sync
# or
pip install -e .
```

For the GPU, an NVIDIA card and CuPy. The pipeline stitches on the GPU whenever CuPy finds one; `--device cpu` forces the CPU.

```bash
uv sync --extra gpu
# or
pip install -e .[gpu]
```

## Usage

```bash
# single frame, full resolution, with denoising
uv run python x5_pipeline.py input.insv -o output.jpg -w 7680 --denoise

# full video
uv run python x5_pipeline.py input.insv -o output.mp4 -w 3840 --video

# stabilization off (required on un-calibrated hardware, see below)
uv run python x5_pipeline.py input.insv --no-stab -o output.jpg

# PSNR against a Studio-rendered reference
uv run python x5_pipeline.py input.insv --gt studio_render.mp4 -o output.jpg

# on the CPU, even with a GPU available
uv run python x5_pipeline.py input.insv -o output.mp4 --video --device cpu
```

## Speed

X6 clip, 3840 output stitched at 5760, stabilization and flow on, Core Ultra 9 275HX with an RTX 5090 Laptop GPU:

| | Seconds per frame |
|---|---|
| Before GPU support (CPU) | 24 |
| CPU | 6.7 |
| GPU, one frame | 0.25 |
| GPU, video (300 frames, decode and encode included) | 0.21 |

A video frame takes less than a single frame: each one begins on the GPU while the CPU computes the previous one's DIS flow.

The CPU and the GPU both render the image the pipeline rendered before GPU support, bit for bit. DIS flow needs it: a one-level change of its input moves the flow by tens of pixels. See "Compute Backend" in `PIPELINE.md`.

The `.insv` needs to sit inside the camera's default layout:

```
DCIM/Camera01/VID_xxx_00_001.insv
MISC/Camera01/VID_xxx_00_001.insv.pb
```

## Limitations

IMU calibration is camera-specific. `IMU_CALIBRATION_BY_CAMERA` in `x5_pipeline.py` holds one calibration per model, each from a single unit. The X6 rotation was fitted on the gyroscope and on gravity against Studio renders. The X5 one was solved upstream via Wahba's method on the accelerometer alone; its gyro axes are unverified, so X5 leveling does not fuse the gyro. Unit-to-unit PCB mounting variation will degrade stabilization on other cameras. Pass `--no-stab`, or re-solve against a Studio render from your own hardware.

Close-object parallax. The 30 mm inter-lens baseline shifts close objects between the lenses, 20 px (median) on bike handlebars at 3840 px. DIS flow aligns the lenses across the seam but leaves local warping on the closest objects, and does not match Insta360's learned `ai_stitch_model_v2.ins` on repetitive patterns like fence mesh or foliage.

## License

MIT. See `LICENSE`.
