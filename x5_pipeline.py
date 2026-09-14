"""
x5_pipeline.py: Insta360 X5 stitching pipeline for Linux.

MEI (Unified Omnidirectional) model with extended distortion:
  4 radial (k1-k4) + 2 tangential (p1-p2) + 4 thin prism (s1-s4) + 2 extra
Dual fisheye → equirectangular with IMU-based stabilization.
"""

import numpy as np
import cv2
import os
import queue
import subprocess
import threading
import time
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
import functools
from functools import cache
from pathlib import Path
from string import Template
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, lfilter

import telemetry_parser

try:
    import cupy
except ImportError:  # CPU only: pip install -e .[gpu] for the GPU
    cupy = None


# ============================================================
# 0. Compute Backend
# ============================================================
#
# The per-pixel stages take numpy or CuPy arrays and run where their inputs
# live. On the CPU the heaviest ones split into row blocks across a thread
# pool; on the GPU they run whole. The GPU reproduces the CPU: the Lanczos
# remap bit for bit, everything else to floating-point rounding.

def gpu_available():
    if cupy is None:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def resolve_device(device):
    """Array module for 'cpu', 'gpu', or 'auto' (the GPU when there is one)."""
    if device == 'auto':
        device = 'gpu' if gpu_available() else 'cpu'
    if device == 'cpu':
        return np
    if device != 'gpu':
        raise ValueError(f"device must be auto, cpu or gpu, not {device!r}")
    if not gpu_available():
        raise RuntimeError("device 'gpu' needs CuPy and a CUDA GPU "
                           "(pip install -e .[gpu])")
    return cupy


def array_module(a):
    """numpy or cupy, whichever holds a."""
    return np if cupy is None else cupy.get_array_module(a)


def to_numpy(a):
    return a.get() if cupy is not None and isinstance(a, cupy.ndarray) else a


@cache
def _thread_pool():
    return ThreadPoolExecutor(os.cpu_count())


def parallel_map(fn, *iterables):
    """list(map(fn, ...)) across the thread pool (numpy and OpenCV release the GIL)."""
    return list(_thread_pool().map(fn, *iterables))


def fused(fn):
    """
    fn(xp, *args), made of elementwise arithmetic and comparisons, xp.sqrt,
    xp.maximum and xp.where on its array and float arguments: called as plain
    numpy on numpy arrays, and on CuPy arrays as one CUDA kernel traced from
    the same code. The kernel keeps numpy's order of operations and fuses no
    multiply-adds, so it matches numpy bit for bit (cupy.fuse contracts them,
    which moves a distorted pixel by a float32 step).
    """
    kernels = {}

    @functools.wraps(fn)
    def run(*args):
        if array_module(args[0]) is np:
            return fn(np, *args)
        key = tuple(a.dtype.name if isinstance(a, cupy.ndarray) else 'float64' for a in args)
        if key not in kernels:
            kernels[key] = _Trace(fn, key).kernel()
        return kernels[key](*args)
    return run


class _Trace:
    """Records fn(self, ...) as the statements of a CUDA elementwise kernel."""

    _CTYPES = {'float64': 'double', 'float32': 'float', 'bool': 'bool'}

    def __init__(self, fn, dtypes):
        self.fn, self.dtypes, self.lines = fn, dtypes, []

    def emit(self, ctype, expr):
        name = f't{len(self.lines)}'
        self.lines.append(f'{ctype} {name} = {expr};')
        return _Traced(self, name, ctype)

    def sqrt(self, a):
        return self.emit('double', f'sqrt({_Traced.c(a)})')

    def maximum(self, a, b):
        return self.emit('double', f'fmax({_Traced.c(a)}, {_Traced.c(b)})')

    def where(self, cond, a, b):
        ctype = _Traced.promote(a, b)
        return self.emit(ctype, f'{_Traced.c(cond)} ? ({ctype}){_Traced.c(a)} : ({ctype}){_Traced.c(b)}')

    def kernel(self):
        args = [_Traced(self, f'a{i}', self._CTYPES[d]) for i, d in enumerate(self.dtypes)]
        outs = self.fn(self, *args)
        outs = outs if isinstance(outs, tuple) else (outs,)
        dtype_of = {c: d for d, c in self._CTYPES.items()}
        return cupy.ElementwiseKernel(
            ', '.join(f'{d} a{i}' for i, d in enumerate(self.dtypes)),
            ', '.join(f'{dtype_of[o.ctype]} o{j}' for j, o in enumerate(outs)),
            '\n'.join(self.lines + [f'o{j} = {o.name};' for j, o in enumerate(outs)]),
            name=self.fn.__name__.strip('_'), options=('--fmad=false',))


class _Traced:
    """A C variable in a _Trace: arithmetic on it appends statements."""

    def __init__(self, trace, name, ctype):
        self.trace, self.name, self.ctype = trace, name, ctype

    @staticmethod
    def c(v):
        return v.name if isinstance(v, _Traced) else repr(float(v))

    @staticmethod
    def promote(*vs):
        types = {v.ctype if isinstance(v, _Traced) else 'double' for v in vs}
        return next(t for t in ('double', 'float', 'bool') if t in types)

    def _binary(self, other, op, reverse=False, ctype=None):
        a, b = (other, self) if reverse else (self, other)
        return self.trace.emit(ctype or self.promote(a, b), f'{self.c(a)} {op} {self.c(b)}')

    __add__ = lambda s, o: s._binary(o, '+')
    __radd__ = lambda s, o: s._binary(o, '+', True)
    __sub__ = lambda s, o: s._binary(o, '-')
    __rsub__ = lambda s, o: s._binary(o, '-', True)
    __mul__ = lambda s, o: s._binary(o, '*')
    __rmul__ = lambda s, o: s._binary(o, '*', True)
    __truediv__ = lambda s, o: s._binary(o, '/')
    __rtruediv__ = lambda s, o: s._binary(o, '/', True)
    __gt__ = lambda s, o: s._binary(o, '>', ctype='bool')
    __lt__ = lambda s, o: s._binary(o, '<', ctype='bool')
    __ge__ = lambda s, o: s._binary(o, '>=', ctype='bool')
    __le__ = lambda s, o: s._binary(o, '<=', ctype='bool')
    __and__ = lambda s, o: s._binary(o, '&&', ctype='bool')
    __or__ = lambda s, o: s._binary(o, '||', ctype='bool')
    __neg__ = lambda s: s.trace.emit(s.ctype, f'-{s.name}')
    __invert__ = lambda s: s.trace.emit('bool', f'!{s.name}')


def _launch(kernel, n, *args):
    """Run a CUDA kernel over n items, in blocks of 256 threads (1024 fail to launch under WDDM)."""
    kernel(((n + 255) // 256,), (256,), args)


def map_rows(fn, rays, eq_w, eq_h, min_rows=32):
    """
    fn(rays, first, stop) -> tuple of (stop - first, eq_w) arrays, for the
    rays (3, eq_h * eq_w) of a grid's rows first:stop, over the whole grid: in
    one call on the GPU, in row blocks across the thread pool on the CPU.
    """
    if array_module(rays) is not np:
        return fn(rays, 0, eq_h)
    n = max(1, min(os.cpu_count(), eq_h // min_rows))
    bounds = np.linspace(0, eq_h, n + 1).astype(int).tolist()
    parts = parallel_map(
        lambda first, stop: fn(np.ascontiguousarray(rays[:, first * eq_w:stop * eq_w]),
                               first, stop),
        bounds[:-1], bounds[1:])
    return tuple(np.concatenate(p) for p in zip(*parts))


# cv2.remap with INTER_LANCZOS4 on the GPU, reproducing OpenCV bit for bit on
# 8-bit images: map coordinates rounded to 1/32 px, OpenCV's 8x8 weight tables
# (15-bit fixed point for 8-bit images, float otherwise) and its border rules.
INTER_TAB_SIZE = 32
INTER_REMAP_COEF_SCALE = 1 << 15


def _lanczos4_1d(x):
    """OpenCV's interpolateLanczos4 at sub-pixel offset x, in its float32 steps."""
    s45 = 0.70710678118654752440084436210485
    cs = [(1, 0), (-s45, -s45), (0, 1), (s45, -s45),
          (-1, 0), (s45, s45), (0, -1), (-s45, s45)]
    y0 = -(float(x) + 3) * np.pi * 0.25
    s0, c0 = np.sin(y0), np.cos(y0)
    coeffs = np.zeros(8, np.float32)
    total = np.float32(0)
    for i in range(8):
        d = np.float32(x + np.float32(3 - i))
        if abs(d) >= np.float32(1e-6):
            y = -float(d) * np.pi * 0.25
            coeffs[i] = np.float32((cs[i][0] * s0 + cs[i][1] * c0) / (y * y))
        else:
            coeffs[i] = np.float32(1e30)
        total = np.float32(total + coeffs[i])
    return coeffs * (np.float32(1.0) / total)


@cache
def lanczos4_tables():
    """OpenCV's 2D Lanczos4 weights per 1/32 px offset: float32 and fixed point."""
    tab1 = [_lanczos4_1d(np.float32(i) * np.float32(1.0 / INTER_TAB_SIZE))
            for i in range(INTER_TAB_SIZE)]
    ftab = np.empty((INTER_TAB_SIZE ** 2, 64), np.float32)
    itab = np.empty((INTER_TAB_SIZE ** 2, 64), np.int32)
    for i in range(INTER_TAB_SIZE):
        for j in range(INTER_TAB_SIZE):
            v = np.outer(tab1[i], tab1[j]).astype(np.float32)
            iv = np.rint(v * np.float32(INTER_REMAP_COEF_SCALE)).astype(np.int32)
            # Fixed-point weights must sum to one exactly: OpenCV takes the
            # difference from the largest or smallest of the four central ones.
            diff = int(iv.sum()) - INTER_REMAP_COEF_SCALE
            if diff:
                lo = hi = (4, 4)
                for k in ((4, 4), (4, 5), (5, 4), (5, 5)):
                    if iv[k] < iv[lo]:
                        lo = k
                    elif iv[k] > iv[hi]:
                        hi = k
                iv[hi if diff < 0 else lo] -= diff
            ftab[i * INTER_TAB_SIZE + j] = v.ravel()
            itab[i * INTER_TAB_SIZE + j] = iv.ravel()
    return ftab, itab


_LANCZOS_SOURCE = Template(r'''
extern "C" __global__
void remap_lanczos4(const $T* src, const int sh, const int sw, const int cn,
                    const float* map_x, const float* map_y, $T* dst,
                    const int n, const $W* tab, const int replicate)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int X = (int)rint(map_x[i] * 32.0f), Y = (int)rint(map_y[i] * 32.0f);
    int sx = (X >> 5) - 3, sy = (Y >> 5) - 3;
    const $W* w = tab + ((Y & 31) * 32 + (X & 31)) * 64;
    $T* d = dst + i * cn;
    if (sx >= 0 && sx + 8 <= sw && sy >= 0 && sy + 8 <= sh) {
        for (int k = 0; k < cn; k++) {
            $W sum = 0;
            for (int r = 0; r < 8; r++) {
                const $T* s = src + ((sy + r) * sw + sx) * cn + k;
                for (int c = 0; c < 8; c++) sum += s[c * cn] * w[r * 8 + c];
            }
            d[k] = $CAST;
        }
        return;
    }
    if (!replicate && (sx >= sw || sx + 8 <= 0 || sy >= sh || sy + 8 <= 0)) {
        for (int k = 0; k < cn; k++) d[k] = 0;
        return;
    }
    for (int k = 0; k < cn; k++) {
        $W sum = 0;
        for (int r = 0; r < 8; r++) {
            int y = sy + r;
            if (replicate) y = y < 0 ? 0 : (y >= sh ? sh - 1 : y);
            else if (y < 0 || y >= sh) continue;
            for (int c = 0; c < 8; c++) {
                int x = sx + c;
                if (replicate) x = x < 0 ? 0 : (x >= sw ? sw - 1 : x);
                else if (x < 0 || x >= sw) continue;
                sum += src[(y * sw + x) * cn + k] * w[r * 8 + c];
            }
        }
        d[k] = $CAST;
    }
}
''')

# Per image dtype: pixel type, weight type, and the cast of the weighted sum.
_LANCZOS_TYPES = {
    'uint8': dict(T='unsigned char', W='int',
                  CAST='(unsigned char)(sum < -16384 ? 0 : '
                       '((sum + 16384) >> 15) > 255 ? 255 : (sum + 16384) >> 15)'),
    'float32': dict(T='float', W='float', CAST='sum'),
}


@cache
def _lanczos_kernel(dtype):
    table = lanczos4_tables()[0 if dtype == 'float32' else 1]
    return (_cuda_kernel(_LANCZOS_SOURCE.substitute(_LANCZOS_TYPES[dtype]), 'remap_lanczos4'),
            cupy.asarray(table))


def remap_lanczos(image, map_x, map_y, replicate=False):
    """
    cv2.remap(image, map_x, map_y, INTER_LANCZOS4) with a zero border, or a
    replicated one, on the device holding image.
    """
    if array_module(image) is np:
        border = cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT
        return cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4,
                         borderMode=border, borderValue=0)
    kernel, table = _lanczos_kernel(image.dtype.name)
    return _remap_gpu(kernel, image, map_x, map_y, table, np.int32(replicate))


def _remap_gpu(kernel, image, map_x, map_y, *extra):
    """Launch a remap kernel(src, sh, sw, cn, map_x, map_y, dst, n, *extra)."""
    image = cupy.ascontiguousarray(image)
    cn = image.shape[2] if image.ndim == 3 else 1
    out = cupy.empty(map_x.shape + image.shape[2:], image.dtype)
    _launch(kernel, map_x.size,
            image, np.int32(image.shape[0]), np.int32(image.shape[1]), np.int32(cn),
            cupy.ascontiguousarray(cupy.asarray(map_x, cupy.float32)),
            cupy.ascontiguousarray(cupy.asarray(map_y, cupy.float32)),
            out, np.int32(map_x.size), *extra)
    return out


# cv2.remap with INTER_LINEAR and a replicated border on 8-bit images. OpenCV
# 5 interpolates float coordinates in float and rounds; its SIMD code fuses
# multiply-adds, as the CUDA compiler does here by default.
_LINEAR_SOURCE = r'''
extern "C" __global__
void remap_linear(const unsigned char* src, const int sh, const int sw, const int cn,
                  const float* map_x, const float* map_y, unsigned char* dst,
                  const int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float fx = map_x[i], fy = map_y[i];
    int sx = (int)floor(fx), sy = (int)floor(fy);
    float ax = fx - sx, ay = fy - sy;
    int x0 = sx < 0 ? 0 : (sx >= sw ? sw - 1 : sx);
    int x1 = sx + 1 < 0 ? 0 : (sx + 1 >= sw ? sw - 1 : sx + 1);
    int y0 = sy < 0 ? 0 : (sy >= sh ? sh - 1 : sy);
    int y1 = sy + 1 < 0 ? 0 : (sy + 1 >= sh ? sh - 1 : sy + 1);
    for (int k = 0; k < cn; k++) {
        float s00 = src[(y0 * sw + x0) * cn + k], s01 = src[(y0 * sw + x1) * cn + k];
        float s10 = src[(y1 * sw + x0) * cn + k], s11 = src[(y1 * sw + x1) * cn + k];
        float top = s00 + ax * (s01 - s00), bottom = s10 + ax * (s11 - s10);
        float v = top + ay * (bottom - top);
        dst[i * cn + k] = (unsigned char)rint(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}
'''


def remap_linear(image, map_x, map_y):
    """cv2.remap(image, map_x, map_y, INTER_LINEAR, BORDER_REPLICATE) of an 8-bit image, on its device."""
    if array_module(image) is np:
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return _remap_gpu(_cuda_kernel(_LINEAR_SOURCE, 'remap_linear'), image, map_x, map_y)


@cache
def _cuda_kernel(source, name, options=()):
    return cupy.RawModule(code=source, options=options).get_function(name)


# The Gaussian blur of gaussian_blur_rows, as OpenCV's separable filter runs it
# on float images: along each row every tap is added in turn, along each
# column the centre first, then each pair of rows equally far above and below.
# Columns up to the last multiple of 8 go through SIMD code that fuses the
# multiply-adds, the rest through scalar code that does not. Compiled without
# contraction, so only the explicit fma() fuses.
_BLUR_SOURCE = r'''
extern "C" __global__
void blur_rows(const float* src, const int h, const int w, const float* k,
               const int r, float* dst, const int simd_end)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= h * w) return;
    int y = i / w, x = i % w;
    float s = 0;
    for (int j = 0; j <= 2 * r; j++) {
        int c = x + j - r;                                  // reflect 101
        c = c < 0 ? -c : (c >= w ? 2 * w - 2 - c : c);
        s = x < simd_end ? fma(src[y * w + c], k[j], s) : s + src[y * w + c] * k[j];
    }
    dst[i] = s;
}

extern "C" __global__
void blur_cols(const float* src, const int h, const int w, const float* k,
               const int r, float* dst, const int simd_end)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= h * w) return;
    int y = i / w, x = i % w;
    float s = src[i] * k[r];
    for (int j = 1; j <= r; j++) {                          // zero beyond
        float a = (y + j < h ? src[(y + j) * w + x] : 0.0f)
                + (y - j >= 0 ? src[(y - j) * w + x] : 0.0f);
        s = x < simd_end ? fma(a, k[r + j], s) : s + a * k[r + j];
    }
    dst[i] = s;
}
'''


def gaussian_blur_rows(image, sigma):
    """
    cv2.GaussianBlur(image, (0, 0), sigma) of a float32 image as if it were
    one band of a taller image that is zero above and below: columns reflect
    at the sides (OpenCV's default border), rows beyond the band count as zero.
    """
    ksize = round(sigma * 4 * 2 + 1) | 1      # OpenCV's size for float images
    kernel = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
    if array_module(image) is np:
        one = np.ones((1, 1), np.float32)
        out = cv2.sepFilter2D(image, -1, kernel, one, borderType=cv2.BORDER_REFLECT_101)
        return cv2.sepFilter2D(out, -1, one, kernel, borderType=cv2.BORDER_CONSTANT)
    h, w = image.shape
    kernel, tmp, out = cupy.asarray(kernel.ravel()), cupy.empty_like(image), cupy.empty_like(image)
    for name, src, dst in (('blur_rows', cupy.ascontiguousarray(image), tmp),
                           ('blur_cols', tmp, out)):
        _launch(_cuda_kernel(_BLUR_SOURCE, name, ('--fmad=false',)), image.size,
                src, np.int32(h), np.int32(w), kernel, np.int32(ksize // 2), dst,
                np.int32(w - w % 8))
    return out


def overlap_rows(mask):
    """(first, stop) of the rows holding any True pixel of a 2D mask, or None."""
    rows = np.flatnonzero(to_numpy(mask.any(axis=1)))
    return None if rows.size == 0 else (int(rows[0]), int(rows[-1]) + 1)


# ============================================================
# 1. Data Structures
# ============================================================

@dataclass
class MEILensParams:
    """MEI unified spherical camera model with extended distortion."""
    xi: float               # Mirror parameter (2.0 for X5)
    fx: float               # Focal length x (video resolution)
    fy: float               # Focal length y (video resolution)
    cx: float               # Principal point x (video resolution, cx_fix applied)
    cy: float               # Principal point y (video resolution)
    # Radial distortion (up to r^8)
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0         # Extended: r^8 term
    # Tangential distortion
    p1: float = 0.0
    p2: float = 0.0
    # Thin prism distortion
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    # Extrinsics
    R_extrinsic: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_extrinsic: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Sensor
    width: int = 3840
    height: int = 3840
    # Calibrated field of view (deg), None when the calibration has none
    fov_deg: float | None = None

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def D(self):
        """Legacy 5-element distortion for backward compat."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class ImuCalibration:
    """
    How normalized_imu() maps into the camera frame (see section 3).

    imu_to_cam: proper rotation Q, applied to gyro and accel alike.
    accel_sign: -1 when Q @ accel points up at rest (specific force), +1 when
        it points down.
    fuse_gyro: False when the gyro axes are unverified for this model. The
        leveling then low-passes the accelerometer alone; rolling shutter
        still uses the gyro.
    time_offset: seconds; the IMU sample stamped t + time_offset belongs to
        the video frame at t.
    """
    imu_to_cam: np.ndarray
    accel_sign: float
    fuse_gyro: bool = True
    time_offset: float = 0.0


@dataclass
class ImuOrientation:
    """Camera motion and gravity at every IMU sample (see section 3)."""
    ts: np.ndarray            # (n,) seconds from the first video frame
    motion: Slerp             # C(t): world-fixed directions, camera(ts[0]) -> camera(t)
    down: np.ndarray          # (n, 3) unit gravity direction in the camera frame
    heading: np.ndarray       # (n,) turn about down (rad) smoothing the heading

    def motion_at(self, t):
        return self.motion(np.clip(t, self.ts[0], self.ts[-1]))

    def down_at(self, t):
        d = np.stack([np.interp(t, self.ts, self.down[:, j]) for j in range(3)],
                     axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def leveling_at(self, t):
        """Camera-from-output rotation: level horizon, smoothed heading."""
        d = self.down_at(t)
        turn = np.asarray(np.interp(t, self.ts, self.heading))[..., None]
        return Rotation.from_rotvec(d * turn) * leveling_rotation(d)


@dataclass
class PipelineMetadata:
    lens_params: list         # [MEILensParams front, MEILensParams back]
    imu_samples: list         # normalized_imu output
    fps: float
    frame_count: int
    frame_readout_time: float # ms
    width: int
    height: int
    offset: list
    imu_calib: ImuCalibration = None  # resolved from the camera model


# ============================================================
# 2. Metadata Extraction
# ============================================================

def _find_pb_path(insv_path: str) -> str | None:
    """Find the .pb protobuf sidecar file for an .insv file."""
    insv = Path(insv_path)
    # Look in MISC/Camera01/ relative to the DCIM/Camera01/ path
    misc_dir = insv.parent.parent.parent / 'MISC' / 'Camera01'
    pb_name = insv.name + '.pb'
    pb_path = misc_dir / pb_name
    if pb_path.exists():
        return str(pb_path)
    return None


# Every .insv ends with this ASCII magic, preceded by the trailer size.
INSV_TRAILER_MAGIC = b'8db42d694ccc418790edff439fe026bf'


def _scan_calibration_string(text: str) -> list[float] | None:
    """
    Scan decoded text for the extended calibration string.

    Format is "<num_lenses>_<float>_<float>_...", 27 elements per lens plus a
    leading count. Returns the parsed float values, or None if not found.
    """
    # Find calibration strings: "2_<numbers...>" with >40 elements
    for match in re.finditer(r'2_[\d]+\.[\d]+_', text):
        start = match.start()
        # Extract the full numeric string
        chunk = text[start:]
        parts = []
        for token in chunk.split('_'):
            try:
                parts.append(float(token))
            except ValueError:
                break
        if len(parts) >= 55:  # Extended format has 56 elements
            return parts

    return None


def _parse_extended_calibration(pb_path: str) -> list[float] | None:
    """
    Extract the extended 56-element calibration from a .pb sidecar (X5).
    Returns the parsed float values, or None if not found.
    """
    with open(pb_path, 'rb') as f:
        data = f.read()

    text = data.decode('latin-1')

    # Find base64-encoded content
    b64_match = re.search(r'[A-Za-z0-9+/=]{100,}', text)
    if not b64_match:
        return None

    decoded = base64.b64decode(b64_match.group()).decode('latin-1')
    return _scan_calibration_string(decoded)


def _read_insv_trailer_calibration(insv_path: str) -> list[float] | None:
    """
    Extract the extended calibration from the .insv metadata trailer (X6).

    The X6 writes no .pb sidecar, but stores the same 56-element string as
    plain text inside the trailer. The file ends with:
        [...trailer...][extra_size uint32 LE][version uint32 LE][magic 32 ASCII]
    """
    with open(insv_path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size < 40:
            return None

        f.seek(file_size - 32)
        if f.read(32) != INSV_TRAILER_MAGIC:
            return None

        f.seek(file_size - 40)
        extra_size = int.from_bytes(f.read(4), 'little')
        if not 0 < extra_size <= file_size:
            return None

        f.seek(file_size - extra_size)
        trailer = f.read(extra_size)

    return _scan_calibration_string(trailer.decode('latin-1'))


def _sensor_to_video(fx, fy, cx, cy, sensor_w, crop_dst, video_w):
    """
    Convert calibration values from sensor resolution to video resolution.

    The per-lens sensor area (sensor_w) is centre-cropped to crop_dst
    (window_crop_info), then scaled down to video_w. Focal lengths follow the
    scaling only; the principal point is shifted by the crop first.

    X5: 5376 -> 5312 -> 3840.  X6: 7744 -> 7680 -> 3840.
    """
    crop_offset = (sensor_w - crop_dst) / 2.0
    scale = video_w / crop_dst
    cx_v = (cx - crop_offset) * scale
    cy_v = (cy - crop_offset) * scale
    fx_v = fx * scale
    fy_v = fy * scale
    return fx_v, fy_v, cx_v, cy_v


def extract_metadata(insv_path: str) -> PipelineMetadata:
    """
    Extract all metadata from .insv file.
    Prefers extended protobuf calibration (13-coeff model) if available,
    falls back to offset_v3 / Gyroflow profile.
    """
    tp = telemetry_parser.Parser(insv_path)
    telem = tp.telemetry()
    item = telem[0]

    meta = item['Default']['Metadata']
    # The X6 exposes no 'Lens' block at all; the X5 keeps fisheye_params there.
    lens_data = item.get('Lens', {}).get('Data', {})
    dim = meta['dimension']
    width, height = dim['x'], dim['y']
    offset = meta['offset']

    # Extended calibration: .pb sidecar (X5) first, then the .insv trailer (X6).
    pb_path = _find_pb_path(insv_path)
    ext = _parse_extended_calibration(pb_path) if pb_path else None
    calib_source = 'protobuf sidecar'
    if not ext:
        ext = _read_insv_trailer_calibration(insv_path)
        calib_source = '.insv trailer'

    if ext and len(ext) >= 55:
        # ---- Extended 56-element calibration (best available) ----
        # Format: [0]=lens count, then 27 elements per lens:
        #   [+0]=xi, [+1]=fx, [+2]=fy, [+3]=cx, [+4]=cy,
        #   [+5]=yaw, [+6]=pitch, [+7]=see note below,
        #   [+8..+10]=tx,ty,tz (old/FINDINGS.md reads these as a Rodrigues vector),
        #   [+11..+23]=13 distortion coefficients,
        #   [+24]=full dual-fisheye width, [+25]=per-lens width, [+26]=FOV deg
        # Total is 1 + 2*27 = 55 elements.
        #
        # [+7] is ~89.5-90 on both the X5 and the X6, and on both lenses of one
        # body. The code labelled it roll, old/FINDINGS.md labelled it half_fov.
        # Neither is obvious: [+26] carries the FOV (193 on X6), so a half-FOV
        # would read 96.5, not 89.5. A ~90 degree roll does match the X6's
        # cam_posture=CameraRotate90. Set INSV_CALIB_ROLL=1 to apply it as roll;
        # the default keeps upstream behaviour and ignores it.
        apply_roll = os.environ.get('INSV_CALIB_ROLL') == '1'

        # Sensor geometry read from the calibration and window_crop_info rather
        # than hardcoded: X5 is 5376->5312, X6 is 7744->7680, both down to 3840.
        sensor_w = ext[1 + 25]
        crop = meta.get('window_crop_info') or {}
        crop_dst = float(crop.get('dst_width') or sensor_w)

        lenses = []
        for lens_idx in range(2):
            s = 1 + lens_idx * 27
            xi = ext[s]
            fx_s, fy_s = ext[s+1], ext[s+2]
            cx_s, cy_s = ext[s+3], ext[s+4]
            yaw, pitch, roll = ext[s+5], ext[s+6], ext[s+7]
            tx, ty, tz = ext[s+8], ext[s+9], ext[s+10]
            coeffs = ext[s+11:s+24]

            # Convert to video resolution
            if lens_idx == 1:
                cx_s -= sensor_w  # Remove concatenated offset
            fx_v, fy_v, cx_v, cy_v = _sensor_to_video(
                fx_s, fy_s, cx_s, cy_s, sensor_w, crop_dst, width)

            # Extrinsic rotation (small yaw/pitch corrections, plus roll if enabled)
            order = 'YXZ' if apply_roll else 'YX'
            angles = [yaw, pitch, roll] if apply_roll else [yaw, pitch]
            if lens_idx == 0:
                R = Rotation.from_euler(order, angles, degrees=True).as_matrix()
            else:
                R_back = Rotation.from_euler('Y', 180, degrees=True)
                R_corr = Rotation.from_euler(order, angles, degrees=True)
                R = (R_back * R_corr).as_matrix()

            lens = MEILensParams(
                xi=xi, fx=fx_v, fy=fy_v, cx=cx_v, cy=cy_v,
                k1=coeffs[0], k2=coeffs[1], k3=coeffs[2], k4=coeffs[3],
                p1=coeffs[5], p2=coeffs[6],
                s1=coeffs[7], s2=coeffs[8], s3=coeffs[9], s4=coeffs[10],
                R_extrinsic=R, t_extrinsic=np.array([tx, ty, tz]),
                width=width, height=height, fov_deg=ext[s+26],
            )
            lenses.append(lens)

        print(f"  Using extended calibration from {calib_source} "
              f"(13 coefficients per lens, xi={lenses[0].xi:.5f}, "
              f"sensor {sensor_w:.0f}->{crop_dst:.0f}->{width}"
              f"{', roll applied' if apply_roll else ''})")
    else:
        # ---- Fallback: Gyroflow / offset_v3 (5-coeff model) ----
        if not lens_data or not offset:
            raise RuntimeError(
                f"No lens calibration available for {insv_path}.\n"
                "  - nothing in a .pb sidecar, nothing in the .insv trailer\n"
                "  - no Gyroflow offset_v3 fallback either (Lens block "
                f"{'present' if lens_data else 'missing'}, offset "
                f"{'present' if offset else 'empty'})\n"
                f"  camera reported as: {meta.get('camera_type', 'unknown')}"
            )
        fp = lens_data['fisheye_params']
        dc = fp['distortion_coeffs']
        cm = fp['camera_matrix']

        xi = dc[5]
        k1, k2, k3 = dc[0], dc[1], dc[2]
        p1, p2 = dc[3], dc[4]
        fx_0 = cm[0][0]
        fy_0 = cm[1][1]
        cx_0 = cm[0][2] * 2.0  # cx_fix=2 for X5
        cy_0 = cm[1][2]

        yaw_0, pitch_0 = offset[4], offset[5]
        R0 = Rotation.from_euler('YX', [yaw_0, pitch_0], degrees=True).as_matrix()
        lens_front = MEILensParams(
            xi=xi, fx=fx_0, fy=fy_0, cx=cx_0, cy=cy_0,
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            R_extrinsic=R0, width=width, height=height,
        )

        fx_ratio = offset[7] / offset[1]
        cx_1_offset = offset[8] - offset[14]
        yaw_1, pitch_1 = offset[10], offset[11]
        R_back = Rotation.from_euler('Y', 180, degrees=True)
        R1 = (R_back * Rotation.from_euler('YX', [yaw_1, pitch_1], degrees=True)).as_matrix()
        lens_back = MEILensParams(
            xi=xi, fx=fx_0 * fx_ratio, fy=fy_0 * fx_ratio,
            cx=cx_0 * (cx_1_offset / offset[2]),
            cy=cy_0 * (offset[9] / offset[3]),
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
            R_extrinsic=R1, width=width, height=height,
        )
        lenses = [lens_front, lens_back]
        print(f"  Using Gyroflow/offset_v3 calibration (5 coefficients)")

    imu_samples = tp.normalized_imu()
    # The X5 reports this under Lens.Data; the X6 only has Default.Metadata.
    frame_readout_time = lens_data.get('frame_readout_time')
    if frame_readout_time is None:
        frame_readout_time = float(meta.get('rolling_shutter_time') or 21.24)

    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_frames,r_frame_rate',
           '-of', 'default=nw=1', insv_path]
    stream = {}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        stream = dict(line.split('=', 1) for line in r.stdout.split())
    except Exception:
        pass
    # The metadata rounds 30000/1001 fps to 30, which would drift frame times
    # against the IMU by 33 ms every 1000 frames; the stream rate is exact.
    try:
        fps_val = float(Fraction(stream['r_frame_rate']))
    except (KeyError, ValueError, ZeroDivisionError):
        fps_val = float(meta.get('frame_rate', 30))
    try:
        frame_count = int(stream['nb_frames'])
    except (KeyError, ValueError):
        # HEVC streams often carry no nb_frames; the camera records the count.
        frame_count = int(meta.get('total_frames') or 0)

    camera_type = meta.get('camera_type', '')
    imu_calib = IMU_CALIBRATION_BY_CAMERA.get(camera_type)
    if imu_calib is None:
        print(f"  WARNING: no IMU calibration for {camera_type!r}, falling back "
              "to the X5 one. Stabilization will be wrong; use --no-stab or "
              "re-solve against a Studio render.")
        imu_calib = IMU_CALIBRATION_BY_CAMERA['Insta360 X5']
    elif not imu_calib.fuse_gyro:
        print(f"  WARNING: the {camera_type} gyro axes are unverified; leveling "
              "uses the accelerometer alone.")

    return PipelineMetadata(
        lens_params=lenses,
        imu_samples=imu_samples, fps=fps_val,
        frame_count=frame_count,
        frame_readout_time=frame_readout_time,
        width=width, height=height, offset=offset,
        imu_calib=imu_calib,
    )


# ============================================================
# 3. IMU Stabilization
# ============================================================
#
# One convention serves leveling and rolling shutter. The camera frame is
# X right, Y down, Z forward, and the renderer samples the camera along
# rays_cam = R @ rays_out, so every rotation here maps output directions into
# the camera frame.
#
#   w_k      camera angular velocity, rad/s, right-handed: Q @ gyro_k
#   C_k      camera motion since the first IMU sample: a world-fixed direction
#            has camera coordinates v_k = C_k @ v_0, C_{k+1} = exp(-w_k dt_k) C_k
#   down_k   gravity direction in the camera frame
#   R_level  shortest arc taking +Y onto down, which levels the horizon, then
#            turned about down so the output heading follows the camera's
#            heading smoothed
#   R_row    C(t_row) C(t_frame)^T R_level(t_frame), for a scanline read at t_row

GRAVITY = 9.81

# Half turn about Y.
_FLIP_XZ = np.diag([-1.0, 1.0, -1.0])

# IMU-to-camera rotation for the X5, as solved upstream.
# Solved via Wahba's method from GT-aligned gravity vectors across two videos
# on one specific X5 unit. The IMU sits at ~120 degrees from identity
# on the PCB.
#
# Specific to one camera. Unit-to-unit PCB mounting variation will degrade
# stabilization on other X5s. Use --no-stab, or re-solve against a Studio
# render from your own hardware.
#
# It was fitted for a leveling that took IMU_TO_CAM @ accel onto +Y. That arc
# is exactly the arc taking +Y onto _FLIP_XZ @ IMU_TO_CAM @ accel, which is what
# the X5 calibration below uses, so X5 leveling directions are unchanged. Its
# gyro axes were never checked against footage.
IMU_TO_CAM = np.array([
    [-0.5678,  0.4608, -0.6821],
    [ 0.7642,  0.6031, -0.2287],
    [ 0.3060, -0.6511, -0.6946],
], dtype=np.float64)

# IMU-to-camera rotation for the X6. Each frame of our own unlevelled render is
# aligned photometrically to an 8K equirectangular Insta360 Studio render,
# which gives the true up direction in the camera frame. The rotation is first
# solved so that the gyro, integrated over 0.17 s, carries each up direction
# onto the next. Small rotations barely constrain its tilt, so it is then
# turned by 6.5 deg to align the filtered gravity with the measured vertical
# on a handheld clip. On a ridden clip left out of that fit, the horizon is
# within 0.96 deg of Studio's (median; 1.73 deg q90), and the gyro predicts
# rotations over 0.17 to 1 s within 0.2 deg (median).
IMU_TO_CAM_X6 = np.array([
    [ 0.018092,  0.009943,  0.999787],
    [-0.013182,  0.999866, -0.009705],
    [-0.999749, -0.013003,  0.018220],
], dtype=np.float64)

IMU_CALIBRATION_BY_CAMERA = {
    'Insta360 X5': ImuCalibration(_FLIP_XZ @ IMU_TO_CAM, accel_sign=1.0,
                                  fuse_gyro=False),
    # X6 time offset, with each pixel timed by its sensor row. On a bumpy ridden
    # stretch, this single value steadies both a field seen near the top of
    # the front sensor and a house near the bottom of the back one; 2 ms
    # earlier or later is rougher on one of them. Aligning whole consecutive
    # frames against the gyro, which mixes all rows, put it 4 to 6 ms earlier.
    'Insta360 X6': ImuCalibration(IMU_TO_CAM_X6, accel_sign=-1.0,
                                  time_offset=0.001),
}


def imu_arrays(imu_samples, calib):
    """Video-time stamps (s), camera angular velocity (rad/s), camera-frame accel."""
    ts = (np.array([s['timestamp_ms'] for s in imu_samples]) / 1000.0
          - calib.time_offset)
    Q = calib.imu_to_cam
    gyro = np.deg2rad(np.array([s['gyro'] for s in imu_samples])) @ Q.T
    accel = np.array([s['accl'] for s in imu_samples], dtype=np.float64) @ Q.T
    return ts, gyro, accel


def integrate_gyro(ts, gyro_cam):
    """
    Integrate the camera angular velocity (rad/s) into C_k, the rotation taking
    a world-fixed direction from camera coordinates at ts[0] to those at ts[k].
    """
    steps = Rotation.from_rotvec(-gyro_cam[:-1] * np.diff(ts)[:, None]).as_quat()
    x, y, z, w = 0.0, 0.0, 0.0, 1.0
    quats = [(x, y, z, w)]
    for bx, by, bz, bw in steps.tolist():
        # C_{k+1} = step * C_k, as a Hamilton product
        x, y, z, w = (bw * x + bx * w + by * z - bz * y,
                      bw * y + by * w + bz * x - bx * z,
                      bw * z + bz * w + bx * y - by * x,
                      bw * w - bx * x - by * y - bz * z)
        quats.append((x, y, z, w))
    return Rotation.from_quat(quats)


def leveling_rotation(down_cam):
    """
    Rotation R with R @ [0, 1, 0] = down_cam: the shortest arc that levels the
    output horizon when the camera sees gravity along down_cam, (3,) or (n, 3).
    """
    d = np.asarray(down_cam, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    axis = np.stack([d[..., 2], np.zeros_like(d[..., 0]), -d[..., 0]], axis=-1)  # +Y x d
    s = np.linalg.norm(axis, axis=-1, keepdims=True)
    # Upside down, the arc is a half turn about any horizontal axis: take X.
    unit = np.where(s > 1e-9, axis / np.maximum(s, 1e-300), [1.0, 0.0, 0.0])
    return Rotation.from_rotvec(unit * np.arctan2(s, d[..., 1:2]))


def _ema(x, alpha):
    """Causal exponentially weighted sum of x along axis 0."""
    return lfilter([alpha], [1.0, alpha - 1.0], x, axis=0)


def _heading_correction(motion, down, sigma_samples):
    """
    Turn about gravity (rad), per sample, that makes the levelled output's
    heading follow a Gaussian-smoothed version of the camera's.
    """
    forward = motion.inv().apply(leveling_rotation(down).apply([0.0, 0.0, 1.0]))
    gravity = motion.inv().apply(down).mean(axis=0)
    gravity /= np.linalg.norm(gravity)
    e1 = np.cross(gravity, np.eye(3)[np.argmin(np.abs(gravity))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(gravity, e1)
    heading = np.unwrap(np.arctan2(forward @ e2, forward @ e1))
    return gaussian_filter1d(heading, sigma_samples, mode='nearest') - heading


def compute_stabilization_from_imu(imu_samples, calib, tau=16.0,
                                   accel_sigma=1.0, weight_floor=0.05,
                                   lowpass_hz=3.0, heading_sigma=0.5):
    """
    Estimate gravity in the camera frame with a complementary filter, and a
    smoothed heading.

    The gyroscope carries the orientation over short time scales and the
    accelerometer pulls it towards gravity with time constant tau (s). The pull
    is weighted by exp(-((|a_lp| - g) / accel_sigma)^2), floored at
    weight_floor, where a_lp is the accelerometer low-passed at lowpass_hz:
    vibration and cornering count less, but never zero, because on a bike |a|
    is rarely close to g.

    The filter runs offline and without lag: the accelerometer is expressed in
    the gyro-integrated frame (C_k^T a_k), where gravity barely moves, summed
    there with exponential weights forwards and backwards, then rotated back
    to each sample. Near either end of the clip the pass that has seen more
    data dominates the sum. A constant gyro bias tilts the two passes in
    opposite directions, so it cancels to first order.

    Leveling leaves the camera's rotation about the vertical, which on a bike
    is mostly vibration. The output heading therefore follows the camera's
    through a Gaussian of heading_sigma seconds; 0 keeps it on the camera.
    Without gyro fusion it always stays on the camera.

    Returns: ImuOrientation, or None without IMU data.
    """
    if not imu_samples:
        return None

    ts, gyro, accel = imu_arrays(imu_samples, calib)
    motion = integrate_gyro(ts, gyro)
    gravity = calib.accel_sign * accel
    if calib.fuse_gyro:
        gravity = motion.inv().apply(gravity)

    rate = 1.0 / np.median(np.diff(ts))
    b, a = butter(2, lowpass_hz / (rate / 2))
    magnitude = np.linalg.norm(filtfilt(b, a, gravity, axis=0), axis=1)
    weight = np.maximum(
        weight_floor, np.exp(-((magnitude - GRAVITY) / accel_sigma) ** 2))

    alpha = 1.0 / (tau * rate)
    weighted = gravity * weight[:, None]
    smooth = _ema(weighted, alpha) + _ema(weighted[::-1], alpha)[::-1]
    if calib.fuse_gyro:
        smooth = motion.apply(smooth)
    down = smooth / np.linalg.norm(smooth, axis=1, keepdims=True)
    if calib.fuse_gyro and heading_sigma:
        heading = _heading_correction(motion, down, heading_sigma * rate)
    else:
        heading = np.zeros(len(ts))
    return ImuOrientation(ts=ts, motion=Slerp(ts, motion), down=down,
                          heading=heading)


def compute_rs_rotations(orientation, frame_num, fps, readout_time_ms,
                         n_scanlines=32, apply_leveling=True):
    """
    Compute per-scanline orientation for rolling shutter correction.

    Scanline fraction f is read at t_frame + (f - 0.5) * readout. Its rotation
    is the camera motion since the frame centre, C(t_row) C(t_frame)^T,
    composed with the frame's leveling rotation (level horizon, smoothed
    heading).

    build_equirect_remap ignores its R_stabilization argument whenever
    rs_rotations is supplied, so the returned rotation is TOTAL. With
    apply_leveling=False the camera stays in its own frame; X5Pipeline asks for
    that, composes the seam frame, and levels the stitched image afterwards.

    Returns: list of (scanline_frac, Rotation) pairs, or None without IMU data.
    """
    if orientation is None:
        return None

    t_frame = frame_num / fps
    fracs = np.linspace(0.0, 1.0, n_scanlines)
    t_rows = t_frame + (fracs - 0.5) * readout_time_ms / 1000.0
    if apply_leveling:
        R_base = orientation.leveling_at(t_frame)
    else:
        R_base = Rotation.identity()
    R_rows = (orientation.motion_at(t_rows)
              * orientation.motion_at(t_frame).inv() * R_base)
    return [(float(f), R_rows[i]) for i, f in enumerate(fracs)]


# ============================================================
# 4. Frame Decoding
# ============================================================

def decode_frame(insv_path, frame_num, track, width, height):
    """Decode a single frame from a specific video track."""
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', insv_path, '-map', f'0:{track}',
        '-vf', f'select=eq(n\\,{frame_num})',
        '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    expected = width * height * 3
    if len(r.stdout) != expected:
        raise RuntimeError(
            f"Decode failed: got {len(r.stdout)} bytes, expected {expected}")
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


class InsvFrameReader:
    """
    Sequential dual-track frame reader.

    decode_frame() above spawns one ffmpeg per frame and uses select=eq(n,N)
    with no seek, so every request re-decodes from frame 0. Measured on an X6
    clip: 1.04s for frame 0 but 6.02s for frame 150, i.e. quadratic cost over
    a whole video. Here one ffmpeg process per track stays open and frames are
    read in order, which measured 0.039s per frame for both tracks.

    Tracks are read from two processes because a single ffmpeg cannot write
    two rawvideo streams to one pipe.
    """

    def __init__(self, insv_path, width, height, fps, start_frame=0):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.next_index = start_frame
        self.procs = []
        for track in (0, 1):
            cmd = ['ffmpeg', '-loglevel', 'error']
            if start_frame > 0:
                # Input-side seek: ffmpeg jumps to the keyframe before, then
                # decodes and discards up to the target, frame-accurately.
                # Half a frame early, so rounding never lands on a neighbour.
                cmd += ['-ss', f'{(start_frame - 0.5) / float(fps):.6f}']
            cmd += ['-i', insv_path, '-map', f'0:{track}']
            # Never duplicate or drop frames to keep a constant rate: after a
            # seek between two frames, ffmpeg would otherwise repeat the first
            # one and shift every later frame by a whole frame against the IMU.
            cmd += ['-fps_mode', 'passthrough',
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
            self.procs.append(subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0))

    def _read_one(self, proc):
        buf = bytearray(self.frame_size)
        view = memoryview(buf)
        got = 0
        while got < self.frame_size:
            n = proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            self.height, self.width, 3)

    def read(self):
        """Return the next (front, back) pair, or None at end of stream."""
        front = self._read_one(self.procs[0])
        back = self._read_one(self.procs[1])
        if front is None or back is None:
            return None
        self.next_index += 1
        return front, back

    def close(self):
        for proc in self.procs:
            try:
                proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                pass
        self.procs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_ahead(read, count, depth=2):
    """Yield read() up to count times, until it returns None, calling it ahead on a thread."""
    items, failure = queue.Queue(depth), []

    def run():
        try:
            for _ in range(count):
                item = read()
                if item is None:
                    break
                items.put(item)
        except BaseException as e:
            failure.append(e)
        finally:
            items.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (item := items.get()) is not None:
        yield item
    if failure:
        raise failure[0]


class BackgroundWriter:
    """Calls write(item) on a thread for each item put, in order."""

    def __init__(self, write, depth=2):
        self._write, self._items, self._failure = write, queue.Queue(depth), None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (item := self._items.get()) is not None:
            if self._failure is None:
                try:
                    self._write(item)
                except BaseException as e:
                    self._failure = e

    def put(self, item):
        if self._failure is not None:
            raise self._failure
        self._items.put(item)

    def close(self):
        self._items.put(None)
        self._thread.join()
        if self._failure is not None:
            raise self._failure


# ============================================================
# 5. MEI Forward Projection
# ============================================================

def mei_forward(X, Y, Z, xi, K, D_or_lens=None, lens=None):
    """
    Project 3D rays → pixel coordinates through MEI model.

    Supports both legacy 5-element D array and full MEILensParams with
    extended distortion (k1-k4, p1-p2, s1-s4).

    Returns: u, v (pixel coords), valid (bool mask)
    """
    # Use MEILensParams if provided, else legacy D array (no k4, no thin prism)
    if lens is not None or (isinstance(D_or_lens, MEILensParams)):
        L = lens if lens is not None else D_or_lens
        coeffs = (L.k1, L.k2, L.k3, L.k4, L.p1, L.p2, L.s1, L.s2, L.s3, L.s4)
    else:
        D = D_or_lens
        coeffs = (D[0], D[1], D[4] if len(D) > 4 else 0.0, 0.0, D[2], D[3],
                  0.0, 0.0, 0.0, 0.0)
    return _mei_project(X, Y, Z, float(xi), *map(float, coeffs),
                        float(K[0, 0]), float(K[0, 2]), float(K[1, 1]), float(K[1, 2]))


@fused
def _mei_project(xp, X, Y, Z, xi, k1, k2, k3, k4, p1, p2, s1, s2, s3, s4,
                 fx, cx, fy, cy):
    """mei_forward on its coefficients, as one kernel on the GPU."""
    norm = xp.sqrt(X*X + Y*Y + Z*Z)
    norm = xp.maximum(norm, 1e-10)
    Xs, Ys, Zs = X/norm, Y/norm, Z/norm

    denom = Zs + xi
    valid = denom > 1e-6

    x = xp.where(valid, Xs / denom, 0.0)
    y = xp.where(valid, Ys / denom, 0.0)

    r2 = x*x + y*y
    r4 = r2 * r2

    radial = 1.0 + k1*r2 + k2*r4 + k3*r2*r4 + k4*r4*r4
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x) + s1*r2 + s2*r4
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y + s3*r2 + s4*r4

    return fx * xd + cx, fy * yd + cy, valid


# ============================================================
# 6. Equirectangular Remap Builder
# ============================================================
#
# The stitch is built in the seam frame, whose poles are the lens axes: the
# front lens looks at its north pole, the back lens at its south pole, and the
# seam between them is the equator, sampled evenly all the way round. In the
# camera frame the seam runs through the camera's up and down directions, the
# poles of a camera-frame equirectangular image, where blend ramps and flow
# bands shrink to nothing; on a tilted camera a close object can sit there
# (4 deg from the handlebars on a bike). Seam longitude 0 is the camera's down
# direction, so the band wraps at its up direction. stitch_frame then turns
# the stitch to the output.

# camera ray = CAMERA_FROM_SEAM @ seam-frame ray
CAMERA_FROM_SEAM = Rotation.from_euler('X', -90, degrees=True)


def equirect_rays(eq_w, eq_h, xp=np):
    """Unit rays (3, eq_h * eq_w) through an equirectangular grid, Y down."""
    uu, vv = xp.meshgrid(
        xp.arange(eq_w, dtype=xp.float64),
        xp.arange(eq_h, dtype=xp.float64))

    lon = (uu / eq_w) * 2 * np.pi - np.pi
    lat = np.pi / 2 - (vv / eq_h) * np.pi

    return xp.stack([(xp.cos(lat) * xp.sin(lon)).ravel(),
                     (-xp.sin(lat)).ravel(),   # Y-down camera convention
                     (xp.cos(lat) * xp.cos(lon)).ravel()], axis=0)


def rotate_equirect(image, R, out_w=None, out_h=None, rays=None, pad=4):
    """
    Resample an equirectangular image so that the output direction r shows the
    input's direction R @ r, at out_w x out_h (default: the input size); rays
    are the output grid's equirect_rays, when already at hand. Lanczos,
    wrapping round the ±180 deg meridian and across the poles; pad covers the
    kernel's reach.
    """
    xp = array_module(image)
    h, w = image.shape[:2]
    out_w, out_h = out_w or w, out_h or h
    if rays is None:
        rays = equirect_rays(out_w, out_h, xp)
    # The maps are computed where the rays live, the image resampled where it
    # lives: float32 trigonometry rounds differently on the GPU, which moves a
    # few edge pixels by 1/32 px, so CPU rays reproduce CPU maps exactly.
    rp = array_module(rays)
    M = rp.asarray(R.as_matrix().astype(rays.dtype))

    def maps(rays, first, stop):
        x, y, z = M @ rays
        map_x = (rp.arctan2(x, z) + np.pi) / (2 * np.pi) * w + pad
        map_y = (np.pi / 2 - rp.arcsin(rp.clip(-y, -1.0, 1.0))) / np.pi * h + pad
        return (map_x.astype(rp.float32).reshape(stop - first, out_w),
                map_y.astype(rp.float32).reshape(stop - first, out_w))

    map_x, map_y = map_rows(maps, rays, out_w, out_h)

    # Past the top pole (row 0), row -k is row k half a turn round. The bottom
    # pole would be row h: it takes the mean of the last row, and row h + k is
    # row h - k half a turn round.
    rows = np.arange(-pad, h + pad)
    src = np.where(rows < 0, -rows, np.minimum(rows, 2 * h - rows)).clip(max=h - 1)
    cols = np.arange(-pad, w + pad)
    past_pole = (rows < 0) | (rows >= h)
    padded = image[xp.asarray(src)][:, xp.asarray(cols % w)]
    padded[xp.asarray(past_pole)] = image[xp.asarray(src[past_pole])][
        :, xp.asarray((cols + w // 2) % w)]
    padded[pad + h] = image[h - 1].mean(axis=0)

    return remap_lanczos(padded, map_x, map_y, replicate=True)


def lens_max_angle(lens, samples=721):
    """
    Largest angle (rad) between a ray and the optical axis for the lens to
    image it: half the calibrated FOV, and never past the fold of the
    distortion polynomial (114 deg on the X6), beyond which rays from behind
    the lens land inside the image circle again.
    """
    theta = np.linspace(0.0, np.pi, samples, endpoint=False)
    azimuth = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)[:, np.newaxis]
    u, v, _ = mei_forward(np.sin(theta) * np.cos(azimuth),
                          np.sin(theta) * np.sin(azimuth),
                          np.broadcast_to(np.cos(theta), (8, samples)),
                          lens.xi, lens.K, lens=lens)
    fold = theta[np.argmax(np.hypot(u - lens.cx, v - lens.cy), axis=1)].min()
    if lens.fov_deg is None:
        return fold
    return min(fold, np.radians(lens.fov_deg) / 2.0)


def _project_rays(lens, rays, eq_w, eq_h, depth_map=None, max_angle=None):
    """
    Camera-frame rays (3, eq_h * eq_w) to fisheye source pixels and validity;
    max_angle is lens_max_angle(lens), when already at hand.
    """
    xp = array_module(rays)
    if max_angle is None:
        max_angle = lens_max_angle(lens)
    # Transform ray directions to lens-local frame
    rays_lens = xp.asarray(lens.R_extrinsic.T) @ rays

    Xl = rays_lens[0].reshape(eq_h, eq_w)
    Yl = rays_lens[1].reshape(eq_h, eq_w)
    Zl = rays_lens[2].reshape(eq_h, eq_w)

    # Translation-aware projection: account for lens offset from camera center.
    # For a 3D point at distance d along the ray:
    #   P_world = d * ray_dir
    #   P_lens = R.T @ (P_world - t_extrinsic)
    #          = d * ray_lens - t_local
    # MEI normalizes to unit sphere internally, so the ratio matters, not scale.
    # At d→∞ the t_local term vanishes → standard projection.
    if depth_map is not None and np.linalg.norm(lens.t_extrinsic) > 1e-6:
        # The protobuf t_extrinsic is the offset FROM this lens TO the
        # camera center (not the lens position). So the lens position
        # is at -t_extrinsic, and we ADD t_local to correct the ray.
        t_local = lens.R_extrinsic.T @ lens.t_extrinsic
        d = xp.asarray(depth_map, dtype=xp.float64)
        Xl = d * Xl + t_local[0]
        Yl = d * Yl + t_local[1]
        Zl = d * Zl + t_local[2]

    u_src, v_src, valid = mei_forward(Xl, Yl, Zl, lens.xi, lens.K, lens=lens)

    # Bounds check against sensor
    valid = valid & (u_src >= 0) & (u_src < lens.width - 1)
    valid = valid & (v_src >= 0) & (v_src < lens.height - 1)

    # Field of view, tested on the ray angle: past the fold the pixel radius
    # shrinks again, so a radius test alone accepts rays from behind the lens.
    cos_axis = Zl / xp.maximum(xp.sqrt(Xl * Xl + Yl * Yl + Zl * Zl), 1e-12)
    valid = valid & (cos_axis > np.cos(max_angle))

    if lens.fov_deg is None:
        # No calibrated FOV: stay inside a margin of the image circle
        margin = 0.05
        r_max = min(lens.width, lens.height) / 2.0 * (1.0 - margin)
        dist_from_center = xp.sqrt(
            (u_src - lens.cx) ** 2 + (v_src - lens.cy) ** 2)
        valid = valid & (dist_from_center < r_max)
    return u_src, v_src, valid


def _rotate_rays_at(rs_rotations, fracs, rays, chunk=1 << 20):
    """
    Rotate the rays (3, N) by the rotation at their readout fraction, a scalar
    or one per ray. Neighbouring keyframes differ by well under a tenth of a
    degree, so their matrices are interpolated linearly rather than slerped.
    """
    xp = array_module(rays)
    keys = np.array([f for f, _ in rs_rotations])
    mats = xp.asarray(Rotation.concatenate([R for _, R in rs_rotations]).as_matrix())
    if np.ndim(fracs) == 0:
        pos = np.interp(fracs, keys, np.arange(len(keys), dtype=np.float64))
        idx = min(int(pos), len(keys) - 2)
        alpha = pos - idx
        return ((1.0 - alpha) * mats[idx] + alpha * mats[idx + 1]) @ rays
    pos = xp.interp(fracs, xp.asarray(keys), xp.arange(len(keys), dtype=xp.float64))
    idx = xp.minimum(pos.astype(xp.int64), len(keys) - 2)
    alpha = pos - idx
    out = xp.empty_like(rays)
    for s in range(0, rays.shape[1], chunk):
        i, a = idx[s:s + chunk], alpha[s:s + chunk, None, None]
        M = (1.0 - a) * mats[i] + a * mats[i + 1]
        out[:, s:s + chunk] = xp.einsum('nij,jn->in', M, rays[:, s:s + chunk])
    return out


def build_equirect_remap(lens, eq_w, eq_h, R_stabilization=None,
                         rs_rotations=None, depth_map=None, rays=None):
    """
    Build remap tables: equirectangular pixels → fisheye source pixels.
    Camera convention: X=right, Y=down, Z=forward.

    Args:
        rs_rotations: Optional list of (readout_frac, Rotation) pairs for
            rolling shutter correction, readout_frac being the sensor row
            over the image height.
        depth_map: Optional H×W float array of scene depth (meters). When
            provided, the lens translation (t_extrinsic) is used to compute
            parallax-corrected projections. At d=∞ the result is identical
            to the standard projection; at finite d, close objects shift
            to their geometrically correct position for this specific lens.
        rays: the grid's equirect_rays, when already at hand. The maps are
            built where the rays live, on the CPU or the GPU.
    """
    if rays is None:
        rays = equirect_rays(eq_w, eq_h)
    max_angle = lens_max_angle(lens)

    def maps(rays, first, stop):
        xp = array_module(rays)
        rows = stop - first
        depth = None if depth_map is None else depth_map[first:stop]
        if rs_rotations is not None and len(rs_rotations) > 1:
            # Rolling shutter: a source pixel is exposed when its sensor row
            # is read, and rows run down the stored fisheye image. Project
            # once with the mid-readout rotation to find each pixel's row,
            # then again with the rotation at that row's time.
            _, v_mid, _ = _project_rays(
                lens, _rotate_rays_at(rs_rotations, 0.5, rays), eq_w, rows,
                depth, max_angle)
            row = xp.clip(xp.nan_to_num(v_mid.ravel() / lens.height, nan=0.5), 0.0, 1.0)
            rays = _rotate_rays_at(rs_rotations, row, rays)
        elif R_stabilization is not None:
            rays = xp.asarray(R_stabilization.as_matrix()) @ rays

        u_src, v_src, valid = _project_rays(lens, rays, eq_w, rows, depth, max_angle)

        map_x = xp.where(valid, u_src, 0).astype(xp.float32)
        map_y = xp.where(valid, v_src, 0).astype(xp.float32)
        return map_x, map_y, valid

    return map_rows(maps, rays, eq_w, eq_h)


def remap_fisheye(fisheye_bgr, map_x, map_y, valid):
    """Apply remap tables to a fisheye image."""
    result = remap_lanczos(fisheye_bgr, map_x, map_y)
    result[~valid] = 0
    return result


# ============================================================
# 7. Blending
# ============================================================

def compute_blend_weights(valid_front, valid_back, eq_w, eq_h,
                          blend_width_deg=15.0):
    """
    Blend weights in the seam frame (see CAMERA_FROM_SEAM), combining
    latitude preference with coverage depth.

    In the overlap region, each lens's weight is proportional to:
      latitude_preference × coverage_depth

    This naturally:
    - Assigns front/back hemisphere ownership by the side of the seam
    - Favors the lens further from its fisheye edge (more central = less
      distortion, less parallax). No hardcoded feather distance needed
    - Smoothly transitions at coverage boundaries

    Where only one lens is valid it takes the pixel whole, and where neither
    is, both weights are 0, so only the rows holding overlap are computed.
    """
    xp = array_module(valid_front)
    w_front = (valid_front & ~valid_back).astype(xp.float32)
    w_back = (~valid_front & valid_back).astype(xp.float32)
    overlap = valid_front & valid_back
    rows = overlap_rows(overlap)
    if rows is None:
        return w_front, w_back
    band = slice(*rows)

    # Latitude preference: the front lens owns the upper half, ramping over
    # blend_width_deg on either side of the seam
    lat_deg = 90.0 - xp.arange(*rows, dtype=xp.float32) / eq_h * 180.0
    front_pref = xp.clip(0.5 + lat_deg / (2.0 * blend_width_deg), 0.0, 1.0)[:, np.newaxis]

    # Coverage depth: how far each pixel is from its lens's coverage edge
    front_dist, back_dist = (xp.asarray(d) for d in
                             coverage_depth((valid_front, valid_back), overlap, rows))

    # Combined weight: longitude preference × coverage depth
    # Coverage depth naturally fades to 0 at the edge. No hardcoded feather.
    wf = front_pref * front_dist
    wb = (1.0 - front_pref) * back_dist

    # Normalize
    total = xp.maximum(wf + wb, 1e-6)
    w_front[band] = xp.where(overlap[band], wf / total, w_front[band])
    w_back[band] = xp.where(overlap[band], wb / total, w_back[band])
    return w_front, w_back


def coverage_depth(valids, overlap, rows):
    """
    cv2.distanceTransform(valid, DIST_L2, 5) for each coverage mask in
    valids, on the rows first:stop given by rows: each pixel's distance (px)
    to its lens's coverage edge, exact on the overlap pixels.

    It runs on a strip reaching past those rows. A 5x5 chamfer distance is at
    least the row offset, and the strip's edges only lengthen distances, so a
    distance within the strip's reach is exact; beyond it, the whole image is
    used.
    """
    first, stop = rows
    h = overlap.shape[0]
    lo, hi = max(first - (stop - first) - 16, 0), min(stop + (stop - first) + 16, h)

    def depth(valid, lo, hi):
        mask = np.ascontiguousarray(to_numpy(valid[lo:hi])).view(np.uint8)
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5)[first - lo:stop - lo]

    dists = parallel_map(lambda v: depth(v, lo, hi), valids)
    reach = min(first - lo if lo > 0 else np.inf, hi - stop if hi < h else np.inf)
    inside = to_numpy(overlap[first:stop])
    if max(d[inside].max() for d in dists) > reach:
        dists = parallel_map(lambda v: depth(v, 0, h), valids)
    return dists


def compute_gain_compensation(front_overlap, back_overlap, overlap_mask):
    """Per-channel global gain to equalize exposure in overlap."""
    valid = overlap_mask
    gains = np.ones(3, dtype=np.float64)
    for c in range(3):
        mf = front_overlap[:, :, c][valid].mean()
        mb = back_overlap[:, :, c][valid].mean()
        if mb > 1:
            gains[c] = mf / mb
    return np.clip(gains, 0.5, 2.0)


def compute_spatial_gain(eq_front, eq_back, overlap_mask, blur_sigma=80):
    """
    Compute a spatially-varying per-channel gain field to equalize
    the back image to match the front image's appearance.

    Uses the overlap region to measure the per-pixel intensity ratio,
    then smooths it into a continuous gain field that extends beyond
    the overlap.

    The images may be a band of rows cut from the stitch: rows beyond it
    count as holding no overlap, which is exact when the band holds all of it.
    """
    xp = array_module(eq_front)
    h, w = eq_front.shape[:2]
    gain_field = xp.empty((h, w, 3), dtype=xp.float32)

    for c in range(3):
        f = eq_front[:, :, c].astype(xp.float32)
        b = eq_back[:, :, c].astype(xp.float32)

        # Per-pixel ratio in overlap (where both have signal), outliers
        # clamped, zero elsewhere
        valid = overlap_mask & (b > 10) & (f > 5)
        ratio = xp.where(valid, xp.clip(f / xp.maximum(b, 1), 0.3, 3.0), np.float32(0))
        weight = valid.astype(xp.float32)

        # Weighted Gaussian blur: blur(ratio * weight) / blur(weight)
        ratio_blurred = gaussian_blur_rows(ratio, blur_sigma)
        weight_blurred = gaussian_blur_rows(weight, blur_sigma)
        weight_blurred = xp.maximum(weight_blurred, 1e-6)

        gain_field[:, :, c] = ratio_blurred / weight_blurred

    # Clamp to reasonable range
    gain_field = xp.clip(gain_field, 0.5, 2.0)
    return gain_field


def apply_gain(image, gains):
    """Apply per-channel gain (scalar or spatial field)."""
    result = image.astype(np.float32)
    if isinstance(gains, np.ndarray) and gains.ndim == 3:
        result *= gains
    else:
        for c in range(3):
            result[:, :, c] *= gains[c]
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 8. Optical Flow (DIS)
# ============================================================

class FlowEngine:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis.setVariationalRefinementIterations(5)
        self.dis.setFinestScale(0)
        self.dis.setGradientDescentIterations(25)

    def compute(self, front_gray, back_gray, prev_flow=None):
        return self.dis.calc(front_gray, back_gray, prev_flow)


# ============================================================
# 9. Pipeline Orchestrator
# ============================================================

def find_overlap_bands(valid_front, valid_back):
    """Find contiguous overlap column ranges."""
    overlap = valid_front & valid_back
    cols = np.any(overlap, axis=0)
    diff = np.diff(cols.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if cols[0]:
        starts = np.concatenate([[0], starts])
    if cols[-1]:
        ends = np.concatenate([ends, [len(cols)]])
    return overlap, list(zip(starts.tolist(), ends.tolist()))


class X5Pipeline:

    def __init__(self, insv_path, eq_width=3840,
                 enable_stabilization=True, enable_flow=True,
                 denoise=False, stitch_width=None, device='auto'):
        self.insv_path = insv_path
        self.xp = resolve_device(device)
        self.eq_w = eq_width
        self.eq_h = eq_width // 2
        # The stitch is built in the seam frame, then turned to the output. At
        # the output width that second resampling visibly softens detail; at
        # 1.5x it matches a single remap, for twice the time per frame.
        self.st_w = stitch_width or eq_width * 3 // 2
        self.st_h = self.st_w // 2
        # Width of the seam frame as seen by the parallax flow
        self.flow_width = min(eq_width, self.st_w)
        # Output directions, turned into the seam frame after each stitch
        # (on the CPU, see rotate_equirect), and seam-frame directions, for
        # the remap tables
        self.out_rays = equirect_rays(self.eq_w, self.eq_h).astype(np.float32)
        self.st_rays = equirect_rays(self.st_w, self.st_h, self.xp)
        self.enable_stabilization = enable_stabilization
        self.enable_flow = enable_flow
        self.denoise = denoise

        print("[1/4] Extracting metadata...")
        self.meta = extract_metadata(insv_path)
        print(f"  Video: {self.meta.width}x{self.meta.height} @ "
              f"{self.meta.fps}fps, {self.meta.frame_count} frames")

        # Rolling shutter needs the gyro even when leveling is off.
        print("[2/4] Computing IMU orientation...")
        self.imu_orientation = compute_stabilization_from_imu(
            self.meta.imu_samples, self.meta.imu_calib)
        if self.imu_orientation is None:
            print("  No IMU data: no stabilization, no rolling shutter correction")

        print("[3/4] Building remap tables...")
        self._build_maps(frame_idx=0)

        if enable_flow:
            self.flow_engine = FlowEngine()

        print("[4/4] Ready.")
        print(f"  Output: {self.eq_w}x{self.eq_h}, stitched at {self.st_w}x{self.st_h} "
              f"on the {'CPU' if self.xp is np else 'GPU'}")

    def _build_maps(self, frame_idx=0, depth_map=None):
        """Seam-frame remap tables with rolling shutter correction."""
        # The maps stay in the seam frame; stitch_frame levels the result.
        rs_rots = compute_rs_rotations(
            self.imu_orientation, frame_idx, self.meta.fps,
            self.meta.frame_readout_time, n_scanlines=32,
            apply_leveling=False)
        if rs_rots is not None:
            rs_rots = [(f, R * CAMERA_FROM_SEAM) for f, R in rs_rots]

        self.maps = []
        for lens in self.meta.lens_params:
            mx, my, valid = build_equirect_remap(
                lens, self.st_w, self.st_h, R_stabilization=CAMERA_FROM_SEAM,
                rs_rotations=rs_rots, depth_map=depth_map, rays=self.st_rays)
            self.maps.append((mx, my, valid))

        self.w_front, self.w_back = compute_blend_weights(
            self.maps[0][2], self.maps[1][2], self.st_w, self.st_h)

    def _seam_from_output(self, frame_num):
        """Rotation taking output directions into the seam frame."""
        R = CAMERA_FROM_SEAM.inv()
        if self.enable_stabilization and self.imu_orientation is not None:
            R = R * self.imu_orientation.leveling_at(frame_num / self.meta.fps)
        return R

    def _align_band(self, fc, bc, valid_f, valid_b, alpha):
        """
        Warp both lenses of a band whose seam runs down its columns, front lens
        on the left, onto a common position given by DIS flow. alpha is the
        back lens's blend weight: where it is 0 the front lens stays put, where
        it is 1 the back lens does, so the band meets the single-lens image on
        either side without a step. The arrays live on the CPU or the GPU;
        DIS flow itself runs on the CPU.
        """
        xp = array_module(fc)
        # Fill invalid pixels with the other lens's data before flow
        # so DIS doesn't try to match black regions to content.
        fc = xp.where(valid_f[:, :, np.newaxis], fc, bc)
        bc = xp.where(valid_b[:, :, np.newaxis], bc, fc)

        # Flow is estimated with the seam frame scaled to flow_width.
        fg = cv2.cvtColor(to_numpy(fc), cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(to_numpy(bc), cv2.COLOR_BGR2GRAY)
        scale = self.flow_width / self.st_w
        if scale < 1.0:
            size = (round(fg.shape[1] * scale), round(fg.shape[0] * scale))
            flow = self.flow_engine.compute(
                cv2.resize(fg, size, interpolation=cv2.INTER_AREA),
                cv2.resize(bg, size, interpolation=cv2.INTER_AREA))
            flow = cv2.resize(flow, (fg.shape[1], fg.shape[0])) / scale
        else:
            flow = self.flow_engine.compute(fg, bg)

        # Clamp flow magnitude to physical limit: the ~30 mm lens baseline
        # seen from 25 cm spans about 8 deg
        flow = xp.asarray(flow)
        max_flow = 8.0 / 360.0 * self.st_w
        mag = xp.linalg.norm(flow, axis=2, keepdims=True)
        flow *= xp.minimum(1.0, max_flow / (mag + 1e-6))

        # Zero flow outside the overlap (no correction needed there)
        flow[~(valid_f & valid_b)] = 0

        # DIS gives front(x) ~ back(x + flow). A feature at x in the front
        # lens is drawn at x + alpha * flow by both warped lenses.
        h, w = flow.shape[:2]
        xs = xp.arange(w, dtype=xp.float32)
        ys = xp.arange(h, dtype=xp.float32)[:, np.newaxis]
        fw = remap_linear(fc, xs - alpha * flow[:, :, 0],
                          ys - alpha * flow[:, :, 1])
        bw = remap_linear(bc, xs + (1 - alpha) * flow[:, :, 0],
                          ys + (1 - alpha) * flow[:, :, 1])

        return fw, bw

    def stitch_frame(self, frame_num=0, frames=None):
        """
        Stitch a single frame with depth-aware translation correction.

        Pass `frames` as a (front, back) pair to reuse an open
        InsvFrameReader; otherwise one is opened for this frame alone.
        Returns a numpy image, whichever device stitched it.
        """
        t0 = time.time()
        xp = self.xp

        if self.imu_orientation is not None:
            self._build_maps(frame_num)

        if frames is None:
            with InsvFrameReader(self.insv_path, self.meta.width,
                                 self.meta.height, self.meta.fps,
                                 frame_num) as reader:
                frames = reader.read()
            if frames is None:
                raise RuntimeError(f"Decode failed: no frame {frame_num}")
        front, back = (xp.asarray(f) for f in frames)

        eq_front = remap_fisheye(front, *self.maps[0])
        eq_back = remap_fisheye(back, *self.maps[1])

        valid_f, valid_b = self.maps[0][2], self.maps[1][2]
        overlap_mask = valid_f & valid_b

        # Outside the rows holding overlap, each pixel belongs to one lens
        # whole (its weight is 1, the other's 0) or to neither (both images
        # are 0 there): only the overlap rows need blending.
        result = xp.where(valid_f[:, :, np.newaxis], eq_front, eq_back)
        rows = overlap_rows(overlap_mask)

        if rows is not None:
            band = slice(*rows)
            w_front = self.w_front[band][:, :, np.newaxis]
            w_back = self.w_back[band][:, :, np.newaxis]

            # Color harmonization: SYMMETRIC correction at the seam.
            # Both lenses are adjusted toward their geometric mean in the overlap,
            # so neither side has an uncorrected color step at the transition.
            # The correction fades to zero away from the seam.
            eq_front_f = eq_front[band].astype(xp.float32)
            eq_back_f = eq_back[band].astype(xp.float32)
            gain_field = compute_spatial_gain(eq_front[band], eq_back[band],
                                              overlap_mask[band])
            # gain = front/back. To meet in the middle:
            #   front_adj = front / sqrt(gain) = front * sqrt(back/front)
            #   back_adj  = back  * sqrt(gain) = back  * sqrt(front/back)
            sqrt_gain = xp.sqrt(xp.maximum(gain_field, 0.01))
            inv_sqrt_gain = 1.0 / xp.maximum(sqrt_gain, 0.01)

            # Correction weight: strongest at seam center, zero in primary regions
            correction_weight = 2.0 * xp.minimum(w_front, w_back)

            # Symmetrically correct both lenses, fading with distance from seam
            eq_front_f = eq_front_f * (1.0 - correction_weight) + \
                         eq_front_f * inv_sqrt_gain * correction_weight
            eq_back_f = eq_back_f * (1.0 - correction_weight) + \
                        eq_back_f * sqrt_gain * correction_weight

            eq_front[band] = xp.clip(eq_front_f, 0, 255).astype(xp.uint8)
            eq_back[band] = xp.clip(eq_back_f, 0, 255).astype(xp.uint8)

            # Base canvas: smooth weighted blend
            result[band] = xp.clip(
                eq_front[band].astype(xp.float32) * w_front +
                eq_back[band].astype(xp.float32) * w_back,
                0, 255).astype(xp.uint8)

        if self.enable_flow and rows is not None:
            # Parallax correction in a band around the seam, the seam frame's
            # equator. The band is laid on its side so the seam runs down its
            # columns, front lens on the left, and padded round the wrap at the
            # camera's up direction.
            hw = int(15.0 / 180.0 * self.st_h)
            flow_rows = slice(self.st_h // 2 - hw, self.st_h // 2 + hw)

            def band(image):
                image = xp.concatenate([image[:, -hw:], image, image[:, :hw]], axis=1)
                return xp.ascontiguousarray(xp.swapaxes(image, 0, 1))

            # The aligned lenses are blended with the canvas weights, so the
            # band matches the canvas wherever one lens alone is valid.
            alpha = band(self.w_back[flow_rows])
            fw, bw = self._align_band(
                *(band(image[flow_rows]) for image in (eq_front, eq_back, valid_f, valid_b)),
                alpha)
            alpha = alpha[:, :, np.newaxis]
            blended = fw * (1.0 - alpha) + bw * alpha
            result[flow_rows] = xp.clip(xp.swapaxes(blended, 0, 1)[:, hw:-hw],
                                        0, 255).astype(xp.uint8)

        # Turn the stitch from the seam frame to the output
        result = to_numpy(rotate_equirect(result, self._seam_from_output(frame_num),
                                          self.eq_w, self.eq_h, self.out_rays))

        # Optional edge-preserving denoise (bilateral filter)
        if self.denoise:
            result = cv2.bilateralFilter(result, 9, 40, 40)

        dt = time.time() - t0
        print(f"Frame {frame_num}: {dt:.2f}s")
        return result

    def stitch_frame_to_file(self, frame_num, output_path):
        result = self.stitch_frame(frame_num)
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")
        return result

    def stitch_video(self, output_path, start=0, end=-1):
        """
        Process frames to video using ffmpeg pipe for output. Decoding and
        encoding run ahead and behind on their own threads, so the stitch
        waits for neither.
        """
        if end == -1:
            end = self.meta.frame_count

        # Use ffmpeg pipe for H.264 output (much better compression than mp4v)
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{self.eq_w}x{self.eq_h}',
            '-r', str(self.meta.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '18', '-pix_fmt', 'yuv420p',
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        t_total = time.time()
        reader = InsvFrameReader(self.insv_path, self.meta.width,
                                 self.meta.height, self.meta.fps, start)
        writer = BackgroundWriter(lambda image: proc.stdin.write(image.tobytes()))
        n = start
        try:
            for pair in read_ahead(reader.read, end - start):
                writer.put(self.stitch_frame(n, frames=pair))
                n += 1
        finally:
            reader.close()
            writer.close()
            proc.stdin.close()
            proc.wait()
        if n < end:
            print(f"  stream ended early at frame {n}")
            end = n
        dt = time.time() - t_total
        n_frames = end - start
        print(f"Video saved: {output_path} "
              f"({n_frames} frames in {dt:.1f}s, {n_frames/dt:.1f} fps)")


# ============================================================
# 10. Ground Truth Comparison
# ============================================================

def extract_gt_frame(gt_path, frame_num=0):
    """Extract a frame from ground truth video."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'csv=p=0', gt_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    parts = [p for p in r.stdout.strip().split(',') if p]
    w, h = int(parts[0]), int(parts[1])

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', gt_path,
           '-vf', f'select=eq(n\\,{frame_num})',
           '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(h, w, 3).copy()


def compare_psnr(our, gt):
    """Compute PSNR between two images (resizes gt to match)."""
    if our.shape != gt.shape:
        gt = cv2.resize(gt, (our.shape[1], our.shape[0]))
    mse = np.mean((our.astype(float) - gt.astype(float)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Insta360 X5 Stitching Pipeline')
    parser.add_argument('input', help='Path to .insv file')
    parser.add_argument('-o', '--output', default='output.jpg')
    parser.add_argument('-f', '--frame', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=3840)
    parser.add_argument('--stitch-width', type=int, default=None,
                        help='Width of the seam-frame stitch before it is '
                             'turned to the output (default: 1.5x --width; '
                             '--width is twice as fast but softer)')
    parser.add_argument('--no-stab', action='store_true',
                        help='Disable IMU stabilization')
    parser.add_argument('--no-flow', action='store_true',
                        help='Disable optical flow stitching')
    parser.add_argument('--denoise', action='store_true',
                        help='Apply bilateral denoising (matches Insta360 Studio)')
    parser.add_argument('--video', action='store_true',
                        help='Process all frames to video')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame (for video mode)')
    parser.add_argument('--end', type=int, default=-1,
                        help='End frame (for video mode, -1 = all)')
    parser.add_argument('--gt', help='Ground truth video for comparison')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto',
                        help='Where to stitch (default: the GPU when CuPy '
                             'finds one, else the CPU)')
    args = parser.parse_args()

    pipeline = X5Pipeline(args.input, eq_width=args.width,
                          enable_stabilization=not args.no_stab,
                          enable_flow=not args.no_flow,
                          denoise=args.denoise,
                          stitch_width=args.stitch_width,
                          device=args.device)

    if args.video:
        pipeline.stitch_video(args.output, args.start, args.end)
    else:
        result = pipeline.stitch_frame_to_file(args.frame, args.output)
        if args.gt:
            gt = extract_gt_frame(args.gt, args.frame)
            psnr = compare_psnr(result, gt)
            print(f"PSNR vs ground truth: {psnr:.2f} dB")
