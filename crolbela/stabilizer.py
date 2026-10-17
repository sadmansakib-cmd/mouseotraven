# stabilizer.py -- Cursor smoothing pipeline: landmark EMA, 1-Euro filter, deadzone,
# re-engage glide, and the CursorFreezer used while clicking.
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import config


def _now() -> float:
    return time.perf_counter()


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# 1-Euro Filter (1D)
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    Adaptive low-pass filter (Casiez et al., CHI 2012).

      - Slow movement -> low cutoff  -> heavy smoothing -> no jitter.
      - Fast movement -> high cutoff -> light smoothing -> no lag.
  (Extension: speeds below ONE_EURO_SPEED_FLOOR px/s are ignored by the cutoff rule.)

    Cutoffs are in Hz (standard 1-Euro definition, alpha = 1 / (1 + tau/dt),
    tau = 1 / (2*pi*cutoff)). Time-step is measured, so behaviour does not
    change when the camera FPS changes.
    """

    def __init__(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
        speed_floor: Optional[float] = None,
    ) -> None:
        self._min_cutoff = config.ONE_EURO_MIN_CUTOFF if min_cutoff is None else min_cutoff
        self._beta = config.ONE_EURO_BETA if beta is None else beta
        self._d_cutoff = config.ONE_EURO_D_CUTOFF if d_cutoff is None else d_cutoff
        self._speed_floor = config.ONE_EURO_SPEED_FLOOR if speed_floor is None else speed_floor

        self._x_prev: float = 0.0
        self._dx_prev: float = 0.0
        self._last_ts: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        self._initialized = False

    def filter(self, x: float, t: Optional[float] = None) -> float:
        now = _now() if t is None else t

        if not self._initialized:
            self._x_prev = x
            self._dx_prev = 0.0
            self._last_ts = now
            self._initialized = True
            return x

        dt = _clamp(now - self._last_ts, 1.0 / 240.0, 0.25)
        self._last_ts = now

        # Smoothed speed estimate
        dx = (x - self._x_prev) / dt
        dx_hat = self._low_pass(dx, self._dx_prev, self._alpha(dt, self._d_cutoff))
        self._dx_prev = dx_hat

        # Speed-dependent cutoff. Speeds below the floor are treated as sensor noise,
        # so tracking noise at rest can never open the filter up.
        cutoff = self._min_cutoff + self._beta * max(0.0, abs(dx_hat) - self._speed_floor)
        x_hat = self._low_pass(x, self._x_prev, self._alpha(dt, cutoff))
        self._x_prev = x_hat
        return x_hat

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(x: float, x_prev: float, alpha: float) -> float:
        return alpha * x + (1.0 - alpha) * x_prev


class OneEuroFilter2D:
    """Independent 1-Euro filters on X and Y."""

    def __init__(self) -> None:
        self._fx = OneEuroFilter()
        self._fy = OneEuroFilter()

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def filter(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        return self._fx.filter(x, t), self._fy.filter(y, t)


# ---------------------------------------------------------------------------
# Landmark EMA pre-smoother (time-aware)
# ---------------------------------------------------------------------------

class LandmarkEMA:
    """
    Light exponential smoothing applied before the 1-Euro filter. It removes the
    frame-to-frame sensor noise so the 1-Euro speed estimate is not fooled by it.
    """

    def __init__(self, tau: Optional[float] = None) -> None:
        self._tau = config.LANDMARK_SMOOTH_TAU if tau is None else tau
        self._x: float = 0.0
        self._y: float = 0.0
        self._last_ts: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        self._initialized = False

    def filter(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        now = _now() if t is None else t
        if not self._initialized:
            self._x, self._y, self._last_ts = x, y, now
            self._initialized = True
            return x, y

        dt = _clamp(now - self._last_ts, 1.0 / 240.0, 0.25)
        self._last_ts = now
        a = 1.0 - math.exp(-dt / self._tau)
        self._x += a * (x - self._x)
        self._y += a * (y - self._y)
        return self._x, self._y


# ---------------------------------------------------------------------------
# Deadzone
# ---------------------------------------------------------------------------

class DeadzoneFilter:
    """
    Ignores micro-movements below a radius, then follows continuously (no snap)
    once the movement leaves the radius.
    """

    def __init__(self, radius: Optional[float] = None) -> None:
        self._radius = config.DEADZONE_RADIUS_PX if radius is None else radius
        self._last_x: float = 0.0
        self._last_y: float = 0.0
        self._initialized: bool = False

    def reset(self) -> None:
        self._initialized = False

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        if not self._initialized:
            self._last_x, self._last_y = x, y
            self._initialized = True
            return x, y

        dist = math.hypot(x - self._last_x, y - self._last_y)
        if dist < self._radius:
            return self._last_x, self._last_y

        scale = (dist - self._radius) / dist
        self._last_x += (x - self._last_x) * scale
        self._last_y += (y - self._last_y) * scale
        return self._last_x, self._last_y


# ---------------------------------------------------------------------------
# Cursor pipeline (EMA -> 1-Euro -> deadzone -> re-engage glide)
# ---------------------------------------------------------------------------

class CursorPipeline:
    """
    Turns raw fingertip screen coordinates into a stable, smooth cursor position.

    Why the "glide": whenever tracking is interrupted (scroll, click, pause, hand
    lost) the filters must restart, and the fingertip is usually somewhere else
    than the cursor. Without a glide the cursor teleports. After suspend(), the
    next update() eases from the cursor's current position to the fingertip over
    CURSOR_REENGAGE_SECONDS instead.
    """

    def __init__(self) -> None:
        self._ema = LandmarkEMA()
        self._f = OneEuroFilter2D()
        self._dz = DeadzoneFilter()

        self._active: bool = False
        self._pos: Optional[Tuple[float, float]] = None
        self._glide_from: Optional[Tuple[float, float]] = None
        self._glide_t0: float = 0.0
        self._glide_dur: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        return self._pos

    def sync(self, pos: Tuple[float, float]) -> None:
        """Tell the pipeline where the OS cursor really is (used as glide start)."""
        self._pos = pos

    def suspend(self) -> None:
        """Stop tracking (cursor stays where it is). Next update() glides back in."""
        self._active = False

    def update(self, sx: float, sy: float, t: Optional[float] = None) -> Tuple[float, float]:
        now = _now() if t is None else t

        if not self._active:
            self._ema.reset()
            self._f.reset()
            self._dz.reset()
            self._glide_from = self._pos
            self._glide_t0 = now
            self._glide_dur = None
            self._active = True

        x, y = self._ema.filter(sx, sy, now)
        x, y = self._f.filter(x, y, now)
        x, y = self._dz.apply(x, y)

        if self._glide_from is not None:
            gx, gy = self._glide_from
            if self._glide_dur is None:   # fixed on the first frame, from the real distance
                gap = math.hypot(x - gx, y - gy)
                self._glide_dur = (config.CURSOR_REENGAGE_SECONDS
                                   + min(0.3, gap * config.CURSOR_REENGAGE_PER_PX))
            dur = self._glide_dur
            s = (now - self._glide_t0) / dur if dur > 0 else 1.0
            if s >= 1.0:
                self._glide_from = None
            else:
                w = s * s * (3.0 - 2.0 * s)          # smoothstep ease-in-out
                x = gx + (x - gx) * w
                y = gy + (y - gy) * w

        self._pos = (x, y)
        return x, y


# ---------------------------------------------------------------------------
# Cursor Freezer
# ---------------------------------------------------------------------------

class CursorFreezer:
    """Locks the cursor position while a pinch click is being detected."""

    def __init__(self) -> None:
        self._frozen: bool = False
        self._frozen_x: float = 0.0
        self._frozen_y: float = 0.0

    def freeze(self, x: float, y: float) -> None:
        if not self._frozen:
            self._frozen = True
            self._frozen_x = x
            self._frozen_y = y

    def release(self) -> None:
        self._frozen = False

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        if self._frozen:
            return self._frozen_x, self._frozen_y
        return x, y

    @property
    def is_frozen(self) -> bool:
        return self._frozen
