# os_controller.py -- Thread-safe high-speed OS mouse control via pynput.
from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

from pynput.mouse import Button, Controller as MouseController

import config


class OSMouseController:
    """
    Thread-safe wrapper around pynput.mouse.Controller.

    Features:
    - High-speed direct OS input injection with zero artificial pause.
    - Fractional scroll accumulator: accumulates sub-unit scroll values across
      frames and emits integer scroll steps as soon as abs(accumulator) >= 1.0.
    - Zero drift decay: when fingers stop, residual fraction is cleared so
      scrolling stops dead without creeping.
    """

    def __init__(self) -> None:
        self._mouse = MouseController()
        self._lock = threading.Lock()
        self._last_pos: Tuple[float, float] = (0.0, 0.0)
        self._ix: Optional[int] = None
        self._iy: Optional[int] = None

        # Fractional scroll accumulators
        self._scroll_acc_x: float = 0.0
        self._scroll_acc_y: float = 0.0
        self._scroll_last_input_t: float = 0.0

    # ------------------------------------------------------------------
    # Cursor Movement
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float) -> None:
        """
        Move the OS cursor to absolute screen position (x, y).
        1 px hysteresis on the rounding: a value hovering around x.5 can no longer
        make the cursor flicker between two pixels.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        with self._lock:
            ix = self._ix if (self._ix is not None and abs(x - self._ix) <= 0.65) else int(round(x))
            iy = self._iy if (self._iy is not None and abs(y - self._iy) <= 0.65) else int(round(y))
            if ix != self._ix or iy != self._iy:
                self._mouse.position = (ix, iy)
                self._ix, self._iy = ix, iy
            self._last_pos = (x, y)

    def read_os_position(self) -> Tuple[float, float]:
        """Where the OS cursor really is right now (also follows a physical mouse)."""
        with self._lock:
            try:
                px, py = self._mouse.position
                self._ix, self._iy = int(px), int(py)
                return float(px), float(py)
            except Exception:
                return self._last_pos

    def get_position(self) -> Tuple[float, float]:
        with self._lock:
            return self._last_pos

    # ------------------------------------------------------------------
    # Click
    # ------------------------------------------------------------------

    def left_click(self) -> None:
        with self._lock:
            self._mouse.press(Button.left)
            self._mouse.release(Button.left)

    def right_click(self) -> None:
        with self._lock:
            self._mouse.press(Button.right)
            self._mouse.release(Button.right)

    def double_click(self) -> None:
        with self._lock:
            self._mouse.click(Button.left, 2)

    # ------------------------------------------------------------------
    # Smooth Scrolling (Touchpad Fractional Accumulator)
    # ------------------------------------------------------------------

    def scroll(self, dx: float, dy: float) -> None:
        """
        Accumulate fractional wheel ticks and send whole ticks.

        Sign convention (pynput): dy > 0 scrolls up, dy < 0 scrolls down.

        - The remainder is kept while the hand is moving, so slow strokes add up
          to real ticks (the old 0.9 decay on every quiet frame ate them).
        - It is cleared only after SCROLL_IDLE_CLEAR_SECONDS of stillness and when
          the direction reverses, so scrolling stops with the hand.
        - At most SCROLL_MAX_TICKS_PER_FRAME are sent per frame; the unsent part
          waits (up to SCROLL_MAX_BACKLOG) instead of being thrown away.
        """
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return
        now = time.perf_counter()

        with self._lock:
            if dx != 0.0 or dy != 0.0:
                self._scroll_last_input_t = now
                if (self._scroll_acc_x > 0 and dx < 0) or (self._scroll_acc_x < 0 and dx > 0):
                    self._scroll_acc_x = 0.0
                if (self._scroll_acc_y > 0 and dy < 0) or (self._scroll_acc_y < 0 and dy > 0):
                    self._scroll_acc_y = 0.0
                self._scroll_acc_x += dx
                self._scroll_acc_y += dy
            elif (now - self._scroll_last_input_t) > config.SCROLL_IDLE_CLEAR_SECONDS:
                self._scroll_acc_x = 0.0
                self._scroll_acc_y = 0.0

            self._scroll_acc_x, ticks_x = self._take_ticks(self._scroll_acc_x)
            self._scroll_acc_y, ticks_y = self._take_ticks(self._scroll_acc_y)

            if ticks_x != 0 or ticks_y != 0:
                self._mouse.scroll(ticks_x, ticks_y)

    @staticmethod
    def _take_ticks(acc: float) -> Tuple[float, int]:
        """Remove up to SCROLL_MAX_TICKS_PER_FRAME whole ticks from acc."""
        limit = config.SCROLL_MAX_TICKS_PER_FRAME
        whole = int(math.trunc(acc))
        ticks = max(-limit, min(limit, whole))
        acc -= ticks
        backlog = config.SCROLL_MAX_BACKLOG
        acc = max(-backlog, min(backlog, acc))
        return acc, ticks

    def reset_scroll_accumulator(self) -> None:
        """Reset fractional accumulators when entering/exiting scroll."""
        with self._lock:
            self._scroll_acc_x = 0.0
            self._scroll_acc_y = 0.0
