# gesture_engine.py -- State Machine for gesture recognition with stroke scrolling.
from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Optional, Tuple

import config
from hand_detector import HandData, HandDetector


class GestureState(Enum):
    SLEEP     = auto()   # True tight fist held ~1.3s -- system paused
    POINTING  = auto()   # Index extended -- cursor moves
    PINCH     = auto()   # Thumb approaching index -- click zone
    SCROLLING = auto()   # Joined index + middle fingers -- touchpad stroke scroll


class GestureEngine:
    """
    Debounced Gesture State Machine with Stroke-Based Scrolling & Fast Clicks.

    Key Features:
    - Stroke / Swipe Scrolling: Only moves when you actively stroke your fingers.
      Stationary fingers produce ZERO scroll (deadband eliminates auto-runaway).
      Stroking UP scrolls content DOWN; stroking DOWN scrolls content UP.
    - Fast Double-Tap: 120ms cooldown and instant unfreeze on release enable
      snappy, responsive single clicks and double clicks with zero lag.
    - Strict Fist: 10-frame hold requirement ensures quick swipes never trigger SLEEP.
    """

    def __init__(self) -> None:
        self.current_state: GestureState = GestureState.POINTING

        self._candidate_state: GestureState = GestureState.POINTING
        self._candidate_count: int = 0

        # Strict fist hold counter (leaky: a few bad frames slow it, they do not reset it)
        self._fist_hold_count: int = 0
        self._wake_count: int = 0

        # True while the cursor must NOT follow the fingertip (fist closing, scroll pose
        # being confirmed, ...). main.py reads this so the cursor stays where it was.
        self.cursor_hold: bool = False

        # Click debounce and single-tap latch tracking
        self._click_cooldown_ts: float = 0.0
        self._click_ready: bool = False
        self._click_latched: bool = False

        # Stroke scroll tracking, reversal suppression and stabilization
        self._prev_scroll_point: Optional[Tuple[float, float]] = None
        self._scroll_sm_x: Optional[float] = None
        self._scroll_sm_y: Optional[float] = None
        self._scroll_prev_t: Optional[float] = None
        self._scroll_enter_count: int = 0
        self._scroll_exit_count: int = 0
        self._scroll_posture_ok: bool = True   # this frame still looks like a scroll pose
        self._stroke_intent_dir: int = 0       # 1 for moving down, -1 for moving up
        self._last_motion_time: float = 0.0    # last time the hand moved (any direction)

        self._in_pinch: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, hand: Optional[HandData]) -> GestureState:
        """Process one frame and return the confirmed debounced gesture state."""
        if hand is None:
            self.reset_scroll_origin()
            self._scroll_enter_count = 0
            self._wake_count = 0
            self._fist_hold_count = max(0, self._fist_hold_count - 2)
            self.cursor_hold = True
            return self.current_state

        # ── Paused: only a deliberate wake pose held for a few frames wakes us ──
        if self.current_state == GestureState.SLEEP:
            self.cursor_hold = True
            if self._is_wake_pose(hand):
                self._wake_count += 1
                if self._wake_count >= config.WAKE_PERSIST_FRAMES:
                    self._wake_count = 0
                    self._fist_hold_count = 0
                    self.current_state = GestureState.POINTING
                    self._candidate_state = GestureState.POINTING
                    self._candidate_count = config.GESTURE_PERSIST_FRAMES
                    self.cursor_hold = False
            else:
                self._wake_count = 0
            return self.current_state

        # ── Scroll Latch Handling ─────────────────────────────────────
        if self.current_state == GestureState.SCROLLING:
            if self._should_keep_scroll_latched(hand):
                self._fist_hold_count = 0
                self.cursor_hold = True
                return GestureState.SCROLLING
            self.current_state = GestureState.POINTING
            self._candidate_state = GestureState.POINTING
            self._candidate_count = 0
            self.reset_scroll_origin()
            self.cursor_hold = False
            return GestureState.POINTING

        # ── Normal State Machine Transitions ──────────────────────────
        raw = self._classify_raw(hand)
        self._advance_state_machine(raw, hand)

        # Hold the cursor while the pose is changing into pause / scroll, otherwise the
        # fingertip (which moves a lot while the hand closes) drags the cursor away.
        fu = hand.fingers_up
        self.cursor_hold = (
            self._fist_hold_count > 0
            or hand.is_strict_fist
            or not any(fu[1:])
            or (raw == GestureState.SCROLLING and self.current_state != GestureState.SCROLLING)
        )
        return self.current_state

    @staticmethod
    def _is_wake_pose(hand: HandData) -> bool:
        fu = hand.fingers_up
        return (
            not hand.is_strict_fist
            and (all(fu)
                 or HandDetector.is_pointing(fu)
                 or HandDetector.is_scroll_gesture(fu, hand.index_middle_ratio))
        )

    def should_click(self) -> bool:
        """True once per confirmed pinch click. Auto-resets after read."""
        if self._click_ready:
            self._click_ready = False
            return True
        return False

    def check_tap_click(self, hand: HandData) -> bool:
        """
        Fast single-tap click detection:
        - Fires on first contact (pinch_ratio < PINCH_THRESHOLD_RATIO).
        - Latches to prevent accidental multi-clicks while holding.
        - Re-arms when fingers separate (pinch_ratio > PINCH_RELEASE_RATIO).
        - Cooldown allows fast double-taps.
        """
        now = time.perf_counter()

        if hand.pinch_ratio > config.PINCH_RELEASE_RATIO:
            self._click_latched = False
            return False

        if hand.pinch_ratio < config.PINCH_THRESHOLD_RATIO:
            if not self._click_latched and (now - self._click_cooldown_ts >= config.PINCH_COOLDOWN_SECONDS):
                self._click_cooldown_ts = now
                self._click_latched = True
                return True

        return False

    def get_scroll_delta(self, hand: HandData) -> Tuple[float, float]:
        """
        Stroke-based (dx, dy) in wheel ticks for this frame.

        Pipeline: time-aware smoothing -> hand speed -> soft speed deadband ->
        axis lock -> return-stroke guard -> displacement * sensitivity.

        - Everything is measured in seconds, so a slow camera does not change the feel.
        - The deadband acts on SPEED with a soft edge, so slow deliberate strokes
          still scroll (the old per-frame deadband swallowed them) while a still
          hand gives exactly zero.
        - Return-stroke guard: after moving one way, the opposite direction is ignored
          until the hand has rested SCROLL_REVERSAL_PAUSE seconds. Sideways wobble
          can no longer poison it (that was a bug) and a long return motion stays
          ignored for as long as it lasts.
        """
        now = time.perf_counter()
        cur_x, cur_y = hand.scroll_point_norm

        if (self._scroll_sm_x is None or self._scroll_sm_y is None
                or self._scroll_prev_t is None or (now - self._scroll_prev_t) > 0.25):
            self._scroll_sm_x, self._scroll_sm_y = cur_x, cur_y
            self._scroll_prev_t = now
            return 0.0, 0.0

        dt = min(max(now - self._scroll_prev_t, 1.0 / 240.0), 0.1)
        self._scroll_prev_t = now

        a = 1.0 - math.exp(-dt / config.SCROLL_SMOOTH_TAU)
        new_x = self._scroll_sm_x + a * (cur_x - self._scroll_sm_x)
        new_y = self._scroll_sm_y + a * (cur_y - self._scroll_sm_y)
        vx = (new_x - self._scroll_sm_x) / dt
        vy = (new_y - self._scroll_sm_y) / dt
        self._scroll_sm_x, self._scroll_sm_y = new_x, new_y

        # Pose is not a clean scroll pose this frame (about to exit): no output
        if not self._scroll_posture_ok:
            return 0.0, 0.0

        # Speed knee: still hand -> 0, then a smooth ramp, then full gain. Continuous,
        # so there is no sudden onset, and normal strokes keep the full sensitivity.
        speed_y = abs(vy)
        vy = vy * self._knee(speed_y, 1.0)
        vx = vx * self._knee(abs(vx), 1.5)

        # Axis lock: vertical unless horizontal is clearly dominant
        ax, ay = abs(vx), abs(vy)
        if ax > ay * config.SCROLL_HORIZONTAL_DOMINANCE and ax > 0.0:
            vy = 0.0
        else:
            vx = 0.0

        # Return-stroke guard (vertical only; horizontal never touches the intent)
        if vy != 0.0:
            direction = 1 if vy > 0 else -1
            rested = (now - self._last_motion_time) >= config.SCROLL_REVERSAL_PAUSE
            if speed_y >= config.SCROLL_MOTION_SPEED:
                self._last_motion_time = now   # any real motion (even an ignored one) restarts the rest timer
            if self._stroke_intent_dir == 0 or direction == self._stroke_intent_dir or rested:
                self._stroke_intent_dir = direction
            else:
                vy = 0.0   # return motion after a stroke: ignore it

        dir_mult_y = 1.0 if config.SCROLL_INVERT_Y else -1.0
        out_x = vx * dt * config.SCROLL_HORIZONTAL_SENSITIVITY
        out_y = vy * dt * config.SCROLL_STROKE_SENSITIVITY * dir_mult_y
        return out_x, out_y

    @staticmethod
    def _knee(speed: float, scale: float) -> float:
        """0 below SCROLL_SPEED_MIN, smoothstep ramp up to 1 at SCROLL_SPEED_FULL."""
        lo = config.SCROLL_SPEED_MIN * scale
        hi = config.SCROLL_SPEED_FULL * scale
        if speed <= lo:
            return 0.0
        if speed >= hi:
            return 1.0
        s = (speed - lo) / (hi - lo)
        return s * s * (3.0 - 2.0 * s)

    def reset_scroll_origin(self) -> None:
        """Anchor the scroll start. Call when entering or leaving SCROLLING."""
        self._scroll_sm_x = None
        self._scroll_sm_y = None
        self._scroll_prev_t = None
        self._scroll_exit_count = 0
        self._scroll_posture_ok = True
        self._stroke_intent_dir = 0
        self._last_motion_time = 0.0

    def is_pre_pinch(self, hand: HandData) -> bool:
        return hand.pinch_ratio < config.PINCH_PRE_PINCH_RATIO

    def is_full_pinch(self, hand: HandData) -> bool:
        return hand.pinch_ratio < config.PINCH_THRESHOLD_RATIO

    def can_click(self) -> bool:
        return (time.perf_counter() - self._click_cooldown_ts) >= config.PINCH_COOLDOWN_SECONDS

    def record_click(self) -> None:
        self._click_cooldown_ts = time.perf_counter()
        self._click_ready = False

    # ------------------------------------------------------------------
    # Scroll Latch & Classification Logic
    # ------------------------------------------------------------------

    def _should_keep_scroll_latched(self, hand: HandData) -> bool:
        """
        Keep SCROLLING latched during finger stroke swipes.
        During downward swipes, fingers bend/flex together at the knuckles.
        Maintains scroll latch as long as fingers remain together side-by-side.
        Unlatches cleanly when:
        1. User curls middle finger to return to POINTING (fu[1] and not fu[2] and not fu[3] and not fu[4]).
        2. User spreads fingers into a wide V-shape (> SCROLL_EXIT_SPREAD_RATIO).
        3. User clenches into a strict fist.
        4. User opens entire palm.
        """
        fu = hand.fingers_up
        pinky_down = not fu[4]
        not_spread = hand.index_middle_ratio <= config.SCROLL_EXIT_SPREAD_RATIO
        not_fist = not hand.is_strict_fist
        not_open_palm = not HandDetector.is_open_palm(fu)

        # Explicit return to pointing: Index finger extended while middle, ring, pinky are curled
        is_explicit_pointing = fu[1] and not fu[2] and not fu[3] and not fu[4]

        # While in scrolling, fingers stroke/bend together.
        # As long as fingers are together (not spread into V), pinky is down,
        # not a clenched fist, not open palm, and not explicitly pointing:
        is_scrolling = not_spread and not_fist and not_open_palm and pinky_down and not is_explicit_pointing

        # Leaky counter: a bad frame adds 1, a good frame removes 1. One or two noisy
        # frames never end the scroll; a real change of pose does within a few frames.
        self._scroll_posture_ok = is_scrolling
        if not is_scrolling:
            self._scroll_exit_count += 1
            if self._scroll_exit_count >= config.SCROLL_EXIT_FRAMES:
                self._scroll_exit_count = 0
                return False
        else:
            self._scroll_exit_count = max(0, self._scroll_exit_count - 1)
        return True

    def _classify_raw(self, hand: HandData) -> GestureState:
        """Classify one frame into a candidate gesture state."""
        fu = hand.fingers_up

        # Priority 1: Strict Fist -> SLEEP candidate
        if hand.is_strict_fist:
            return GestureState.SLEEP

        # Priority 2: Joined scroll gesture (Index + Middle touching side-by-side)
        if HandDetector.is_scroll_gesture(fu, hand.index_middle_ratio):
            return GestureState.SCROLLING

        # Priority 3: Pinch Click (Thumb touches Index)
        # Ring and pinky must be curled into palm; middle must be curled (not in scroll gesture)
        middle_down = not fu[2]
        ring_pinky_down = not fu[3] and not fu[4]
        if middle_down and ring_pinky_down:
            # Only enter PINCH on actual contact or while latched in contact
            if hand.pinch_ratio < config.PINCH_THRESHOLD_RATIO or self._click_latched:
                return GestureState.PINCH

        # Priority 4: Pointing
        if HandDetector.is_pointing(fu):
            return GestureState.POINTING

        return GestureState.POINTING

    def _advance_state_machine(self, candidate: GestureState, hand: HandData) -> None:
        """Apply debounce rules, strict fist 10-frame hold, and transitions."""

        # ── Strict Fist Handling (leaky hold counter) ─────────────────
        if candidate == GestureState.SLEEP and hand.is_strict_fist:
            self._scroll_enter_count = 0
            self._fist_hold_count += 1
            if self._fist_hold_count >= config.FIST_PERSIST_FRAMES:
                self._enter_sleep()
            return
        else:
            self._fist_hold_count = max(0, self._fist_hold_count - 2)

        # ── Fast Pinch Handling: enter and exit immediately ────────────
        if candidate == GestureState.PINCH and self.current_state == GestureState.POINTING:
            self.current_state = GestureState.PINCH
            self._candidate_state = GestureState.PINCH
            self._candidate_count = config.GESTURE_PERSIST_FRAMES
            return

        if candidate == GestureState.POINTING and self.current_state == GestureState.PINCH:
            # Immediate unfreeze upon finger release
            self.current_state = GestureState.POINTING
            self._candidate_state = GestureState.POINTING
            self._candidate_count = config.GESTURE_PERSIST_FRAMES
            return

        # ── SCROLLING entry: pose must be confirmed for SCROLL_ENTER_FRAMES ──
        if candidate == GestureState.SCROLLING and self.current_state != GestureState.SCROLLING:
            self._scroll_enter_count += 1
            if self._scroll_enter_count >= config.SCROLL_ENTER_FRAMES:
                self._scroll_enter_count = 0
                prev_state = self.current_state
                self.current_state = GestureState.SCROLLING
                self._candidate_state = GestureState.SCROLLING
                self._candidate_count = config.GESTURE_PERSIST_FRAMES
                self._on_state_transition(prev_state, GestureState.SCROLLING)
                self._scroll_sm_x = hand.scroll_point_norm[0]
                self._scroll_sm_y = hand.scroll_point_norm[1]
                self._scroll_prev_t = time.perf_counter()
            return
        self._scroll_enter_count = 0

        # Standard debounced transitions
        if candidate == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = candidate
            self._candidate_count = 1

        if self._candidate_count >= config.GESTURE_PERSIST_FRAMES:
            prev_state = self.current_state
            self._on_state_transition(prev_state, candidate)
            self.current_state = candidate

    def _on_state_transition(self, prev: GestureState, new: GestureState) -> None:
        if new == GestureState.SCROLLING and prev != GestureState.SCROLLING:
            self.reset_scroll_origin()
        if new != GestureState.PINCH and prev == GestureState.PINCH:
            self._in_pinch = False

    def _enter_sleep(self) -> None:
        if self.current_state != GestureState.SLEEP:
            self._wake_count = 0
            self.current_state = GestureState.SLEEP
            self._candidate_state = GestureState.SLEEP
            self._candidate_count = config.GESTURE_PERSIST_FRAMES
            self.reset_scroll_origin()
