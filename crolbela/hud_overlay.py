# hud_overlay.py -- Debug HUD drawn onto the OpenCV preview window.
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from gesture_engine import GestureState
from hand_detector import HandData


_GREEN   = config.COLOR_GREEN
_RED     = config.COLOR_RED
_YELLOW  = config.COLOR_YELLOW
_CYAN    = config.COLOR_CYAN
_WHITE   = config.COLOR_WHITE
_ORANGE  = config.COLOR_ORANGE
_GRAY    = config.COLOR_GRAY
_DARK    = config.COLOR_DARK
_BLUE    = config.COLOR_BLUE

_STATE_INFO = {
    GestureState.SLEEP:    ("SYSTEM PAUSED (FIST)",          _RED),
    GestureState.POINTING: ("POINTING -- CURSOR MOVE",        _GREEN),
    GestureState.PINCH:    ("PINCH -- CLICK READY",          _YELLOW),
    GestureState.SCROLLING:("SCROLLING (JOINED 2-FINGER)",    _CYAN),
}

_CLICK_FLASH_DURATION = 0.25


class HUDOverlay:
    """Renders all debug HUD elements onto a BGR camera frame."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self._fw = frame_w
        self._fh = frame_h

        mx = int(config.ROI_MARGIN_X * frame_w)
        my = int(config.ROI_MARGIN_Y * frame_h)
        self._roi_pt1 = (mx, my)
        self._roi_pt2 = (frame_w - mx, frame_h - my)

        self._click_flash_ts: float = 0.0
        self._click_flash_pos: Optional[Tuple[int, int]] = None

        self._proc_frame_count: int = 0
        self._proc_fps_ts: float = time.perf_counter()
        self._proc_fps: float = 0.0

    def trigger_click_flash(self, pos: Tuple[int, int]) -> None:
        self._click_flash_ts = time.perf_counter()
        self._click_flash_pos = pos

    def draw(
        self,
        frame: np.ndarray,
        state: GestureState,
        hand: Optional[HandData],
        camera_fps: float,
    ) -> np.ndarray:
        canvas = frame.copy()

        self._proc_frame_count += 1
        elapsed = time.perf_counter() - self._proc_fps_ts
        if elapsed >= 1.0:
            self._proc_fps = self._proc_frame_count / elapsed
            self._proc_frame_count = 0
            self._proc_fps_ts = time.perf_counter()

        if state == GestureState.SLEEP:
            canvas = self._draw_sleep_overlay(canvas)

        if config.HUD_SHOW_ROI_BOX:
            self._draw_roi_box(canvas)

        if config.HUD_SHOW_LANDMARKS and hand is not None:
            self._draw_landmarks(canvas, hand, state)

        if hand is not None:
            self._draw_click_flash(canvas, hand)

        if hand is not None and state in (GestureState.POINTING, GestureState.PINCH):
            self._draw_pinch_bar(canvas, hand.pinch_ratio)

        self._draw_status_panel(canvas, state, hand, camera_fps)

        return canvas

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_sleep_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._fw, self._fh), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)

        text = "SYSTEM PAUSED"
        sub  = "Open palm to resume"
        font = cv2.FONT_HERSHEY_SIMPLEX

        tw = cv2.getTextSize(text, font, 1.4, 3)[0][0]
        sw = cv2.getTextSize(sub,  font, 0.65, 2)[0][0]
        cx = self._fw // 2

        cv2.putText(frame, text, (cx - tw // 2, self._fh // 2 - 20), font, 1.4, _RED, 3, cv2.LINE_AA)
        cv2.putText(frame, sub,  (cx - sw // 2, self._fh // 2 + 30), font, 0.65, _YELLOW, 2, cv2.LINE_AA)
        return frame

    def _draw_roi_box(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, self._roi_pt1, self._roi_pt2, _CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Active ROI",
                    (self._roi_pt1[0] + 4, self._roi_pt1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA)

    def _draw_landmarks(self, frame: np.ndarray, hand: HandData, state: GestureState) -> None:
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        for a, b in CONNECTIONS:
            cv2.line(frame, hand.lm_px[a], hand.lm_px[b], (200, 200, 200), 1, cv2.LINE_AA)

        tips = [4, 8, 12, 16, 20]
        for lm_idx, up in zip(tips, hand.fingers_up):
            px, py = hand.lm_px[lm_idx]
            color = _GREEN if up else _GRAY
            cv2.circle(frame, (px, py), 8, color, -1)
            cv2.circle(frame, (px, py), 8, _WHITE, 1, cv2.LINE_AA)

        for i, (px, py) in enumerate(hand.lm_px):
            if i not in tips:
                cv2.circle(frame, (px, py), 4, _WHITE, -1)

        # Highlight joined scroll fingers when in scroll mode
        if state == GestureState.SCROLLING:
            cv2.line(frame, hand.lm_px[8], hand.lm_px[12], _CYAN, 3, cv2.LINE_AA)
            mx, my = hand.scroll_midpoint_px
            cv2.circle(frame, (mx, my), 8, _CYAN, -1)
            cv2.circle(frame, (mx, my), 10, _WHITE, 2, cv2.LINE_AA)
        else:
            mx, my = hand.scroll_midpoint_px
            cv2.circle(frame, (mx, my), 6, _BLUE, -1)

    def _draw_click_flash(self, frame: np.ndarray, hand: HandData) -> None:
        elapsed = time.perf_counter() - self._click_flash_ts
        if elapsed < _CLICK_FLASH_DURATION and self._click_flash_pos is not None:
            alpha = 1.0 - (elapsed / _CLICK_FLASH_DURATION)
            radius = int(20 + 15 * (1.0 - alpha))
            color = tuple(int(c * alpha) for c in _YELLOW)
            px, py = self._click_flash_pos
            cv2.circle(frame, (px, py), radius, color, 3, cv2.LINE_AA)

    def _draw_pinch_bar(self, frame: np.ndarray, ratio: float) -> None:
        bar_x, bar_y = self._fw - 120, 55
        bar_w, bar_h = 100, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _DARK, -1)
        clamp = max(0.0, min(1.0, 1.0 - (ratio / config.PINCH_PRE_PINCH_RATIO)))
        fill_w = int(bar_w * clamp)
        bar_color = (
            _RED    if ratio < config.PINCH_THRESHOLD_RATIO else
            _YELLOW if ratio < config.PINCH_PRE_PINCH_RATIO else
            _GREEN
        )
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _WHITE, 1)
        cv2.putText(frame, "Pinch", (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _WHITE, 1, cv2.LINE_AA)

    def _draw_status_panel(
        self, frame: np.ndarray, state: GestureState,
        hand: Optional[HandData], camera_fps: float,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = config.HUD_FONT_SCALE
        ft = config.HUD_FONT_THICKNESS

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._fw, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        state_text, state_color = _STATE_INFO.get(state, ("UNKNOWN", _GRAY))
        cv2.putText(frame, "STATE: " + state_text, (8, 22), font, fs, state_color, ft, cv2.LINE_AA)

        fps_text = "CAM %3.0ffps  PROC %3.0ffps" % (camera_fps, self._proc_fps)
        tw = cv2.getTextSize(fps_text, font, 0.48, 1)[0][0]
        cv2.putText(frame, fps_text, (self._fw - tw - 6, 22), font, 0.48, _GRAY, 1, cv2.LINE_AA)

        if hand is not None:
            labels = ["T", "I", "M", "R", "P"]
            for i, (label, up) in enumerate(zip(labels, hand.fingers_up)):
                color = _GREEN if up else _RED
                cv2.putText(frame, label, (8 + i * 22, 44), font, 0.5, color, 1, cv2.LINE_AA)

            # If in scrolling mode or 2 fingers are up, display joined finger ratio indicator
            if state == GestureState.SCROLLING:
                ratio_text = "SCROLLING: %.2f" % hand.index_middle_ratio
                cv2.putText(frame, ratio_text, (130, 44), font, 0.45, _CYAN, 1, cv2.LINE_AA)
            elif hand.fingers_up[1] and hand.fingers_up[2]:
                ratio_text = "JOIN TO SCROLL: %.2f" % hand.index_middle_ratio
                cv2.putText(frame, ratio_text, (130, 44), font, 0.45, _YELLOW, 1, cv2.LINE_AA)

        hint = "Q/ESC: Quit"
        hw = cv2.getTextSize(hint, font, 0.38, 1)[0][0]
        cv2.putText(frame, hint, (self._fw - hw - 6, self._fh - 8), font, 0.38, _GRAY, 1, cv2.LINE_AA)
