# main.py -- Entry point: ties all modules together for the Hand Gesture Mouse Controller.
#
# Architecture:
#   ThreadedCamera  ->  HandDetector  ->  GestureEngine
#                                              |
#                     OneEuroFilter2D  +  DeadzoneFilter  +  CursorFreezer
#                                              |
#                                      OSMouseController  (pynput)
#                                              |
#                                         HUDOverlay  (cv2.imshow)
#
from __future__ import annotations

import sys
import time
from typing import Tuple

import cv2
import numpy as np

import config
from camera_thread import ThreadedCamera
from gesture_engine import GestureEngine, GestureState
from hand_detector import HandDetector
from hud_overlay import HUDOverlay
from os_controller import OSMouseController
from stabilizer import CursorFreezer, CursorPipeline


# ---------------------------------------------------------------------------
# Screen resolution detection (cross-platform, no extra dependency)
# ---------------------------------------------------------------------------

def get_screen_resolution() -> Tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        pass

    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        pass

    return 1920, 1080


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def compute_roi_bounds(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) pixel bounds of the active ROI."""
    x1 = int(config.ROI_MARGIN_X * frame_w)
    y1 = int(config.ROI_MARGIN_Y * frame_h)
    x2 = frame_w - x1
    y2 = frame_h - y1
    return x1, y1, x2, y2


def map_to_screen(
    px: float, py: float,
    roi: Tuple[int, int, int, int],
    screen_w: int, screen_h: int,
) -> Tuple[float, float]:
    """
    Map a pixel coordinate inside the ROI to full-screen coordinates.
    numpy.interp provides clamped linear interpolation automatically.
    """
    x1, y1, x2, y2 = roi
    sx = float(np.interp(px, [x1, x2], [0, screen_w - 1]))
    sy = float(np.interp(py, [y1, y2], [0, screen_h - 1]))
    sx = max(0.0, min(float(screen_w - 1), sx))
    sy = max(0.0, min(float(screen_h - 1), sy))
    return sx, sy


# ---------------------------------------------------------------------------
# Per-frame logic (kept separate from camera / window code so it can be tested)
# ---------------------------------------------------------------------------

class GestureMouseController:
    """
    Turns (hand, gesture state) into mouse actions.

    Continuity rule: whenever the cursor stops following the finger (scroll,
    click, pause, hand lost) the cursor pipeline is suspended. When tracking
    resumes it glides from the cursor's real position to the finger instead of
    teleporting there.
    """

    def __init__(self, screen_w: int, screen_h: int, roi, mouse_ctrl, engine=None) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.roi = roi
        self.mouse = mouse_ctrl
        self.engine = engine if engine is not None else GestureEngine()
        self.cursor = CursorPipeline()
        self.freezer = CursorFreezer()
        self.state: GestureState = GestureState.POINTING
        self._lost_frames: int = 0

    def step(self, hand) -> Tuple[GestureState, bool]:
        """Process one frame. Returns (state, clicked_this_frame)."""
        prev = self.state
        state = self.engine.update(hand)
        self.state = state

        # ── No hand: hold everything; after a few lost frames make the return smooth ──
        if hand is None:
            self._lost_frames += 1
            if self._lost_frames >= config.HAND_LOST_RESET_FRAMES:
                self.cursor.suspend()
                self.freezer.release()
            return state, False
        self._lost_frames = 0

        # ── Leaving a state ──
        if prev == GestureState.SCROLLING and state != GestureState.SCROLLING:
            self.mouse.reset_scroll_accumulator()
            self.cursor.suspend()
        if prev == GestureState.PINCH and state != GestureState.PINCH:
            self.freezer.release()
            self.cursor.suspend()

        # ── Paused (fist): cursor stays exactly where it is ──
        if state == GestureState.SLEEP:
            if prev != GestureState.SLEEP:
                self.mouse.reset_scroll_accumulator()
            self.cursor.suspend()
            self.freezer.release()
            return state, False

        clicked = False

        if state == GestureState.POINTING:
            self.freezer.release()
            if self.engine.cursor_hold:
                # Hand is closing into a fist / forming the scroll pose: do not follow the fingertip
                self.cursor.suspend()
                return state, False

            if not self.cursor.active:
                self.cursor.sync(self.mouse.read_os_position())
            raw_x, raw_y = hand.index_tip_f
            sx, sy = map_to_screen(raw_x, raw_y, self.roi, self.screen_w, self.screen_h)
            x, y = self.cursor.update(sx, sy)
            self.mouse.move_to(x, y)

        elif state == GestureState.PINCH:
            pos = self.cursor.position
            if pos is None:
                pos = self.mouse.read_os_position()
            self.freezer.freeze(pos[0], pos[1])
            fx, fy = self.freezer.apply(pos[0], pos[1])
            self.mouse.move_to(fx, fy)
            if self.engine.check_tap_click(hand):
                self.mouse.left_click()
                clicked = True
                print("[CLICK] at screen (%.0f, %.0f)" % (fx, fy))

        elif state == GestureState.SCROLLING:
            self.freezer.release()
            self.cursor.suspend()
            dx, dy = self.engine.get_scroll_delta(hand)
            self.mouse.scroll(dx, dy)

        return state, clicked


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("  Hand Gesture Mouse Controller")
    print("  Press  Q  or  ESC  to quit")
    print("=" * 60)

    screen_w, screen_h = get_screen_resolution()
    print("[INFO] Screen resolution: %dx%d" % (screen_w, screen_h))

    # Initialize all components
    camera = ThreadedCamera()
    try:
        camera.start()
    except RuntimeError as e:
        print("[FATAL] " + str(e))
        sys.exit(1)

    print("[INFO] Camera opened: %dx%d @ %dfps" % (
        config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT, config.CAPTURE_FPS))

    detector   = HandDetector()
    mouse_ctrl = OSMouseController()

    # Warm-up: wait for first valid frame (max 2 seconds)
    frame = None
    print("[INFO] Waiting for first camera frame...")
    for _ in range(200):
        frame = camera.read()
        if frame is not None:
            break
        time.sleep(0.01)

    if frame is None:
        print("[FATAL] Camera did not produce a frame within 2 seconds.")
        camera.stop()
        sys.exit(1)

    fh, fw = frame.shape[:2]
    roi = compute_roi_bounds(fw, fh)
    hud = HUDOverlay(fw, fh)

    print("[INFO] Frame size: %dx%d" % (fw, fh))
    print("[INFO] Active ROI: %s" % str(roi))
    print("[INFO] Ready -- show your hand to the camera!")

    brain = GestureMouseController(screen_w, screen_h, roi, mouse_ctrl)

    while camera.is_running():
        frame = camera.read()
        if frame is None:
            time.sleep(0.001)
            continue

        try:
            hand = detector.process_frame(frame)
            state, clicked = brain.step(hand)
            if clicked and hand is not None:
                hud.trigger_click_flash(hand.index_tip_px)

            display = hud.draw(frame, state, hand, camera.get_fps())
            cv2.imshow(config.HUD_WINDOW_NAME, display)

            # Check if HUD window was closed by the user
            if cv2.getWindowProperty(config.HUD_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user -- shutting down.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in config.QUIT_KEYS:
                print("[INFO] Quit key pressed -- shutting down.")
                break

        except Exception as e:
            print(f"[WARN] Frame loop error: {e}")
            time.sleep(0.005)

    print("[INFO] Stopping camera thread...")
    camera.stop()
    print("[INFO] Releasing MediaPipe resources...")
    detector.close()
    print("[INFO] Closing HUD window...")
    cv2.destroyAllWindows()
    print("[INFO] Shutdown complete. Goodbye!")


if __name__ == "__main__":
    run()
