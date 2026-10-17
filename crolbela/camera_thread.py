# camera_thread.py -- Threaded webcam capture with queue size 1 to avoid stale frames.
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2

import config


class ThreadedCamera:
    """
    Non-blocking webcam reader using a background daemon thread.

    Design decisions:
    - Queue(maxsize=1): old frames are discarded; the main loop always gets the
      latest frame with zero latency.
    - Frame is horizontally flipped (FLIP_CODE=1) so it acts as a mirror.
      This makes cursor movement intuitive (move hand right -> cursor moves right).
    - FPS is measured over a rolling 1-second window.
    - The thread is a daemon so it auto-exits if the main process dies.
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running: bool = False
        self._lock = threading.Lock()

        # FPS tracking
        self._frame_count: int = 0
        self._fps_ts: float = time.perf_counter()
        self._current_fps: float = 0.0

    def start(self) -> "ThreadedCamera":
        """Open the camera and start the capture daemon thread."""
        self._cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera ID {config.CAMERA_ID}. "
                "Check that the webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.CAPTURE_FPS)
        # Disable internal OpenCV buffer (use our own queue)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the capture thread to stop and release camera resources."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self):
        """
        Return the latest captured frame (BGR ndarray), or None if none available.
        Non-blocking: returns immediately.
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_fps(self) -> float:
        """Return the measured camera capture FPS."""
        return self._current_fps

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """Background thread: grab frames as fast as the camera allows."""
        while self.is_running():
            try:
                if self._cap is None:
                    time.sleep(0.01)
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                # Mirror: horizontal flip so the view is like a mirror.
                frame = cv2.flip(frame, 1)

                # Drop stale frame if consumer hasn't picked up the previous one.
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()   # discard old
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass

                # Rolling FPS measurement
                self._frame_count += 1
                elapsed = time.perf_counter() - self._fps_ts
                if elapsed >= 1.0:
                    self._current_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_ts = time.perf_counter()

            except Exception:
                time.sleep(0.01)
