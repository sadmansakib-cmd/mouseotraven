# hand_detector.py -- MediaPipe Hands wrapper with robust 3D/distance geometry.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HandData:
    """All per-frame hand information derived from MediaPipe landmarks."""

    # Raw normalized landmarks (x, y) in [0, 1]
    lm_norm: List[Tuple[float, float]] = field(default_factory=list)

    # 3D normalized landmarks (x, y, z) for accurate joint angle calculation
    lm_3d: List[Tuple[float, float, float]] = field(default_factory=list)

    # Pixel coordinates (int) for each of the 21 landmarks
    lm_px: List[Tuple[int, int]] = field(default_factory=list)

    # Which fingers are extended?  [Thumb, Index, Middle, Ring, Pinky]
    fingers_up: List[bool] = field(default_factory=lambda: [False] * 5)

    # Index fingertip pixel position (landmark 8)
    index_tip_px: Tuple[int, int] = (0, 0)

    # Same point as sub-pixel floats. The cursor uses this one: the int version
    # snaps to whole camera pixels, which the ROI->screen magnification turns
    # into ~5 px cursor steps.
    index_tip_f: Tuple[float, float] = (0.0, 0.0)

    # Scale-invariant pinch ratio: dist(thumb_tip, index_tip) / hand_size_px
    pinch_ratio: float = 1.0

    # Scale-invariant ratio: dist(index_tip, middle_tip) / hand_size_px
    index_middle_ratio: float = 1.0

    # Midpoint of index + middle tips (for scroll gesture tracking)
    scroll_midpoint_px: Tuple[int, int] = (0, 0)

    # Normalized midpoint (x, y) in [0, 1] for resolution-independent scrolling
    scroll_point_norm: Tuple[float, float] = (0.0, 0.0)

    # Strict tight closed fist indicator (all 4 tips curled tight to wrist)
    is_strict_fist: bool = False

    # Whether MediaPipe thinks this is a right hand (before mirror flip)
    is_right_hand: bool = True


# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe 21-point model)
# ---------------------------------------------------------------------------

_WRIST          = 0
_THUMB_CMC      = 1
_THUMB_MCP      = 2
_THUMB_IP       = 3
_THUMB_TIP      = 4
_INDEX_MCP      = 5
_INDEX_PIP      = 6
_INDEX_DIP      = 7
_INDEX_TIP      = 8
_MIDDLE_MCP     = 9
_MIDDLE_PIP     = 10
_MIDDLE_DIP     = 11
_MIDDLE_TIP     = 12
_RING_MCP       = 13
_RING_PIP       = 14
_RING_DIP       = 15
_RING_TIP       = 16
_PINKY_MCP      = 17
_PINKY_PIP      = 18
_PINKY_DIP      = 19
_PINKY_TIP      = 20


# ---------------------------------------------------------------------------
# HandDetector
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Wraps mediapipe.solutions.hands for single-hand real-time detection.

    Upgrades:
    1. Distance-from-wrist + 3D PIP joint angle prevents false curled-finger
       classification when fingers tilt forward/downward during scrolling.
    2. Strict tight fist detection ensures down-scroll flicks never trigger SLEEP.
    3. Joined-finger metric enforces index + middle touching side-by-side for scroll.
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            model_complexity=config.MP_MODEL_COMPLEXITY,
            max_num_hands=config.MP_MAX_NUM_HANDS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
        )

    def close(self) -> None:
        self._hands.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """
        Run MediaPipe inference on one BGR frame.

        Returns HandData if a hand is detected, else None.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        lm_proto = results.multi_hand_landmarks[0]
        is_right = True
        if results.multi_handedness and len(results.multi_handedness) > 0:
            try:
                classification = results.multi_handedness[0].classification
                if classification and len(classification) > 0:
                    is_right = (classification[0].label == "Right")
            except Exception:
                is_right = True

        # Build coordinate arrays
        lm_norm = [(lm.x, lm.y) for lm in lm_proto.landmark]
        lm_3d   = [(lm.x, lm.y, lm.z) for lm in lm_proto.landmark]
        lm_px   = [(int(lm.x * w), int(lm.y * h)) for lm in lm_proto.landmark]

        # Hand size for scale invariance: dist(wrist, middle_MCP)
        hand_size_px = self._dist(lm_px, _WRIST, _MIDDLE_MCP)
        if hand_size_px < 1.0:
            hand_size_px = 1.0

        # Robust distance + 3D angle based finger extension
        fingers_up = self._compute_fingers_up(lm_px, lm_3d, is_right)

        # Strict fist check
        strict_fist = self.is_strict_fist(lm_px)

        index_tip_px  = lm_px[_INDEX_TIP]
        middle_tip_px = lm_px[_MIDDLE_TIP]

        pinch_ratio     = self._dist(lm_px, _THUMB_TIP, _INDEX_TIP)  / hand_size_px
        index_mid_ratio = self._dist(lm_px, _INDEX_TIP, _MIDDLE_TIP) / hand_size_px

        # Scroll midpoint in pixel and normalized space:
        # Uses finger TIPS for direct, responsive stroke capture (no dilution from stationary PIP)
        mx = (index_tip_px[0] + middle_tip_px[0]) // 2
        my = (index_tip_px[1] + middle_tip_px[1]) // 2
        sp_norm_x = (lm_norm[_INDEX_TIP][0] + lm_norm[_MIDDLE_TIP][0]) / 2.0
        sp_norm_y = (lm_norm[_INDEX_TIP][1] + lm_norm[_MIDDLE_TIP][1]) / 2.0

        return HandData(
            lm_norm=lm_norm,
            lm_3d=lm_3d,
            lm_px=lm_px,
            fingers_up=fingers_up,
            index_tip_px=index_tip_px,
            index_tip_f=(lm_norm[_INDEX_TIP][0] * w, lm_norm[_INDEX_TIP][1] * h),
            pinch_ratio=pinch_ratio,
            index_middle_ratio=index_mid_ratio,
            scroll_midpoint_px=(mx, my),
            scroll_point_norm=(sp_norm_x, sp_norm_y),
            is_strict_fist=strict_fist,
            is_right_hand=is_right,
        )

    # ------------------------------------------------------------------
    # Gesture primitive checks
    # ------------------------------------------------------------------

    @staticmethod
    def is_strict_fist(lm_px: List[Tuple[int, int]]) -> bool:
        """
        True only when all 4 finger tips (8, 12, 16, 20) are strictly close
        to the wrist/palm base: dist(Tip, Wrist) < dist(MCP, Wrist) * 1.15.
        Prevents forward-tilted or downward-scrolling fingers from triggering fist.
        """
        wrist = lm_px[_WRIST]
        pairs = [
            (_INDEX_TIP,  _INDEX_MCP),
            (_MIDDLE_TIP, _MIDDLE_MCP),
            (_RING_TIP,   _RING_MCP),
            (_PINKY_TIP,  _PINKY_MCP),
        ]
        for tip_idx, mcp_idx in pairs:
            tip = lm_px[tip_idx]
            mcp = lm_px[mcp_idx]
            dist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
            dist_mcp = math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1])
            if dist_tip >= (dist_mcp * config.STRICT_FIST_DIST_RATIO):
                return False
        return True

    @staticmethod
    def is_open_palm(fingers_up: List[bool]) -> bool:
        """True when all five fingers are extended."""
        return all(fingers_up)

    @staticmethod
    def is_pointing(fingers_up: List[bool]) -> bool:
        """True when only the index finger is extended (thumb may relax)."""
        return fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]

    @staticmethod
    def is_scroll_gesture(fingers_up: List[bool], index_middle_ratio: float) -> bool:
        """
        True when Index AND Middle are UP, Pinky is DOWN,
        Index & Middle are joined side-by-side, and not an open palm.
        Permits natural relaxed ring finger so scroll indicator never dies after multiple videos.
        """
        index_up = fingers_up[1]
        mid_up   = fingers_up[2]
        pinky_down = not fingers_up[4]
        not_palm = not all(fingers_up)
        joined_ok = index_middle_ratio < config.SCROLL_ENTRY_JOINED_RATIO
        return index_up and mid_up and pinky_down and not_palm and joined_ok

    # ------------------------------------------------------------------
    # Geometric computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(lm_px: List[Tuple[int, int]], a: int, b: int) -> float:
        ax, ay = lm_px[a]
        bx, by = lm_px[b]
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def _joint_angle_deg(
        lm_3d: List[Tuple[float, float, float]],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
    ) -> float:
        """
        Calculate 3D joint angle at the PIP joint between (MCP -> PIP) and (TIP -> PIP).
        A straight/extended finger has an angle near 180 degrees (> 130 degrees).
        A curled finger bends sharply with an angle < 100 degrees.
        """
        p = lm_3d[pip_idx]
        m = lm_3d[mcp_idx]
        t = lm_3d[tip_idx]

        v1 = (m[0] - p[0], m[1] - p[1], m[2] - p[2])
        v2 = (t[0] - p[0], t[1] - p[1], t[2] - p[2])

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    @classmethod
    def _is_finger_extended(
        cls,
        lm_px: List[Tuple[int, int]],
        lm_3d: List[Tuple[float, float, float]],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
    ) -> bool:
        """
        Determine if finger is extended using distance from wrist, 3D PIP angle,
        relative MCP extension, and 2D projection fallback.
        Works reliably in all directions (up, downward stroke, tilted towards camera).
        """
        wrist = lm_px[_WRIST]
        tip   = lm_px[tip_idx]
        pip   = lm_px[pip_idx]
        mcp   = lm_px[mcp_idx]

        dist_wrist_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        dist_wrist_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
        dist_wrist_mcp = math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1])

        dist_mcp_tip = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])
        dist_mcp_pip = math.hypot(pip[0] - mcp[0], pip[1] - mcp[1])

        # Metric 1: Distance from wrist (Tip at least as far as PIP)
        cond_dist = dist_wrist_tip > (dist_wrist_pip * config.FINGER_WRIST_DIST_RATIO)

        # Metric 2: 3D joint curl angle at PIP (> 110 deg = uncurled/straight)
        angle = cls._joint_angle_deg(lm_3d, mcp_idx, pip_idx, tip_idx)
        cond_angle = angle > config.FINGER_ANGLE_EXTENDED_DEG

        # Metric 3: Upright 2D check (tip.y < pip.y and clearly beyond MCP)
        cond_upright = (tip[1] < pip[1]) and (dist_wrist_tip > dist_wrist_mcp * 1.05)

        # Metric 4: Relative extension from MCP (Tip extends well beyond PIP in any direction, including downward strokes)
        cond_mcp_ext = dist_mcp_tip > (dist_mcp_pip * 1.30)

        # Finger is extended if it's straight, uncurled, or extending outward from MCP
        return cond_dist or cond_angle or cond_upright or cond_mcp_ext

    @classmethod
    def _compute_fingers_up(
        cls,
        lm_px: List[Tuple[int, int]],
        lm_3d: List[Tuple[float, float, float]],
        is_right: bool,
    ) -> List[bool]:
        """
        Calculate extension status for all five fingers:
        [Thumb, Index, Middle, Ring, Pinky]
        """
        # Thumb: compare tip and IP X-coordinate (mirrored), plus distance from wrist
        wrist = lm_px[_WRIST]
        thumb_tip = lm_px[_THUMB_TIP]
        thumb_ip  = lm_px[_THUMB_IP]
        thumb_mcp = lm_px[_THUMB_MCP]

        dist_wrist_thumb = math.hypot(thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
        dist_wrist_tmcp  = math.hypot(thumb_mcp[0] - wrist[0], thumb_mcp[1] - wrist[1])

        if is_right:
            cond_thumb_x = thumb_tip[0] < thumb_ip[0]
        else:
            cond_thumb_x = thumb_tip[0] > thumb_ip[0]

        thumb_angle = cls._joint_angle_deg(lm_3d, _THUMB_CMC, _THUMB_MCP, _THUMB_TIP)
        thumb_up = (cond_thumb_x and dist_wrist_thumb > dist_wrist_tmcp * 1.05) or (thumb_angle > 140.0)

        index_up  = cls._is_finger_extended(lm_px, lm_3d, _INDEX_MCP,  _INDEX_PIP,  _INDEX_TIP)
        middle_up = cls._is_finger_extended(lm_px, lm_3d, _MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP)
        ring_up   = cls._is_finger_extended(lm_px, lm_3d, _RING_MCP,   _RING_PIP,   _RING_TIP)
        pinky_up  = cls._is_finger_extended(lm_px, lm_3d, _PINKY_MCP,  _PINKY_PIP,  _PINKY_TIP)

        return [thumb_up, index_up, middle_up, ring_up, pinky_up]
