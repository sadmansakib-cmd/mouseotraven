# config.py -- All tunable hyperparameters for the Hand Gesture Mouse Controller.
from __future__ import annotations

# Camera and Capture
CAMERA_ID: int = 0
CAPTURE_WIDTH: int = 640
CAPTURE_HEIGHT: int = 480
CAPTURE_FPS: int = 30

# MediaPipe Hand Landmarker
MP_MODEL_COMPLEXITY: int = 1
MP_MAX_NUM_HANDS: int = 1
MP_MIN_DETECTION_CONF: float = 0.75
MP_MIN_TRACKING_CONF: float = 0.75

# Active ROI Margin Box (central 60% of frame maps to 100% of screen for comfortable reach)
ROI_MARGIN_X: float = 0.20
ROI_MARGIN_Y: float = 0.20

# Cursor smoothing pipeline:  landmark EMA -> 1-Euro filter -> deadzone -> re-engage glide
# 1-Euro cutoffs are in Hz (standard definition). Lower MIN_CUTOFF = steadier at rest,
# higher BETA = less lag when you move fast.
ONE_EURO_MIN_CUTOFF: float = 0.6
ONE_EURO_BETA: float = 0.012
ONE_EURO_D_CUTOFF: float = 1.5
ONE_EURO_SPEED_FLOOR: float = 80.0   # screen px/s below which motion counts as noise, not movement

# Landmark pre-smoothing time constant in seconds (0.028 ~ the old alpha 0.70 at 30 FPS)
LANDMARK_SMOOTH_TAU: float = 0.028

# Deadzone (screen px): micro-movements smaller than this are ignored
DEADZONE_RADIUS_PX: float = 3.0

# After scroll / click / pause / hand-lost the cursor glides back to the fingertip
# over this many seconds instead of teleporting.
CURSOR_REENGAGE_SECONDS: float = 0.30
CURSOR_REENGAGE_PER_PX: float = 0.0003   # extra seconds per px of distance (capped at +0.3s) so long glides are not too fast
HAND_LOST_RESET_FRAMES: int = 3   # hand missing this many frames -> glide on return

# Gesture State Machine -- Frame Persistence
GESTURE_PERSIST_FRAMES: int = 3
FIST_PERSIST_FRAMES: int = 40  # ~1.3s of strict fist to trigger SLEEP. A few bad frames only slow it down (leaky counter), they no longer reset it.
WAKE_PERSIST_FRAMES: int = 3   # open palm / pointing / scroll pose must be held this many frames to wake from SLEEP
SCROLL_ENTER_FRAMES: int = 2   # joined-finger pose must be seen this many frames before SCROLLING starts (cursor is held meanwhile)

# Pinch Click Detection -- Fast, Accurate Single-Tap Click
PINCH_THRESHOLD_RATIO: float = 0.12   # Thumb touches index -> instant single-tap click
PINCH_PRE_PINCH_RATIO: float = 0.14   # Tight pre-pinch contact zone (never freezes normal pointing)
PINCH_RELEASE_RATIO: float = 0.20     # Wide release gap: prevents accidental repeat clicks
PINCH_COOLDOWN_SECONDS: float = 0.20  # 200ms debounce ensures exactly one click per tap

# Finger Extension Geometry
FINGER_ANGLE_EXTENDED_DEG: float = 110.0  # PIP 3D angle > 110 deg = straight/extended
FINGER_WRIST_DIST_RATIO: float = 1.02     # dist(Wrist, Tip) > dist(Wrist, PIP) * 1.02
STRICT_FIST_DIST_RATIO: float = 1.15      # dist(Tip, Wrist) < dist(MCP, Wrist) * 1.15 for true fist

# Scroll Gesture (Joined Index + Middle Fingers -- touchpad-style stroke scrolling)
SCROLL_ENTRY_JOINED_RATIO: float = 0.60   # Entry: Index & Middle tips touching side-by-side
SCROLL_EXIT_SPREAD_RATIO: float = 0.75    # Exit: Index & Middle spread apart in V-shape
SCROLL_EXIT_FRAMES: int = 4               # non-scroll pose must persist this many frames to leave SCROLLING (was 2 = too twitchy)
SCROLL_SMOOTH_TAU: float = 0.06           # seconds; smoothing of the scroll point (time based, FPS independent)
SCROLL_SPEED_MIN: float = 0.04            # hand speed (frame-heights/s) below this = hand is still -> zero scroll
SCROLL_SPEED_FULL: float = 0.12           # at/above this speed the full sensitivity applies; in between it ramps up smoothly
SCROLL_MOTION_SPEED: float = 0.10         # speed that counts as "the hand is moving" for the return-stroke guard
SCROLL_STROKE_SENSITIVITY: float = 65.0   # wheel ticks per 1.0 of hand travel (same meaning as before)
SCROLL_HORIZONTAL_SENSITIVITY: float = 25.0
SCROLL_AXIS_LOCK_RATIO: float = 1.20      # vertical wins when |vy| > |vx| * this
SCROLL_HORIZONTAL_DOMINANCE: float = 2.0  # horizontal scroll only when |vx| > |vy| * this (diagonal wobble = vertical only)
SCROLL_REVERSAL_PAUSE: float = 0.18       # seconds the hand must rest before the opposite direction is accepted (return-stroke guard)
SCROLL_MAX_TICKS_PER_FRAME: int = 2       # cap per frame to prevent browser over-scroll
SCROLL_MAX_BACKLOG: float = 1.0           # unsent ticks beyond this are dropped, so scrolling stops dead with the hand
SCROLL_IDLE_CLEAR_SECONDS: float = 0.12   # leftover fractional tick is cleared after this much stillness
SCROLL_INVERT_Y: bool = False             # False: Finger down -> Scroll down (natural document scroll / next reel)

# HUD Overlay
HUD_WINDOW_NAME: str = 'Hand Gesture Controller -- Debug HUD'
HUD_SHOW_LANDMARKS: bool = True
HUD_SHOW_ROI_BOX: bool = True
HUD_FONT_SCALE: float = 0.6
HUD_FONT_THICKNESS: int = 2

# Color palette (BGR)
COLOR_WHITE   = (255, 255, 255)
COLOR_GREEN   = (0, 230, 100)
COLOR_RED     = (0, 60, 255)
COLOR_YELLOW  = (0, 220, 255)
COLOR_CYAN    = (255, 230, 0)
COLOR_ORANGE  = (0, 140, 255)
COLOR_GRAY    = (150, 150, 150)
COLOR_DARK    = (30, 30, 30)
COLOR_BLUE    = (230, 100, 0)

# Quit Keys (q, Q, ESC=27)
QUIT_KEYS: tuple = (ord('q'), ord('Q'), 27)
