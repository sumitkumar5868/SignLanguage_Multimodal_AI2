"""Professional Sign Language Recognition UI.

Uses tkinter for a dark-themed, accessible desktop UI.
Camera feed is polled via root.after() — the UI thread never blocks.

Architecture:
    SignLanguageApp (class)
        ├── _build_ui()          — creates all widgets with proper pixel layouts
        ├── _start_camera()      — opens webcam + MediaPipe
        ├── _stop_camera()       — releases camera resources safely
        ├── _poll_camera()       — recurring frame read + inference (non-blocking after loop)
        ├── _update_result()     — updates result card widgets
        ├── _update_status()     — updates status bar widgets
        └── on_speak()           — triggers Hindi TTS

Run this module directly to launch the UI:
    python -m modules.ui
"""

import logging
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

import cv2
from PIL import Image, ImageTk

from modules.audio import is_audio_available, speak_hindi
from modules.hand_detection import initialize_hands, process_hand_frame
from modules.predictor import is_model_loaded, predict
from modules.sign_map import get_sign_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens — Modern Dark Theme & High Accessibility Contrast
# ---------------------------------------------------------------------------
BG_DARK       = "#0a0a14"   # Window background
BG_CARD       = "#12122a"   # Card / panel background
BG_CARD2      = "#1a1a38"   # Slightly lighter card (headers/strips)
ACCENT_PURPLE = "#7c5cbf"   # Primary accent
ACCENT_CYAN   = "#00d4ff"   # Detected sign highlight
ACCENT_GREEN  = "#22c55e"   # Status: Ready / Good
ACCENT_AMBER  = "#f59e0b"   # Status: Warning / Waiting
ACCENT_RED    = "#ef4444"   # Status: Error / Not connected
TEXT_PRIMARY  = "#f0f0ff"   # High contrast main text
TEXT_SECONDARY = "#9da3c0"  # Muted readable labels
TEXT_HINDI    = "#fbbf24"   # High contrast amber for Hindi
BORDER_COLOR  = "#2a2a50"   # Card border contrast

# Camera poll interval (milliseconds)
POLL_INTERVAL_MS = 25    # ~40 fps smooth preview

# Smoothing: only update the displayed result when confidence exceeds this
DISPLAY_CONFIDENCE_THRESHOLD = 0.45

# Minimum consecutive frames a sign must be seen before displaying
STABLE_FRAME_COUNT = 4


class SignLanguageApp:
    """Main application window for sign language recognition."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._camera: Optional[cv2.VideoCapture] = None
        self._hands = None
        self._camera_running = False

        # Recognition state
        self._current_label: Optional[str] = None
        self._current_confidence: float = 0.0
        self._pending_label: Optional[str] = None
        self._pending_frames: int = 0

        # Photo reference (must be kept to prevent Python GC from dropping images)
        self._photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._update_status("Camera: Initializing…", "—", "Starting")

        # Auto-start camera smoothly after window is mapped
        self.root.after(150, self._auto_start_camera)

    # ───────────────────────────── UI Construction ──────────────────────────

    def _build_ui(self) -> None:
        """Construct all widgets with proper pixel layouts."""
        root = self.root
        root.title("Sign Language Multimodal AI")
        root.configure(bg=BG_DARK)
        root.resizable(False, False)

        # ── Fonts ────────────────────────────────────────────────────────────
        try:
            f_title   = tkfont.Font(family="Helvetica Neue", size=18, weight="bold")
            f_sub     = tkfont.Font(family="Helvetica Neue", size=10)
            f_label   = tkfont.Font(family="Helvetica Neue", size=9, weight="bold")
            f_sign    = tkfont.Font(family="Helvetica Neue", size=26, weight="bold")
            f_eng     = tkfont.Font(family="Helvetica Neue", size=16, weight="bold")
            f_hindi   = tkfont.Font(family="Helvetica Neue", size=20, weight="bold")
            f_conf    = tkfont.Font(family="Helvetica Neue", size=24, weight="bold")
            f_btn     = tkfont.Font(family="Helvetica Neue", size=11, weight="bold")
            f_status  = tkfont.Font(family="Helvetica Neue", size=9)
        except Exception:
            f_title = f_sub = f_label = f_sign = f_eng = f_hindi = f_conf = f_btn = f_status = None

        # ── Header ───────────────────────────────────────────────────────────
        header = tk.Frame(root, bg="#0e0e24", pady=10)
        header.pack(fill="x")

        tk.Label(
            header,
            text="🤟  Sign Language Multimodal AI",
            bg="#0e0e24", fg=TEXT_PRIMARY,
            font=f_title,
        ).pack()
        tk.Label(
            header,
            text="Real-Time Sign Recognition  •  Hindi Voice Output",
            bg="#0e0e24", fg=TEXT_SECONDARY,
            font=f_sub,
        ).pack(pady=(2, 0))

        # Accent border
        tk.Frame(root, bg=ACCENT_PURPLE, height=2).pack(fill="x")

        # ── Body (Camera on Left, Recognition Card on Right) ──────────────────
        body = tk.Frame(root, bg=BG_DARK, padx=14, pady=12)
        body.pack(fill="both", expand=True)

        # ── LEFT: Camera panel (640x480 container) ───────────────────────────
        cam_card = tk.Frame(
            body, bg=BG_CARD, bd=0, highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        cam_card.pack(side="left", fill="both", padx=(0, 10))

        cam_header = tk.Frame(cam_card, bg=BG_CARD2, pady=6)
        cam_header.pack(fill="x")
        tk.Label(
            cam_header, text="📷  Live Camera Preview",
            bg=BG_CARD2, fg=TEXT_PRIMARY, font=f_label
        ).pack()

        # Fixed 640x480 container frame so text labels cannot blow up geometry
        self._video_box = tk.Frame(cam_card, width=640, height=480, bg="#000000")
        self._video_box.pack_propagate(False)
        self._video_box.pack(padx=0, pady=0)

        self._cam_label = tk.Label(
            self._video_box, bg="#000000",
            text="Initializing Camera Preview…\n\nPlease wait.",
            fg=TEXT_SECONDARY, font=f_sub
        )
        self._cam_label.pack(fill="both", expand=True)

        # Camera status strip directly below video preview
        cam_status_frame = tk.Frame(cam_card, bg=BG_CARD2, pady=6)
        cam_status_frame.pack(fill="x")

        self._lbl_cam_status = tk.Label(
            cam_status_frame, text="● Camera: Connecting…",
            bg=BG_CARD2, fg=ACCENT_AMBER, font=f_status
        )
        self._lbl_cam_status.pack(side="left", padx=12)

        self._lbl_hand_status = tk.Label(
            cam_status_frame, text="✋ Hands detected: 0",
            bg=BG_CARD2, fg=TEXT_SECONDARY, font=f_status
        )
        self._lbl_hand_status.pack(side="right", padx=12)

        # ── RIGHT: Recognition Result Card (width ~360) ──────────────────────
        result_card = tk.Frame(
            body, bg=BG_CARD, bd=0, highlightthickness=1,
            highlightbackground=BORDER_COLOR, width=360
        )
        result_card.pack(side="right", fill="both", expand=True)
        result_card.pack_propagate(False)

        # Result Header
        rh = tk.Frame(result_card, bg=BG_CARD2, pady=6)
        rh.pack(fill="x")
        tk.Label(
            rh, text="🧠  Recognition Result",
            bg=BG_CARD2, fg=TEXT_PRIMARY, font=f_label
        ).pack()

        # Content container
        content_frame = tk.Frame(result_card, bg=BG_CARD, padx=16, pady=10)
        content_frame.pack(fill="both", expand=True)

        # 1. Detected Sign
        tk.Label(
            content_frame, text="DETECTED SIGN",
            bg=BG_CARD, fg=TEXT_SECONDARY, font=f_label
        ).pack(anchor="w")

        self._lbl_sign = tk.Label(
            content_frame, text="[WAITING]",
            bg=BG_CARD, fg=ACCENT_CYAN, font=f_sign
        )
        self._lbl_sign.pack(anchor="w", pady=(2, 6))

        tk.Frame(content_frame, bg=BORDER_COLOR, height=1).pack(fill="x", pady=4)

        # 2. English
        tk.Label(
            content_frame, text="🇬🇧  English Meaning",
            bg=BG_CARD, fg=TEXT_SECONDARY, font=f_label
        ).pack(anchor="w")

        self._lbl_english = tk.Label(
            content_frame, text="Show your hand to the camera",
            bg=BG_CARD, fg=TEXT_PRIMARY, font=f_eng,
            wraplength=320, justify="left"
        )
        self._lbl_english.pack(anchor="w", pady=(2, 6))

        tk.Frame(content_frame, bg=BORDER_COLOR, height=1).pack(fill="x", pady=4)

        # 3. Hindi
        tk.Label(
            content_frame, text="🇮🇳  Hindi Voice Text",
            bg=BG_CARD, fg=TEXT_SECONDARY, font=f_label
        ).pack(anchor="w")

        self._lbl_hindi = tk.Label(
            content_frame, text="—",
            bg=BG_CARD, fg=TEXT_HINDI, font=f_hindi,
            wraplength=320, justify="left"
        )
        self._lbl_hindi.pack(anchor="w", pady=(2, 6))

        tk.Frame(content_frame, bg=BORDER_COLOR, height=1).pack(fill="x", pady=4)

        # 4. Confidence
        tk.Label(
            content_frame, text="📊  Confidence Score",
            bg=BG_CARD, fg=TEXT_SECONDARY, font=f_label
        ).pack(anchor="w")

        self._lbl_confidence = tk.Label(
            content_frame, text="0%",
            bg=BG_CARD, fg=ACCENT_GREEN, font=f_conf
        )
        self._lbl_confidence.pack(anchor="w", pady=(2, 2))

        # Confidence bar container
        self._conf_bar_bg = tk.Frame(content_frame, bg=BG_CARD2, height=6)
        self._conf_bar_bg.pack(fill="x", pady=(2, 8))
        self._conf_bar_fill = tk.Frame(self._conf_bar_bg, bg=ACCENT_GREEN, height=6, width=0)
        self._conf_bar_fill.place(x=0, y=0, relheight=1.0)

        # 5. Live Recognition Guidance
        self._lbl_rec_status = tk.Label(
            content_frame, text="Recognition: Ready",
            bg=BG_CARD, fg=ACCENT_GREEN, font=f_status,
            wraplength=320, justify="left"
        )
        self._lbl_rec_status.pack(anchor="w", pady=(4, 0))

        # ── Buttons at bottom of Result card ─────────────────────────────────
        btn_frame = tk.Frame(result_card, bg=BG_CARD, pady=12, padx=16)
        btn_frame.pack(fill="x", side="bottom")

        self._btn_speak = tk.Button(
            btn_frame, text="🔊  Speak",
            bg=ACCENT_PURPLE, fg="white", font=f_btn,
            relief="flat", padx=14, pady=7,
            activebackground="#9775da", activeforeground="white",
            cursor="hand2",
            command=self.on_speak,
        )
        self._btn_speak.pack(side="left", padx=(0, 6))

        self._btn_clear = tk.Button(
            btn_frame, text="🗑  Clear",
            bg=BG_CARD2, fg=TEXT_SECONDARY, font=f_btn,
            relief="flat", padx=12, pady=7,
            activebackground=BG_CARD, activeforeground=TEXT_PRIMARY,
            cursor="hand2",
            command=self.on_clear,
        )
        self._btn_clear.pack(side="left", padx=(0, 6))

        self._btn_camera = tk.Button(
            btn_frame, text="⏹  Stop Camera",
            bg=ACCENT_RED, fg="white", font=f_btn,
            relief="flat", padx=12, pady=7,
            activebackground="#dc2626", activeforeground="white",
            cursor="hand2",
            command=self.on_toggle_camera,
        )
        self._btn_camera.pack(side="right")

        # ── Bottom Status bar ────────────────────────────────────────────────
        tk.Frame(root, bg=BORDER_COLOR, height=1).pack(fill="x")

        status_bar = tk.Frame(root, bg="#080812", pady=4, padx=10)
        status_bar.pack(fill="x")

        self._lbl_status_left = tk.Label(
            status_bar, text="Ready",
            bg="#080812", fg=TEXT_SECONDARY, font=f_status,
        )
        self._lbl_status_left.pack(side="left")

        self._lbl_status_right = tk.Label(
            status_bar,
            text="Multimodal Sign Recognition Pipeline Active",
            bg="#080812", fg=TEXT_SECONDARY, font=f_status,
        )
        self._lbl_status_right.pack(side="right")

        # Handle window close cleanly
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ───────────────────────────── Camera Pipeline ──────────────────────────

    def _auto_start_camera(self) -> None:
        """Start camera feed automatically when application opens."""
        if not self._camera_running:
            self._start_camera_stream()

    def _start_camera_stream(self) -> bool:
        """Initialize camera device and MediaPipe Hand Landmarker."""
        self._lbl_cam_status.config(text="● Camera: Connecting…", fg=ACCENT_AMBER)
        self.root.update_idletasks()

        # Try default camera index 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Fallback to index 1 if available
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            logger.error("Webcam could not be opened.")
            self._camera_running = False
            self._lbl_cam_status.config(text="● Camera: Unavailable", fg=ACCENT_RED)
            self._cam_label.config(
                image="",
                text="Camera unavailable — please check camera permission.\n\n"
                     "Ensure webcam permissions are granted in System Settings.",
                fg=ACCENT_RED
            )
            self._btn_camera.config(text="▶  Start Camera", bg=ACCENT_GREEN)
            self._update_status("Camera: Unavailable", "0", "Check Permissions")
            return False

        self._camera = cap

        try:
            self._hands = initialize_hands()
        except RuntimeError as exc:
            logger.error("MediaPipe initialization error: %s", exc)
            self._cam_label.config(
                image="",
                text=f"MediaPipe Initialization Error:\n\n{exc}",
                fg=ACCENT_RED
            )
            self._camera.release()
            self._camera = None
            return False

        self._camera_running = True
        self._btn_camera.config(text="⏹  Stop Camera", bg=ACCENT_RED)
        self._lbl_cam_status.config(text="● Camera: Connected", fg=ACCENT_GREEN)
        self._update_status("Camera: Connected", "0", "Recognition: Ready")

        # Begin recurring camera polling loop
        self.root.after(POLL_INTERVAL_MS, self._poll_camera)
        return True

    def _stop_camera_stream(self) -> None:
        """Release camera and MediaPipe resources safely."""
        self._camera_running = False
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None

        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None

        self._photo = None
        self._cam_label.config(
            image="",
            text="Camera stopped.\n\nClick  ▶  Start Camera  to resume recognition.",
            fg=TEXT_SECONDARY, bg="#000000"
        )
        self._btn_camera.config(text="▶  Start Camera", bg=ACCENT_GREEN)
        self._lbl_cam_status.config(text="● Camera: Stopped", fg=ACCENT_AMBER)
        self._lbl_hand_status.config(text="✋ Hands detected: —", fg=TEXT_SECONDARY)
        self._update_status("Camera: Stopped", "—", "Idle")

    def _poll_camera(self) -> None:
        """Read webcam frame, detect landmarks, run inference, and update UI."""
        if not self._camera_running or self._camera is None:
            return

        ret, frame = self._camera.read()
        if not ret or frame is None:
            self._lbl_cam_status.config(text="● Camera: Frame error", fg=ACCENT_RED)
            self.root.after(POLL_INTERVAL_MS, self._poll_camera)
            return

        # Mirror frame horizontally for natural user interaction
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks/skeletons on frame
        annotated_frame, detected_hands = process_hand_frame(frame, self._hands)

        # Run ML model inference
        label, confidence = predict(detected_hands)

        # Multi-frame smoothing for stable recognition output
        if label is not None and confidence >= DISPLAY_CONFIDENCE_THRESHOLD:
            if label == self._pending_label:
                self._pending_frames += 1
            else:
                self._pending_label = label
                self._pending_frames = 1

            if self._pending_frames >= STABLE_FRAME_COUNT:
                if label != self._current_label:
                    self._current_label = label
                    self._current_confidence = confidence
                    self._update_result(label, confidence)
                else:
                    self._current_confidence = confidence
                    self._draw_confidence_bar(confidence)
        else:
            self._pending_label = None
            self._pending_frames = 0

        # Update Hand Count badge
        hand_count = len(detected_hands)
        if hand_count == 0:
            self._lbl_hand_status.config(text="✋ Hands detected: 0", fg=ACCENT_AMBER)
            self._lbl_rec_status.config(text="Show your hand to the camera", fg=TEXT_SECONDARY)
        elif hand_count == 1:
            self._lbl_hand_status.config(text="✋ Hands detected: 1", fg=ACCENT_GREEN)
            if label is None:
                self._lbl_rec_status.config(text="Hand detected — analysing gesture…", fg=ACCENT_CYAN)
            else:
                self._lbl_rec_status.config(text="Recognition: Active", fg=ACCENT_GREEN)
        else:
            self._lbl_hand_status.config(text=f"✋ Hands detected: {hand_count}", fg=ACCENT_CYAN)
            self._lbl_rec_status.config(text="Multiple hands detected", fg=ACCENT_CYAN)

        # Render frame onto UI (Resize to exact 640x480 box)
        try:
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image = pil_image.resize((640, 480), Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(image=pil_image)
            self._cam_label.config(image=self._photo, text="")
        except Exception as exc:
            logger.debug("Frame render error: %s", exc)

        # Reschedule next frame
        if self._camera_running:
            self.root.after(POLL_INTERVAL_MS, self._poll_camera)

    # ───────────────────────────── Result Display ───────────────────────────

    def _update_result(self, label: str, confidence: float) -> None:
        """Update the recognition result card."""
        english, hindi = get_sign_info(label)

        self._lbl_sign.config(text=f"[{label}]", fg=ACCENT_CYAN)
        self._lbl_english.config(text=english, fg=TEXT_PRIMARY)
        self._lbl_hindi.config(text=hindi, fg=TEXT_HINDI)

        pct = int(confidence * 100)
        conf_color = ACCENT_GREEN if pct >= 70 else (ACCENT_AMBER if pct >= 50 else ACCENT_RED)
        self._lbl_confidence.config(text=f"{pct}%", fg=conf_color)

        self._draw_confidence_bar(confidence, conf_color)
        self._update_status("Camera: Connected", str(len(self._hands) if self._hands else 1), f"Detected: {label}")

    def _draw_confidence_bar(self, confidence: float, color: str = ACCENT_GREEN) -> None:
        """Update the visual confidence fill meter."""
        try:
            total_width = self._conf_bar_bg.winfo_width()
            if total_width <= 1:
                total_width = 300
            fill_width = max(4, int(total_width * min(1.0, confidence)))
            self._conf_bar_fill.config(bg=color)
            self._conf_bar_fill.place(x=0, y=0, relheight=1.0, width=fill_width)
        except Exception:
            pass

    def _update_status(self, cam_text: str, hand_text: str, rec_text: str) -> None:
        """Update bottom status bar text."""
        try:
            self._lbl_status_left.config(text=f"{cam_text}  |  Hands: {hand_text}  |  {rec_text}")
        except Exception:
            pass

    # ───────────────────────────── Event Handlers ───────────────────────────

    def on_speak(self) -> None:
        """Speak the Hindi translation via background TTS."""
        if not self._current_label:
            self._lbl_rec_status.config(text="No sign detected yet — please show a sign first.", fg=ACCENT_AMBER)
            return

        _, hindi = get_sign_info(self._current_label)
        if hindi and hindi not in ("—", "वर्तमान मॉडल में उपलब्ध नहीं"):
            speak_hindi(hindi)
            self._lbl_rec_status.config(text=f"🔊 Speaking: {hindi}", fg=ACCENT_CYAN)
        else:
            self._lbl_rec_status.config(text="Hindi text not available for this sign.", fg=ACCENT_AMBER)

    def on_clear(self) -> None:
        """Reset recognition card values."""
        self._current_label = None
        self._current_confidence = 0.0
        self._pending_label = None
        self._pending_frames = 0

        self._lbl_sign.config(text="[WAITING]", fg=ACCENT_CYAN)
        self._lbl_english.config(text="Show your hand to the camera", fg=TEXT_PRIMARY)
        self._lbl_hindi.config(text="—", fg=TEXT_HINDI)
        self._lbl_confidence.config(text="0%", fg=ACCENT_GREEN)
        self._conf_bar_fill.place(x=0, y=0, relheight=1.0, width=0)
        self._lbl_rec_status.config(text="Recognition: Ready", fg=ACCENT_GREEN)

    def on_toggle_camera(self) -> None:
        """Toggle camera stream on and off."""
        if self._camera_running:
            self._stop_camera_stream()
        else:
            self._start_camera_stream()

    def on_close(self) -> None:
        """Clean up resources on window close."""
        self._stop_camera_stream()
        self.root.destroy()
