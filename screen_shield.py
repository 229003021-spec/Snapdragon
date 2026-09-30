"""HP Smart Local Privacy Guard - Isolated OS Screen Shield Process.

Powered by Snapdragon Hexagon NPU.

Runs a lightweight, process-isolated Tkinter topmost overlay.
Monitors IPC signal files in system temp directory to show/hide blurred screen shield.
Includes emergency <Escape> hotkey and button to prevent user lockout.
"""

import ctypes
import os
import sys
import tempfile
import time
import tkinter as tk
import cv2
import mss
import numpy as np
import psutil
from PIL import Image, ImageTk

# IPC Directory & Signal File Constants
TEMP_DIR = os.path.join(tempfile.gettempdir(), "hp_privacy_guard")
os.makedirs(TEMP_DIR, exist_ok=True)

SIGNAL_FILE = os.path.join(TEMP_DIR, ".shield_active")
STOP_FILE = os.path.join(TEMP_DIR, ".shield_stop")
OVERRIDE_FILE = os.path.join(TEMP_DIR, ".shield_override")
ACK_FILE = os.path.join(TEMP_DIR, ".shield_ack")
LOCK_FILE = os.path.join(TEMP_DIR, ".shield_lock")

# Windows High DPI Awareness
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def acquire_instance_lock() -> bool:
    """Enforces single running process instance lock."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                return False
        except Exception:
            pass
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception:
        return False


class ScreenShieldApp:
    """Isolated topmost fullscreen Tkinter window for desktop privacy shielding."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HP Smart Local Privacy Shield")
        self.root.attributes("-topmost", True)
        self.root.attributes("-fullscreen", True)
        self.root.config(cursor="arrow")

        # Start hidden
        self.root.withdraw()
        self.is_visible = False
        self.sct = mss.mss()
        self.bg_photo = None

        # Main Canvas Setup
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="#0e1117")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Emergency Escape Key Binding
        self.root.bind("<Escape>", self.dismiss_shield)

        # Start Monitoring Loop
        self.check_loop()

    def capture_fast_blurred_desktop(self) -> ImageTk.PhotoImage:
        """Captures desktop monitors and applies fast downscaled Gaussian blur."""
        try:
            # Multi-monitor bounds selection (all monitors combined)
            monitor = self.sct.monitors[0]
            sct_img = self.sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            np_img = np.array(img)
            h, w, _ = np_img.shape

            # Fast Blur Pipeline: Downscale to 1/4 -> Gaussian Blur -> Upscale to full
            small_w, small_h = max(1, w // 4), max(1, h // 4)
            downscaled = cv2.resize(np_img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            blurred_small = cv2.GaussianBlur(downscaled, (21, 21), 10)
            blurred_full = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

            # Darken overlay slightly for security aesthetic
            darkened = cv2.addWeighted(blurred_full, 0.5, np.zeros((h, w, 3), dtype=np.uint8), 0.5, 0)
            return ImageTk.PhotoImage(Image.fromarray(darkened))
        except Exception:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            blank = Image.new("RGB", (sw, sh), color=(15, 23, 42))
            return ImageTk.PhotoImage(blank)

    def draw_warning_overlay(self) -> None:
        """Draws frosted security warning card and restricted dismiss button."""
        self.canvas.delete("all")

        self.bg_photo = self.capture_fast_blurred_desktop()
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor=tk.NW)

        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        cx, cy = width // 2, height // 2

        # Central Security Banner Card
        card_w, card_h = 660, 360
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2

        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0f172a", outline="#ef4444", width=3)

        self.canvas.create_text(
            cx, y1 + 50, text="HP SMART LOCAL PRIVACY GUARD", fill="#00d2ff", font=("Segoe UI", 18, "bold")
        )
        self.canvas.create_text(
            cx, y1 + 85, text="Powered by Snapdragon Hexagon NPU", fill="#94a3b8", font=("Segoe UI", 11, "italic")
        )

        self.canvas.create_text(
            cx, y1 + 145, text="[PRIVACY BREACH DETECTED]", fill="#ef4444", font=("Segoe UI", 22, "bold")
        )
        self.canvas.create_text(
            cx, y1 + 185, text="Unauthorized peeker detected in camera view.", fill="#f87171", font=("Segoe UI", 13)
        )
        self.canvas.create_text(
            cx, y1 + 210, text="Screen shielded instantly to protect sensitive data.", fill="#cbd5e1", font=("Segoe UI", 11)
        )

        # Restricted Dismiss Button (Tag bound specifically to button)
        btn_y = y2 - 50
        btn_rect = self.canvas.create_rectangle(
            cx - 150, btn_y - 20, cx + 150, btn_y + 20, fill="#1e293b", outline="#3b82f6", width=2, tags="dismiss_btn"
        )
        btn_text = self.canvas.create_text(
            cx, btn_y, text="Click Here or Press ESC to Dismiss", fill="#60a5fa", font=("Segoe UI", 11, "bold"), tags="dismiss_btn"
        )

        # Bind click STRICTLY to button tags
        self.canvas.tag_bind("dismiss_btn", "<Button-1>", lambda e: self.dismiss_shield())

        # Write ACK file timestamp for latency tracking
        try:
            with open(ACK_FILE, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

    def dismiss_shield(self, event: Any = None) -> None:
        """Emergency override callback triggered by ESC key or click on dismiss button."""
        try:
            with open(OVERRIDE_FILE, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        self.root.withdraw()
        self.is_visible = False

    def is_signal_valid(self) -> bool:
        """Checks if .shield_active signal exists and is fresh (< 3.0s old)."""
        if not os.path.exists(SIGNAL_FILE):
            return False
        try:
            mtime = os.path.getmtime(SIGNAL_FILE)
            if (time.time() - mtime) > 3.0:
                return False
            return True
        except Exception:
            return False

    def check_loop(self) -> None:
        """Periodically checks signal files to update shield visibility."""
        if os.path.exists(STOP_FILE):
            try:
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
            except Exception:
                pass
            self.root.destroy()
            sys.exit(0)
            return

        should_be_visible = self.is_signal_valid() and not os.path.exists(OVERRIDE_FILE)

        if should_be_visible and not self.is_visible:
            self.draw_warning_overlay()
            self.root.deiconify()
            self.root.focus_force()
            self.is_visible = True
        elif not should_be_visible and self.is_visible:
            self.root.withdraw()
            self.is_visible = False

        self.root.after(100, self.check_loop)


def main() -> None:
    if not acquire_instance_lock():
        print("Another screen_shield process is already running. Exiting.")
        sys.exit(0)

    # Clean stale signal files on boot
    for f in [SIGNAL_FILE, STOP_FILE, OVERRIDE_FILE, ACK_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    root = tk.Tk()
    app = ScreenShieldApp(root)
    try:
        root.mainloop()
    finally:
        if os.path.exists(LOCK_FILE):
            try:
                os.remove(LOCK_FILE)
            except Exception:
                pass


if __name__ == "__main__":
    main()
