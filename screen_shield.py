"""
HP Smart Local Privacy Guard - Isolated OS Screen Shield Process
Powered by Snapdragon Hexagon NPU

Runs a lightweight, process-isolated Tkinter topmost overlay.
Monitors an IPC signal file (.shield_active) to show/hide a blurred screen shield.
Includes an emergency <Escape> hotkey and button to prevent user lockout.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import font as tkfont
import mss
import cv2
import numpy as np
from PIL import Image, ImageTk

SIGNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".shield_active")
STOP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".shield_stop")
OVERRIDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".shield_override")

class ScreenShieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HP Smart Local Privacy Shield")
        self.root.attributes("-topmost", True)
        self.root.attributes("-fullscreen", True)
        self.root.config(cursor="arrow")
        
        # Hide initial window
        self.root.withdraw()
        self.is_visible = False
        self.sct = mss.mss()
        
        # UI components setup
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg="#0e1117")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Emergency escape key binding
        self.root.bind("<Escape>", self.dismiss_shield)
        
        # Start monitoring loop
        self.check_loop()

    def capture_blurred_desktop(self):
        """Captures the primary display and applies heavy Gaussian blur."""
        try:
            monitor = self.sct.monitors[1]  # Primary monitor
            sct_img = self.sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Convert to numpy array for OpenCV blur
            np_img = np.array(img)
            blurred = cv2.GaussianBlur(np_img, (99, 99), 30)
            
            # Darken overlay slightly for security aesthetic
            darkened = cv2.addWeighted(blurred, 0.5, np.zeros(blurred.shape, blurred.dtype), 0.5, 0)
            
            # Convert back to ImageTk format
            blurred_img = Image.fromarray(darkened)
            return ImageTk.PhotoImage(blurred_img)
        except Exception as e:
            # Fallback if screen capture fails
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            blank_img = Image.new("RGB", (screen_width, screen_height), color=(15, 23, 42))
            return ImageTk.PhotoImage(blank_img)

    def draw_warning_overlay(self):
        """Draws the central security badge and emergency escape instructions."""
        self.canvas.delete("all")
        
        # Set blurred screenshot as background
        self.bg_photo = self.capture_blurred_desktop()
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor=tk.NW)
        
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        cx, cy = width // 2, height // 2
        
        # Draw central dark frosted card
        card_w, card_h = 650, 350
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2
        
        # Draw rounded card outline
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#0f172a", outline="#ef4444", width=3)
        
        # Security Header
        self.canvas.create_text(cx, y1 + 50, text="🛡️ HP SMART LOCAL PRIVACY GUARD", 
                                fill="#00d2ff", font=("Segoe UI", 18, "bold"))
        self.canvas.create_text(cx, y1 + 90, text="Powered by Snapdragon Hexagon NPU", 
                                fill="#94a3b8", font=("Segoe UI", 11, "italic"))
        
        # Breach Alert Banner
        self.canvas.create_text(cx, y1 + 150, text="⚠️ PRIVACY BREACH DETECTED!", 
                                fill="#ef4444", font=("Segoe UI", 22, "bold"))
        self.canvas.create_text(cx, y1 + 190, text="Unauthorized peeker detected in camera view.", 
                                fill="#f87171", font=("Segoe UI", 13))
        self.canvas.create_text(cx, y1 + 215, text="Screen shielded instantly to protect your sensitive work.", 
                                fill="#cbd5e1", font=("Segoe UI", 11))
        
        # Dismiss button indicator
        btn_y = y2 - 50
        self.canvas.create_rectangle(cx - 140, btn_y - 20, cx + 140, btn_y + 20, 
                                     fill="#1e293b", outline="#3b82f6", width=2)
        self.canvas.create_text(cx, btn_y, text="Press ESC to Dismiss Shield", 
                                fill="#60a5fa", font=("Segoe UI", 12, "bold"))
        
        self.canvas.tag_bind(tk.ALL, "<Button-1>", lambda e: self.dismiss_shield())

    def dismiss_shield(self, event=None):
        """Emergency override callback triggered by ESC key or click."""
        try:
            with open(OVERRIDE_FILE, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        self.root.withdraw()
        self.is_visible = False

    def check_loop(self):
        """Periodically checks signal files to update shield visibility."""
        if os.path.exists(STOP_FILE):
            self.root.destroy()
            sys.exit(0)
            return

        should_be_visible = os.path.exists(SIGNAL_FILE) and not os.path.exists(OVERRIDE_FILE)
        
        if should_be_visible and not self.is_visible:
            self.draw_warning_overlay()
            self.root.deiconify()
            self.root.focus_force()
            self.is_visible = True
        elif not should_be_visible and self.is_visible:
            self.root.withdraw()
            self.is_visible = False

        # Schedule next check in 150ms
        self.root.after(150, self.check_loop)

def main():
    # Remove old signal files on launch
    for f in [SIGNAL_FILE, STOP_FILE, OVERRIDE_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    root = tk.Tk()
    app = ScreenShieldApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
