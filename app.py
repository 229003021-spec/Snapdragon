"""HP Smart Local Privacy Guard — Master Streamlit Dashboard
Snapdragon AI Lab Challenge (Step 3)

Executive-ready, interactive dashboard optimized for live pitch and video demos.
Integrates with PrivacyGuardEngine, background protection worker thread,
measured hardware telemetry, ONNX QNN Execution Provider status, OS screen shield,
and persistent incident logging.

Usage:
    streamlit run app.py
"""

import atexit
import csv
import datetime
import io
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# Import Privacy Engine contract
from privacy_engine import PrivacyGuardEngine, PrivacyStatus, EngineResult, FaceBox

# Configure Module Logger
logger = logging.getLogger("PrivacyGuardApp")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# IPC Signal File Constants (Stored in system temp dir to prevent repo clutter)
TEMP_DIR = os.path.join(os.getenv("TEMP", "/tmp"), "hp_privacy_guard")
os.makedirs(TEMP_DIR, exist_ok=True)

SIGNAL_FILE = os.path.join(TEMP_DIR, ".shield_active")
STOP_FILE = os.path.join(TEMP_DIR, ".shield_stop")
OVERRIDE_FILE = os.path.join(TEMP_DIR, ".shield_override")
ACK_FILE = os.path.join(TEMP_DIR, ".shield_ack")
BREACH_LOG_FILE = os.path.join(TEMP_DIR, "breaches.csv")

# ------------------------------------------------------------------------------
# 1. Streamlit Page Config & Executive Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HP Smart Local Privacy Guard — Snapdragon AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Executive Dark Theme CSS Injection
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0e1117;
        color: #f8fafc;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Card */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }
    .badge-teal {
        background: rgba(0, 210, 200, 0.15);
        color: #00d2c8;
        border: 1px solid #00d2c8;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-amber {
        background: rgba(255, 176, 32, 0.15);
        color: #ffb020;
        border: 1px solid #ffb020;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-red {
        background: rgba(255, 45, 85, 0.15);
        color: #ff2d55;
        border: 1px solid #ff2d55;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* Main Status Cards */
    .status-card-secure {
        background: radial-gradient(circle at top left, #064e3b 0%, #0f172a 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
        color: #ecfdf5;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .status-card-peeker {
        background: radial-gradient(circle at top left, #7f1d1d 0%, #0f172a 100%);
        border: 2px solid #ff2d55;
        box-shadow: 0 0 20px rgba(255, 45, 85, 0.4);
        color: #fff1f2;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 20px;
        animation: alert-pulse 1.5s infinite alternate;
    }
    .status-card-away {
        background: radial-gradient(circle at top left, #78350f 0%, #0f172a 100%);
        border: 2px solid #ffb020;
        box-shadow: 0 0 15px rgba(255, 176, 32, 0.25);
        color: #fffbeb;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .status-card-error {
        background: radial-gradient(circle at top left, #451a03 0%, #0f172a 100%);
        border: 2px solid #f97316;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.3);
        color: #ffedd5;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 20px;
    }

    @keyframes alert-pulse {
        0% { box-shadow: 0 0 10px rgba(255, 45, 85, 0.3); }
        100% { box-shadow: 0 0 25px rgba(255, 45, 85, 0.7); }
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #00d2c8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Thread-Safe Worker State & Background Protection Worker
# ------------------------------------------------------------------------------
class WorkerState:
    """Thread-safe state container shared between ProtectionWorker and Streamlit UI."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_bgr: Optional[np.ndarray] = None
        self.rendered_frame_bgr: Optional[np.ndarray] = None
        self.result: Optional[EngineResult] = None
        self.metrics: Dict[str, Any] = {}
        self.sensor_error: bool = False
        self.error_message: str = ""
        self.is_running: bool = False
        self.history_latency: List[float] = []
        self.history_cpu: List[float] = []
        self.history_timestamps: List[float] = []
        self.time_to_shield_ms: Optional[float] = None
        self.breach_incidents: List[Dict[str, Any]] = []


class ProtectionWorker:
    """Background worker thread running capture, privacy engine, and shield control.
    
    Runs continuously independently of browser tab focus or reruns.
    """
    def __init__(self) -> None:
        self.state = WorkerState()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.engine: Optional[PrivacyGuardEngine] = None
        self.camera_index: int = 0
        self.use_demo_source: bool = False
        self.enable_shield: bool = True
        self.enable_boxes: bool = True
        self.enable_os_shield: bool = False
        self.test_mode_peeker: bool = False
        self.freeze_frame: bool = False

        # Incident log state tracking
        self._current_incident: Optional[Dict[str, Any]] = None
        self._peeker_start_timestamp: float = 0.0

    def start(self) -> None:
        """Starts the background worker thread if not already running."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_worker, daemon=True)
            self.thread.start()
            logger.info("BackgroundProtectionWorker thread started.")

    def stop(self) -> None:
        """Stops the worker thread cleanly."""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.set_os_shield_signal(False)
        logger.info("BackgroundProtectionWorker thread stopped.")

    def set_os_shield_signal(self, active: bool) -> None:
        """Creates or removes IPC signal file for isolated screen_shield.py process."""
        if active:
            if not os.path.exists(SIGNAL_FILE) and not os.path.exists(OVERRIDE_FILE):
                try:
                    with open(SIGNAL_FILE, "w") as f:
                        f.write(str(time.time()))
                except Exception as e:
                    logger.warning(f"Could not write IPC signal file: {e}")
        else:
            if os.path.exists(SIGNAL_FILE):
                try:
                    os.remove(SIGNAL_FILE)
                except Exception:
                    pass

    def clear_os_override_if_needed(self, stable_status: PrivacyStatus) -> None:
        """Automatically re-arms OS shield when peeker leaves (non-peeker status)."""
        if stable_status in (PrivacyStatus.SECURE, PrivacyStatus.NO_USER):
            if os.path.exists(OVERRIDE_FILE):
                try:
                    os.remove(OVERRIDE_FILE)
                    logger.info("OS Screen Shield override cleared automatically (Re-armed).")
                except Exception:
                    pass

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates a clean synthetic desktop test frame when no camera is present."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 25, 20)  # Dark slate background
        # Draw synthetic desktop header bar
        cv2.rectangle(frame, (0, 0), (640, 40), (45, 40, 35), -1)
        cv2.putText(frame, "HP Confidential Desktop Demo", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw synthetic primary user face
        cv2.ellipse(frame, (320, 260), (60, 80), 0, 0, 360, (180, 210, 230), -1)
        cv2.circle(frame, (300, 240), 6, (40, 40, 40), -1)
        cv2.circle(frame, (340, 240), 6, (40, 40, 40), -1)
        cv2.ellipse(frame, (320, 280), (30, 15), 0, 0, 180, (40, 40, 40), 3)
        return frame

    def _inject_test_peeker(self, frame: np.ndarray) -> np.ndarray:
        """Injects a synthetic peeker face into the frame before detection (Test Mode)."""
        out = frame.copy()
        # Draw second peeker face on the right side
        cv2.ellipse(out, (520, 220), (45, 60), 0, 0, 360, (170, 190, 210), -1)
        cv2.circle(out, (505, 205), 5, (40, 40, 40), -1)
        cv2.circle(out, (535, 205), 5, (40, 40, 40), -1)
        cv2.ellipse(out, (520, 235), (20, 10), 0, 0, 180, (40, 40, 40), 2)
        return out

    def _run_worker(self) -> None:
        """Main background processing loop."""
        self.engine = PrivacyGuardEngine()
        cap: Optional[cv2.VideoCapture] = None
        consecutive_failures = 0
        
        with self.state.lock:
            self.state.is_running = True

        try:
            while not self.stop_event.is_set():
                if self.freeze_frame and self.state.frame_bgr is not None:
                    time.sleep(0.1)
                    continue

                frame: Optional[np.ndarray] = None

                # 1. Grab Frame (Camera or Demo Source)
                if not self.use_demo_source:
                    if cap is None or not cap.isOpened():
                        if sys.platform == "win32":
                            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                        else:
                            cap = cv2.VideoCapture(self.camera_index)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        ret, grabbed = cap.read()
                        if ret and grabbed is not None:
                            frame = grabbed
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures > 5:
                                with self.state.lock:
                                    self.state.sensor_error = True
                                    self.state.error_message = f"Camera index {self.camera_index} read failure."
                                time.sleep(0.5)
                                continue
                    else:
                        with self.state.lock:
                            self.state.sensor_error = True
                            self.state.error_message = f"Could not open camera index {self.camera_index}."
                        time.sleep(1.0)
                        continue
                else:
                    # Demo Source Mode
                    frame = self._generate_synthetic_frame()
                    with self.state.lock:
                        self.state.sensor_error = False

                # 2. Inject Test Mode Peeker if requested
                if self.test_mode_peeker and frame is not None:
                    frame = self._inject_test_peeker(frame)

                # 3. Process Frame with PrivacyGuardEngine
                if frame is not None:
                    t_raw = time.time()
                    result = self.engine.process_frame(frame)
                    
                    # Handle OS Screen Shield Signal & Re-arm
                    if self.enable_os_shield:
                        if result.stable_status == PrivacyStatus.PEEKER_DETECTED:
                            self.set_os_shield_signal(True)
                            # Measure time-to-shield if ACK file written by screen_shield process
                            if os.path.exists(ACK_FILE):
                                try:
                                    with open(ACK_FILE, "r") as f:
                                        ack_ts = float(f.read().strip())
                                        if ack_ts >= self._peeker_start_timestamp:
                                            self.state.time_to_shield_ms = (ack_ts - self._peeker_start_timestamp) * 1000.0
                                except Exception:
                                    pass
                        else:
                            self.set_os_shield_signal(False)
                            self.clear_os_override_if_needed(result.stable_status)
                    else:
                        self.set_os_shield_signal(False)
                        self.clear_os_override_if_needed(result.stable_status)

                    # 4. Handle Breach Incident Logging (Incident-based, NOT per second)
                    if result.stable_status == PrivacyStatus.PEEKER_DETECTED:
                        if self._current_incident is None:
                            self._peeker_start_timestamp = time.time()
                            self._current_incident = {
                                "Start Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "start_epoch": self._peeker_start_timestamp,
                                "Max Peekers": sum(1 for f in result.faces if not f.is_primary),
                                "Shield State": "OS Shield Active" if self.enable_os_shield else "Webcam Blur Active",
                                "Dismissed By": "None (Active)",
                            }
                        else:
                            current_peekers = sum(1 for f in result.faces if not f.is_primary)
                            self._current_incident["Max Peekers"] = max(self._current_incident["Max Peekers"], current_peekers)
                    else:
                        if self._current_incident is not None:
                            end_epoch = time.time()
                            duration_s = round(end_epoch - self._current_incident["start_epoch"], 1)
                            self._current_incident["Duration (s)"] = duration_s
                            if os.path.exists(OVERRIDE_FILE):
                                self._current_incident["Dismissed By"] = "User Esc / Click"
                            else:
                                self._current_incident["Dismissed By"] = "Peeker Left"
                            
                            incident_record = {
                                "Start Time": self._current_incident["Start Time"],
                                "Duration (s)": duration_s,
                                "Max Peekers": self._current_incident["Max Peekers"],
                                "Shield State": self._current_incident["Shield State"],
                                "Dismissed By": self._current_incident["Dismissed By"],
                            }
                            
                            with self.state.lock:
                                self.state.breach_incidents.append(incident_record)
                                self._write_incident_to_csv(incident_record)
                            
                            self._current_incident = None

                    # 5. Render Output Frame (Annotations + Privacy Blur)
                    if self.enable_boxes:
                        rendered = self.engine.render(frame, result, apply_filter=self.enable_shield)
                    else:
                        # Render blur without bounding boxes
                        rendered = frame.copy()
                        if result.stable_status == PrivacyStatus.PEEKER_DETECTED and self.enable_shield:
                            rendered = self.engine.apply_privacy_blur(rendered)
                        # Render top banner
                        rendered = self.engine.draw_annotations(rendered, EngineResult(
                            status=result.status,
                            stable_status=result.stable_status,
                            face_count=result.face_count,
                            faces=[],
                            latency_ms=result.latency_ms,
                            fps=result.fps,
                            backend_name=result.backend_name,
                            frame_size=result.frame_size,
                        ))

                    # 6. Update Thread-Safe Worker State
                    metrics = self.engine.get_measured_metrics()
                    with self.state.lock:
                        self.state.frame_bgr = frame
                        self.state.rendered_frame_bgr = rendered
                        self.state.result = result
                        self.state.metrics = metrics
                        self.state.sensor_error = False
                        
                        # Update rolling performance history
                        self.state.history_latency.append(result.latency_ms)
                        self.state.history_cpu.append(metrics.get("cpu_percent", 0.0))
                        self.state.history_timestamps.append(t_raw)
                        if len(self.state.history_latency) > 120:
                            self.state.history_latency.pop(0)
                            self.state.history_cpu.pop(0)
                            self.state.history_timestamps.pop(0)

                time.sleep(0.02)  # Maintain smooth ~30 FPS processing loop

        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if self.engine is not None:
                self.engine.close()
            with self.state.lock:
                self.state.is_running = False
            logger.info("BackgroundProtectionWorker cleanup finished.")

    def _write_incident_to_csv(self, incident: Dict[str, Any]) -> None:
        """Appends incident record to local CSV log file."""
        file_exists = os.path.exists(BREACH_LOG_FILE)
        try:
            with open(BREACH_LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Start Time", "Duration (s)", "Max Peekers", "Shield State", "Dismissed By"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(incident)
        except Exception as e:
            logger.warning(f"Could not save breach incident to CSV: {e}")


@st.cache_resource
def get_protection_worker() -> ProtectionWorker:
    """Instantiates and starts single background protection worker instance."""
    worker = ProtectionWorker()
    worker.start()
    # Register clean shutdown callback on app exit
    atexit.register(worker.stop)
    return worker

# ------------------------------------------------------------------------------
# 3. Streamlit UI Rendering Engine
# ------------------------------------------------------------------------------

# Initialize background worker
worker = get_protection_worker()

# ------------------------------------------------------------------------------
# Sidebar Controls & Configuration Panel
# ------------------------------------------------------------------------------
st.sidebar.title("🛡️ Privacy Guard Settings")
st.sidebar.markdown("---")

# Camera Hardware / Demo Source Selector
source_mode = st.sidebar.selectbox("Video Input Source", ["Webcam (Hardware)", "Demo Source (Offline / Synthetic)"])
if source_mode == "Webcam (Hardware)":
    camera_index = st.sidebar.selectbox("Webcam Device Index", [0, 1, 2], index=0)
    worker.camera_index = camera_index
    worker.use_demo_source = False
else:
    worker.use_demo_source = True

st.sidebar.markdown("---")
st.sidebar.subheader("Security Protection Toggles")

enable_shield = st.sidebar.checkbox("Enable Privacy Shield Blur", value=True)
worker.enable_shield = enable_shield

enable_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
worker.enable_boxes = enable_boxes

# OS-Level Fullscreen Screen Shield Checkbox
os_shield_available = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_shield.py"))
if os_shield_available:
    enable_os_shield = st.sidebar.checkbox("Also cover desktop workspace (OS Shield)", value=False,
                                          help="Launches topmost blurred screen overlay during breach. Press ESC on overlay to dismiss.")
    worker.enable_os_shield = enable_os_shield
else:
    st.sidebar.info("OS Screen Shield module (screen_shield.py) not detected.")

# Peeker Test Mode Peeker Simulator Toggle
test_mode_peeker = st.sidebar.checkbox("Simulate Peeker Detection (Test Mode)", value=False,
                                       help="Injects synthetic peeker face into camera input before detection to test debounce and shield path.")
worker.test_mode_peeker = test_mode_peeker

# Freeze Frame Toggle
freeze_frame = st.sidebar.checkbox("Freeze Frame", value=False)
worker.freeze_frame = freeze_frame

st.sidebar.markdown("---")
st.sidebar.subheader("Demo Telemetry Controls")
simulated_telemetry = st.sidebar.checkbox("Demo telemetry (SIMULATED)", value=False,
                                         help="Overlays simulated NPU offload benchmark metrics for video demonstrations.")

# Read Worker State
with worker.state.lock:
    current_result = worker.state.result
    current_frame = worker.state.rendered_frame_bgr
    current_metrics = worker.state.metrics.copy()
    sensor_error = worker.state.sensor_error
    error_msg = worker.state.error_message
    history_lat = list(worker.state.history_latency)
    history_cpu = list(worker.state.history_cpu)
    history_ts = list(worker.state.history_timestamps)
    time_to_shield_ms = worker.state.time_to_shield_ms
    incidents = list(worker.state.breach_incidents)

# Determine Engine NPU Status
is_npu = current_result.is_npu if current_result is not None else False
backend_name = current_result.backend_name if current_result is not None else "Initializing..."

# ------------------------------------------------------------------------------
# Header Banner with Responsive Badges
# ------------------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">HP Smart Local Privacy Guard</div>
    <div class="header-subtitle">Real-Time On-Device AI Peeker Detection & Instant Screen Shielding</div>
    <div class="badge-row">
""", unsafe_allow_html=True)

# Render Header Badges
if is_npu and not simulated_telemetry:
    st.markdown('<span class="badge-teal">⚡ Powered by Qualcomm Hexagon NPU (QNN EP)</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge-amber">💻 Running on CPU fallback (MediaPipe)</span>', unsafe_allow_html=True)

st.markdown('<span class="badge-blue">🔒 100% On-Device Local AI</span>', unsafe_allow_html=True)

if test_mode_peeker:
    st.markdown('<span class="badge-red">🧪 TEST MODE ACTIVE</span>', unsafe_allow_html=True)

if worker.use_demo_source:
    st.markdown('<span class="badge-amber">📹 DEMO SOURCE ACTIVE</span>', unsafe_allow_html=True)

if simulated_telemetry:
    st.markdown('<span class="badge-red">🔴 SIMULATED TELEMETRY MODE</span>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# One-Line Hardware Context Description
if is_npu:
    st.info("⚡ **Hardware Offload Active**: Face detection is offloaded to the Hexagon NPU via ONNX Runtime (QNN EP) to protect your screen with minimal CPU load.")
else:
    st.warning("💻 **CPU Execution Mode**: Face detection is currently running on the CPU. Connect the QNN backend on Snapdragon Windows on ARM to offload it to the NPU.")

# ------------------------------------------------------------------------------
# 2-Column Dashboard Layout (1.4 : 1 Ratio)
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([1.4, 1.0])

# ==============================================================================
# LEFT COLUMN: Live Camera Feed & Stream Controls
# ==============================================================================
with col_left:
    st.subheader("📹 Live Privacy Stream")
    
    # Render Stream Video Box
    if current_frame is not None:
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, channels="RGB", use_container_width=True)
    else:
        st.info("⏳ Initializing video stream...")

    # Stream Controls
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("▶️ Start Protection Worker", use_container_width=True, type="primary"):
            worker.start()
            st.rerun()
    with c_btn2:
        if st.button("⏹️ Stop Protection Worker", use_container_width=True):
            worker.stop()
            st.rerun()

# ==============================================================================
# RIGHT COLUMN: Status Card, Telemetry & Incident Logs
# ==============================================================================
with col_right:
    st.subheader("🛡️ Protection Status & Telemetry")
    
    # 1. Main Status Card Driven Strictly by stable_status
    if sensor_error:
        st.markdown("""
        <div class="status-card-error">
            ⚠️ SENSOR ERROR — Protection Paused<br>
            <span style="font-size: 0.9rem; font-weight: 400; opacity: 0.9;">Check webcam connection or hardware index.</span>
        </div>
        """, unsafe_allow_html=True)
    elif current_result is not None:
        stable_st = current_result.stable_status
        
        if stable_st == PrivacyStatus.SECURE:
            st.markdown(f"""
            <div class="status-card-secure">
                🟢 SECURE — {current_result.face_count} User Detected<br>
                <span style="font-size: 0.85rem; font-weight: 400; opacity: 0.9;">Primary user active. No peekers detected.</span>
            </div>
            """, unsafe_allow_html=True)
        elif stable_st == PrivacyStatus.PEEKER_DETECTED:
            shield_label = "Screen Shield Active" if enable_shield else "Screen Shield Disabled"
            st.markdown(f"""
            <div class="status-card-peeker">
                🔴 PRIVACY BREACH — Shoulder Surfer Detected!<br>
                <span style="font-size: 0.85rem; font-weight: 400; opacity: 0.9;">{shield_label}</span>
            </div>
            """, unsafe_allow_html=True)
        elif stable_st == PrivacyStatus.NO_USER:
            st.markdown("""
            <div class="status-card-away">
                🟡 ATTENTION — No User Detected<br>
                <span style="font-size: 0.85rem; font-weight: 400; opacity: 0.9;">Workstation unattended.</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card-away">🟡 INITIALIZING...</div>', unsafe_allow_html=True)

    # 2. Measured Telemetry Cards
    st.markdown("### ⚡ Performance Metrics")
    
    if simulated_telemetry:
        # Simulated Demo Telemetry Mode
        lat_val = 5.4
        fps_val = 60.0
        cpu_val = 0.8
        backend_val = "QNN Execution Provider (Snapdragon NPU)"
        power_val = 0.15
        savings_val = "~88% Energy Savings"
    else:
        # Real Measured Telemetry Mode
        lat_val = current_result.latency_ms if current_result is not None else 0.0
        fps_val = current_result.fps if current_result is not None else 0.0
        cpu_val = current_metrics.get("cpu_percent", 0.0)
        backend_val = backend_name
        # Simple Documented Power Model: NPU fixed ~0.15W vs CPU formula: 0.20W base + (cpu% / 100) * 1.80W
        power_val = 0.15 if is_npu else round(0.20 + (cpu_val / 100.0) * 1.80, 2)
        savings_val = "~88% vs CPU" if is_npu else "n/a (CPU backend)"

    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.metric("Inference Latency", f"{lat_val:.1f} ms", help="Measured detection latency (time.perf_counter)")
        st.metric("Process CPU Load", f"{cpu_val:.1f}%", help="Measured process CPU utilization (psutil)")
    with m_c2:
        st.metric("Frame Rate", f"{fps_val:.1f} FPS", help="Measured frame processing rate")
        st.metric("Active Backend", backend_val)

    p_c1, p_c2 = st.columns(2)
    with p_c1:
        st.metric("Est. Power Impact", f"~ {power_val:.2f} W", help="Estimated power based on hardware backend and measured CPU %")
    with p_c2:
        st.metric("Est. Energy Savings", savings_val)

    if time_to_shield_ms is not None and enable_os_shield:
        st.caption(f"⚡ **Measured OS Time-to-Shield**: {time_to_shield_ms:.1f} ms")

# ------------------------------------------------------------------------------
# 4. Performance Charts & Backend Benchmark Suite
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Hardware Telemetry & Power Analytics")

g_col1, g_col2 = st.columns(2)

with g_col1:
    st.markdown("##### Measured Latency & CPU Load Over Time")
    if history_lat:
        chart_df = pd.DataFrame({
            "Latency (ms)": history_lat,
            "CPU Load (%)": history_cpu,
        })
        st.line_chart(chart_df)
    else:
        st.info("Collecting performance samples...")

with g_col2:
    st.markdown("##### Estimated Power: CPU Pipeline vs NPU Offload (W)")
    st.caption("*Estimated model based on CPU load and NPU offload specifications; not a direct hardware rail measurement.*")
    if history_cpu:
        cpu_power_series = [round(0.20 + (c / 100.0) * 1.80, 2) for c in history_cpu]
        npu_power_series = [0.15] * len(history_cpu)
        power_df = pd.DataFrame({
            "CPU Pipeline (Est. W)": cpu_power_series,
            "NPU Offload (Est. W)": npu_power_series,
        })
        st.area_chart(power_df)

# Interactive Backend Benchmark Button
if st.button("🧪 Run Live Backend Benchmark (N=100 Frames)"):
    st.info("Running N=100 frame benchmark on available backends...")
    test_frame = worker._generate_synthetic_frame()
    
    # Run MediaPipe CPU benchmark
    latencies_cpu = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = worker.engine.process_frame(test_frame)
        t1 = time.perf_counter()
        latencies_cpu.append((t1 - t0) * 1000.0)
    avg_cpu_lat = np.mean(latencies_cpu)
    
    bench_df = pd.DataFrame({
        "Backend": [worker.engine.backend_name],
        "Avg Latency (ms)": [round(avg_cpu_lat, 2)],
    }).set_index("Backend")
    
    st.success(f"Benchmark finished! Average Latency: {avg_cpu_lat:.2f} ms")
    st.bar_chart(bench_df)

# ------------------------------------------------------------------------------
# 5. Breach Incident Audit Log & CSV Download
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Incident Security Log")

if incidents:
    log_df = pd.DataFrame(incidents)
    st.dataframe(log_df, use_container_width=True)
    
    # Generate CSV download buffer
    csv_buffer = io.StringIO()
    log_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Breach Incident Log (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"privacy_breach_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("No privacy breach incidents recorded during this session.")
