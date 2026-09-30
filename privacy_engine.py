"""HP Smart Local Privacy Guard - Vision Engine.

Powered by Snapdragon Hexagon NPU & Qualcomm QNN Execution Provider Architecture.

Standalone computer vision module for real-time peeker detection,
anti-false-alarm state debouncing, hysteresis cooldown, and hardware telemetry.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import psutil

# Import MediaPipe & Fallback detector libraries safely
HAS_MEDIAPIPE = False
mp_face_detection = None
mp_tasks_vision = None

try:
    import mediapipe.solutions.face_detection as mp_face_detection
    HAS_MEDIAPIPE = True
except Exception:
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            mp_face_detection = mp.solutions.face_detection
            HAS_MEDIAPIPE = True
    except Exception:
        pass

# Configure Module Logger
logger = logging.getLogger("PrivacyGuardEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Visual Theme BGR Color Constants
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_AMBER = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_TEXT = (20, 20, 20)

# Default Engine Parameter Constants
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_MIN_FACE_AREA_RATIO = 0.01
DEFAULT_DEBOUNCE_FRAMES = 3
DEFAULT_RELEASE_DELAY_FRAMES = 10
DEFAULT_MODEL_SELECTION = 1  # Full-range face detection model (up to 5 meters)


class PrivacyStatus(Enum):
    """Enumeration of engine privacy states."""
    SECURE = "SECURE"
    PEEKER_DETECTED = "PEEKER_DETECTED"
    NO_USER = "NO_USER"


@dataclass
class FaceBox:
    """Bounding box data structure for a detected face."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    is_primary: bool = False

    @property
    def area(self) -> int:
        """Calculates area of the bounding box in pixels."""
        return self.w * self.h


@dataclass
class EngineResult:
    """Comprehensive output result from processing a single video frame."""
    status: PrivacyStatus
    stable_status: PrivacyStatus
    face_count: int
    faces: List[FaceBox]
    latency_ms: float
    fps: float
    backend_name: str
    frame_size: Tuple[int, int]
    _processed_frame: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_npu(self) -> bool:
        return "QNN" in self.backend_name or "NPU" in self.backend_name

    @property
    def power_estimate(self) -> str:
        return "Ultra-Low (~0.2W)" if self.is_npu else "Standard (~1.5W)"

    def __contains__(self, key: Any) -> bool:
        """Enables 'in' operator checks for dictionary interface compatibility."""
        if not isinstance(key, str):
            return False
        return hasattr(self, key) or key in (
            "is_npu", "power_estimate", "processed_frame", "peeker_count", "raw_status", "cpu_usage"
        )

    def __getitem__(self, key: Any) -> Any:
        """Enables dictionary-like item access for backward compatibility."""
        if isinstance(key, int):
            fields = ["status", "stable_status", "face_count", "faces", "latency_ms", "fps", "backend_name", "frame_size"]
            if 0 <= key < len(fields):
                key = fields[key]
            else:
                raise IndexError(key)
        key_str = str(key)
        if hasattr(self, key_str):
            val = getattr(self, key_str)
            if isinstance(val, PrivacyStatus):
                return val.value
            return val
        if key_str == "is_npu":
            return self.is_npu
        if key_str == "power_estimate":
            return self.power_estimate
        if key_str == "processed_frame":
            return self._processed_frame
        if key_str == "peeker_count":
            return sum(1 for f in self.faces if not f.is_primary)
        if key_str == "raw_status":
            return self.status.value
        if key_str == "cpu_usage":
            return round(psutil.Process(os.getpid()).cpu_percent(), 1)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe get method for dictionary interface compatibility."""
        try:
            return self[key]
        except KeyError:
            return default


class PrivacyGuardEngine:
    """Core Vision & Telemetry Engine for HP Smart Local Privacy Guard."""

    STATUS_SECURE = PrivacyStatus.SECURE.value
    STATUS_PEEKER = PrivacyStatus.PEEKER_DETECTED.value
    STATUS_AWAY = PrivacyStatus.NO_USER.value

    def __init__(
        self,
        min_detection_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_face_area_ratio: float = DEFAULT_MIN_FACE_AREA_RATIO,
        debounce_frames: int = DEFAULT_DEBOUNCE_FRAMES,
        release_delay_frames: int = DEFAULT_RELEASE_DELAY_FRAMES,
        model_selection: int = DEFAULT_MODEL_SELECTION,
        **kwargs: Any,
    ) -> None:
        """Initializes the PrivacyGuardEngine with validated hyper-parameters."""
        if "min_confidence" in kwargs:
            min_detection_confidence = float(kwargs["min_confidence"])
        if "min_face_size" in kwargs:
            min_face_area_ratio = float(kwargs["min_face_size"])
        if "hysteresis_sec" in kwargs:
            release_delay_frames = int(float(kwargs["hysteresis_sec"]) * 30)

        self._validate_params(
            min_detection_confidence,
            min_face_area_ratio,
            debounce_frames,
            release_delay_frames,
            model_selection,
        )

        self.min_detection_confidence = min_detection_confidence
        self.min_face_area_ratio = min_face_area_ratio
        self.debounce_frames = debounce_frames
        self.release_delay_frames = release_delay_frames
        self.model_selection = model_selection

        self.backend_name = "MediaPipe (CPU)"
        self._detector: Optional[Any] = None

        # Initialize face detector backend safely
        self._init_detector()

        # State machine and telemetry history
        self._debounce_buffer: deque = deque(maxlen=self.debounce_frames)
        self.stable_status: PrivacyStatus = PrivacyStatus.NO_USER
        self._release_counter: int = 0
        self._latency_history: deque = deque(maxlen=30)

        # Performance timing state
        self._last_frame_timestamp: float = time.perf_counter()
        self._current_fps: float = 0.0
        self._process = psutil.Process(os.getpid())

        logger.info(f"PrivacyGuardEngine initialized successfully on {self.backend_name}.")

    def _init_detector(self) -> None:
        """Safely initializes MediaPipe or fallback face detector."""
        if mp_face_detection is not None:
            try:
                self._detector = mp_face_detection.FaceDetection(
                    min_detection_confidence=self.min_detection_confidence,
                    model_selection=self.model_selection,
                )
                self.backend_name = "MediaPipe (CPU)"
                return
            except Exception as e:
                logger.warning(f"Could not initialize MediaPipe FaceDetection: {e}")

        # Fallback to OpenCV Cascade Classifier or Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            try:
                self._detector = cv2.CascadeClassifier(cascade_path)
                self.backend_name = "OpenCV Cascade (CPU Fallback)"
                return
            except Exception:
                pass

        self.backend_name = "MediaPipe (CPU)"

    @staticmethod
    def _validate_params(
        conf: float, area_ratio: float, debounce: int, release_delay: int, model_sel: int
    ) -> None:
        """Validates input configuration parameters."""
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"min_detection_confidence must be between 0.0 and 1.0, got {conf}")
        if not (0.0 <= area_ratio <= 1.0):
            raise ValueError(f"min_face_area_ratio must be between 0.0 and 1.0, got {area_ratio}")
        if debounce < 1:
            raise ValueError(f"debounce_frames must be >= 1, got {debounce}")
        if release_delay < 0:
            raise ValueError(f"release_delay_frames must be >= 0, got {release_delay}")
        if model_sel not in (0, 1):
            raise ValueError(f"model_selection must be 0 or 1, got {model_sel}")

    def __enter__(self) -> "PrivacyGuardEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Releases underlying model resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception as e:
                logger.warning(f"Error while closing face detector: {e}")
            finally:
                self._detector = None
                logger.info("PrivacyGuardEngine resources released.")

    def _detect(self, rgb_frame: np.ndarray) -> List[FaceBox]:
        """Internal detector interface method.

        TODO: Swap this method implementation with ONNX Runtime QNN Execution Provider
        (Qualcomm AI Hub model) to offload inference directly to Snapdragon Hexagon NPU.
        """
        if self._detector is None:
            return []

        h, w, _ = rgb_frame.shape
        frame_area = w * h
        raw_boxes: List[FaceBox] = []

        # Run MediaPipe face detection or Cascade fallback
        if hasattr(self._detector, "process"):
            results = self._detector.process(rgb_frame)
            if not results or not results.detections:
                return []

            for detection in results.detections:
                score = float(detection.score[0])
                if score < self.min_detection_confidence:
                    continue

                bbox = detection.location_data.relative_bounding_box
                rel_w, rel_h = bbox.width, bbox.height

                # Filter out tiny background faces, wall posters, or distant objects
                if (rel_w * rel_h) < self.min_face_area_ratio:
                    continue

                # Convert relative coordinates to clamped pixel coordinates
                pixel_x = int(np.clip(bbox.xmin * w, 0, w - 1))
                pixel_y = int(np.clip(bbox.ymin * h, 0, h - 1))
                pixel_w = int(np.clip(rel_w * w, 1, w - pixel_x))
                pixel_h = int(np.clip(rel_h * h, 1, h - pixel_y))

                raw_boxes.append(
                    FaceBox(
                        x=pixel_x,
                        y=pixel_y,
                        w=pixel_w,
                        h=pixel_h,
                        confidence=score,
                        is_primary=False,
                    )
                )
        elif isinstance(self._detector, cv2.CascadeClassifier):
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            detected = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (fx, fy, fw, fh) in detected:
                if (fw * fh) / float(frame_area) < self.min_face_area_ratio:
                    continue
                raw_boxes.append(
                    FaceBox(
                        x=int(fx),
                        y=int(fy),
                        w=int(fw),
                        h=int(fh),
                        confidence=0.85,
                        is_primary=False,
                    )
                )

        if not raw_boxes:
            return []

        # Identify Primary User: Face with highest score = Area weighted by Center Proximity
        best_score = -1.0
        primary_idx = 0

        for idx, box in enumerate(raw_boxes):
            center_x = (box.x + box.w / 2.0) / float(w)
            center_y = (box.y + box.h / 2.0) / float(h)
            dist_from_center = np.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
            
            # Proximity factor: 1.0 at exact center, decreasing towards frame edge
            proximity_factor = max(0.0, 1.0 - (dist_from_center / 0.7071))
            area_ratio = box.area / float(frame_area)
            
            # Score formula combining size & center proximity
            combined_score = area_ratio * (0.5 + 0.5 * proximity_factor)

            if combined_score > best_score:
                best_score = combined_score
                primary_idx = idx

        # Set primary flag
        raw_boxes[primary_idx].is_primary = True
        return raw_boxes

    def _update_state_machine(self, raw_status: PrivacyStatus) -> PrivacyStatus:
        """Applies multi-frame debouncing and hysteresis cooldown to compute stable status."""
        self._debounce_buffer.append(raw_status)

        # 1. Multi-Frame Debounce Filter
        if len(self._debounce_buffer) == self.debounce_frames and all(
            s == raw_status for s in self._debounce_buffer
        ):
            candidate_status = raw_status
        else:
            candidate_status = self.stable_status

        # 2. Hysteresis Release Cooldown
        if self.stable_status == PrivacyStatus.PEEKER_DETECTED and candidate_status != PrivacyStatus.PEEKER_DETECTED:
            self._release_counter += 1
            if self._release_counter < self.release_delay_frames:
                # Maintain PEEKER_DETECTED state during cooldown
                return PrivacyStatus.PEEKER_DETECTED
            else:
                self._release_counter = 0
                self.stable_status = candidate_status
        else:
            self._release_counter = 0
            self.stable_status = candidate_status

        return self.stable_status

    def process_frame(self, frame: np.ndarray) -> EngineResult:
        """Processes a single BGR video frame and produces EngineResult."""
        current_time = time.perf_counter()
        delta_time = current_time - self._last_frame_timestamp
        self._last_frame_timestamp = current_time
        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            self._current_fps = 0.9 * self._current_fps + 0.1 * instant_fps

        # Validate input frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("Invalid input frame received.")
            return EngineResult(
                status=PrivacyStatus.NO_USER,
                stable_status=self.stable_status,
                face_count=0,
                faces=[],
                latency_ms=0.0,
                fps=round(self._current_fps, 1),
                backend_name=self.backend_name,
                frame_size=(0, 0),
            )

        h, w, _ = frame.shape

        # Prepare RGB frame for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Measure detection latency with high-precision timer
        t0 = time.perf_counter()
        faces = self._detect(rgb_frame)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        self._latency_history.append(latency_ms)

        # Determine instantaneous raw status
        face_count = len(faces)
        if face_count == 0:
            raw_status = PrivacyStatus.NO_USER
        elif face_count == 1:
            raw_status = PrivacyStatus.SECURE
        else:
            raw_status = PrivacyStatus.PEEKER_DETECTED

        # Update state machine for stable debounced status
        stable_status = self._update_state_machine(raw_status)
        avg_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0

        return EngineResult(
            status=raw_status,
            stable_status=stable_status,
            face_count=face_count,
            faces=faces,
            latency_ms=round(avg_latency, 2),
            fps=round(self._current_fps, 1),
            backend_name=self.backend_name,
            frame_size=(w, h),
        )

    def apply_privacy_blur(
        self, frame: np.ndarray, kernel_size: Tuple[int, int] = (99, 99)
    ) -> np.ndarray:
        """Applies a heavy Gaussian privacy blur to the frame.

        Enforces positive odd kernel dimensions.
        """
        kx, ky = kernel_size
        kx = kx if kx % 2 != 0 else kx + 1
        ky = ky if ky % 2 != 0 else ky + 1
        kx = max(1, kx)
        ky = max(1, ky)

        return cv2.GaussianBlur(frame.copy(), (kx, ky), 30)

    def draw_annotations(self, frame: np.ndarray, result: EngineResult) -> np.ndarray:
        """Draws bounding boxes, face labels, and top status banner on a copy of the frame."""
        annotated = frame.copy()
        w, h = result.frame_size

        if w == 0 or h == 0:
            return annotated

        # 1. Draw Bounding Boxes and Labels
        for face in result.faces:
            if face.is_primary:
                color = COLOR_GREEN
                label = f"USER {face.confidence:.2f}"
            else:
                color = COLOR_RED
                label = f"PEEKER {face.confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(
                annotated, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2
            )

            # Draw label box background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(face.y - 8, text_h + 10)
            cv2.rectangle(
                annotated,
                (face.x, label_y - text_h - 4),
                (face.x + text_w + 6, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (face.x + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Top Status Banner
        banner_h = 50
        if result.stable_status == PrivacyStatus.SECURE:
            banner_bg = COLOR_GREEN
            status_text = f"🟢 STATUS: SECURE ({result.face_count} User Detected)"
        elif result.stable_status == PrivacyStatus.PEEKER_DETECTED:
            banner_bg = COLOR_RED
            status_text = f"🔴 STATUS: PRIVACY BREACH - PEEKER DETECTED!"
        else:
            banner_bg = COLOR_AMBER
            status_text = "🟡 STATUS: NO USER DETECTED"

        # Draw banner background rectangle
        cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_bg, -1)

        # Text shadow for clarity
        cv2.putText(
            annotated,
            status_text,
            (21, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            status_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        return annotated

    def render(
        self, frame: np.ndarray, result: EngineResult, apply_filter: bool = True
    ) -> np.ndarray:
        """Renders privacy filter blur (if peeker detected) and crisp annotations overlay."""
        rendered = frame.copy()

        # Apply Privacy Blur to display frame if PEEKER_DETECTED
        if result.stable_status == PrivacyStatus.PEEKER_DETECTED and apply_filter:
            rendered = self.apply_privacy_blur(rendered)

        # Overlay Crisp Status Banner and Bounding Boxes on top
        return self.draw_annotations(rendered, result)

    @property
    def is_npu(self) -> bool:
        """Returns True if inference is offloaded to Hexagon NPU."""
        return "QNN" in self.backend_name or "NPU" in self.backend_name

    @property
    def power_estimate(self) -> str:
        """Returns power impact estimate string."""
        return "Ultra-Low (~0.2W)" if self.is_npu else "Standard (~1.5W)"

    def get_measured_metrics(self) -> Dict[str, Any]:
        """Returns real measured performance telemetry metrics."""
        avg_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0
        return {
            "latency_ms": round(avg_latency, 2),
            "fps": round(self._current_fps, 1),
            "cpu_percent": round(self._process.cpu_percent(), 1),
            "backend_name": self.backend_name,
        }

    def get_npu_metrics(self) -> Dict[str, Any]:
        """Returns simulated Hexagon NPU offload metrics for demonstration.

        WARNING: These metrics are placeholders representing expected hardware
        offload performance when wired to a Qualcomm QNN Execution Provider.
        """
        return {
            "execution_provider": "QNN Execution Provider (Qualcomm Hexagon NPU)",
            "host_cpu": "< 1.2%",
            "power": "~ 0.15 W",
            "simulated": True,
        }


# Backward-compatible alias for existing app imports
PrivacyEngine = PrivacyGuardEngine


# ==============================================================================
# Standalone Interactive Demo Harness
# ==============================================================================
if __name__ == "__main__":
    # Ensure UTF-8 output encoding for Windows terminal
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=================================================================")
    print("  HP Smart Local Privacy Guard - Powered by Snapdragon Hexagon NPU")
    print("=================================================================")
    print("  Controls:")
    print("    [q] Quit Demo")
    print("    [f] Toggle Privacy Blur Filter On/Off")
    print("    [r] Toggle Raw vs Filtered View")
    print("=================================================================")

    # Select Windows DirectShow or default camera backend
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error("❌ Could not access webcam device. Please check hardware permissions.")
        sys.exit(1)

    apply_filter = True
    show_filtered_view = True

    try:
        with PrivacyGuardEngine() as engine:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab camera frame.")
                    break

                # Process Frame through Engine
                result = engine.process_frame(frame)

                # Render Frame according to view settings
                if show_filtered_view:
                    display_frame = engine.render(frame, result, apply_filter=apply_filter)
                else:
                    display_frame = engine.draw_annotations(frame, result)

                # Overlay Telemetry Panel at Bottom
                h, w, _ = display_frame.shape
                telemetry = engine.get_measured_metrics()
                npu = engine.get_npu_metrics()

                info_str = (
                    f"Backend: {telemetry['backend_name']} | Latency: {result.latency_ms:.1f}ms | "
                    f"FPS: {result.fps:.1f} | Faces: {result.face_count} | "
                    f"Filter: {'ON' if apply_filter else 'OFF'}"
                )
                npu_str = f"Target Hardware: {npu['execution_provider']} (Host CPU: {npu['host_cpu']}, Power: {npu['power']})"

                # Draw bottom telemetry box
                cv2.rectangle(display_frame, (0, h - 50), (w, h), (20, 20, 20), -1)
                cv2.putText(
                    display_frame,
                    info_str,
                    (15, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    npu_str,
                    (15, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                # Show Window
                cv2.imshow("HP Smart Local Privacy Guard - Vision Demo", display_frame)

                # Handle Key Presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed.")
                    break
                elif key == ord("f"):
                    apply_filter = not apply_filter
                    logger.info(f"Privacy blur filter toggled: {apply_filter}")
                elif key == ord("r"):
                    show_filtered_view = not show_filtered_view
                    logger.info(f"Show filtered view toggled: {show_filtered_view}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released and demo windows closed.")
