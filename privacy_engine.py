"""HP Smart Local Privacy Guard - Vision Engine.

Powered by Snapdragon Hexagon NPU & Qualcomm QNN Execution Provider Architecture.

Standalone computer vision module for real-time peeker detection,
anti-false-alarm state debouncing, hysteresis cooldown, and hardware telemetry.
"""

from abc import ABC, abstractmethod
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
COLOR_PURPLE = (200, 0, 200)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_TEXT = (20, 20, 20)


class PrivacyStatus(Enum):
    """Enumeration of engine privacy states."""
    SECURE = "SECURE"
    PEEKER_DETECTED = "PEEKER_DETECTED"
    NO_USER = "NO_USER"
    UNKNOWN_USER = "UNKNOWN_USER"
    ENGINE_ERROR = "ENGINE_ERROR"


@dataclass
class FaceBox:
    """Bounding box data structure for a detected face."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    is_primary: bool = False
    track_id: int = -1

    @property
    def area(self) -> int:
        """Calculates area of the bounding box in pixels."""
        return max(0, self.w) * max(0, self.h)


@dataclass
class EngineConfig:
    """Configuration parameters for PrivacyGuardEngine."""
    min_detection_confidence: float = 0.4
    min_face_area_ratio: float = 0.01          # Minimum 1% of total frame area
    peeker_relative_size_ratio: float = 0.15   # Peeker must be >= 15% size of primary face
    debounce_frames: int = 3
    release_delay_sec: float = 1.0
    model_selection: int = 0                   # 0 = short range (<2m), 1 = full range (<5m)


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
    def peeker_count(self) -> int:
        """Returns the number of non-primary (peeker) faces detected."""
        return sum(1 for f in self.faces if not f.is_primary)

    @property
    def is_npu(self) -> bool:
        """Returns True if the backend is running on Hexagon NPU via QNN Execution Provider."""
        return "QNN" in self.backend_name or "NPU" in self.backend_name

    @property
    def power_estimate(self) -> str:
        """Returns estimated hardware power impact string."""
        return "Ultra-Low (~0.2W)" if self.is_npu else "Standard (~1.5W)"


# ==============================================================================
# Abstract Face Detector Backend Interface & Concrete Implementations
# ==============================================================================

class FaceDetectorBackend(ABC):
    """Abstract base class for modular face detector inference backends."""

    @abstractmethod
    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        """Runs face detection on an RGB frame and returns raw FaceBox list."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable description of the active hardware/model backend."""
        pass

    def close(self) -> None:
        """Releases backend resources cleanly."""
        pass


class OnnxQnnBackend(FaceDetectorBackend):
    """ONNX Runtime backend supporting QNN Execution Provider on Snapdragon NPU."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._name = "ONNX Runtime (CPU Fallback)"
        self.session = None

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", "face_detector.onnx"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at {model_path}")

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is not installed.")

        available_providers = ort.get_available_providers()
        sess_options = ort.SessionOptions()

        if "QNNExecutionProvider" in available_providers:
            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"backend_path": "QnnHtp.dll"}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self.session = ort.InferenceSession(
            model_path, sess_options, providers=providers, provider_options=provider_options
        )

        active_providers = self.session.get_providers()
        if "QNNExecutionProvider" in active_providers:
            self._name = "ONNX Runtime (QNN NPU)"
            logger.info("ONNX Runtime initialized with Qualcomm QNN Execution Provider on Hexagon NPU.")
        else:
            self._name = "ONNX Runtime (CPU Fallback)"
            logger.info("ONNX Runtime initialized with CPU Execution Provider.")

    @property
    def name(self) -> str:
        return self._name

    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        if self.session is None:
            return []

        h, w, _ = rgb_frame.shape
        input_meta = self.session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape  # e.g., [1, 3, 300, 300] or [1, 300, 300, 3]

        # Resize and normalize frame according to expected model input dimensions
        target_h, target_w = 300, 300
        if len(input_shape) == 4:
            if input_shape[1] == 3:
                target_h, target_w = input_shape[2], input_shape[3]
            elif input_shape[3] == 3:
                target_h, target_w = input_shape[1], input_shape[2]

        resized = cv2.resize(rgb_frame, (target_w, target_h))
        normalized = (resized.astype(np.float32) - 127.5) / 127.5

        if len(input_shape) == 4 and input_shape[1] == 3:
            inp = np.transpose(normalized, (2, 0, 1))
            inp = np.expand_dims(inp, axis=0)
        else:
            inp = np.expand_dims(normalized, axis=0)

        outputs = self.session.run(None, {input_name: inp})
        boxes: List[FaceBox] = []

        # Parse output tensors (assumes SSD/YOLO standard output format [num_dets, 6])
        if outputs and len(outputs) > 0:
            dets = outputs[0]
            if dets.ndim == 3:
                dets = dets[0]
            for det in dets:
                if len(det) >= 6:
                    score = float(det[2])
                    if score < config.min_detection_confidence:
                        continue
                    x1 = int(det[3] * w)
                    y1 = int(det[4] * h)
                    x2 = int(det[5] * w)
                    y2 = int(det[6] if len(det) > 6 else det[5] * h)
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    if (bw * bh) / float(w * h) < config.min_face_area_ratio:
                        continue
                    boxes.append(FaceBox(x=x1, y=y1, w=bw, h=bh, confidence=score))

        return boxes

    def close(self) -> None:
        self.session = None


class MediaPipeBackend(FaceDetectorBackend):
    """MediaPipe face detection backend with Tasks API and Legacy API support."""

    def __init__(self, config: EngineConfig) -> None:
        self._name = "MediaPipe (CPU)"
        self._detector: Any = None
        self._is_tasks = False

        # 1. Try MediaPipe Tasks API with local TFLite model
        try:
            from mediapipe.tasks import python as mp_python
            tflite_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", "blaze_face_short_range.tflite"
            )
            if os.path.exists(tflite_path):
                options = mp_python.vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=tflite_path),
                    min_detection_confidence=config.min_detection_confidence,
                )
                self._detector = mp_python.vision.FaceDetector.create_from_options(options)
                self._name = "MediaPipe Tasks (CPU)"
                self._is_tasks = True
                logger.info("MediaPipe Tasks FaceDetector initialized successfully.")
                return
        except Exception as e:
            logger.debug(f"MediaPipe Tasks API init skipped: {e}")

        # 2. Try MediaPipe Legacy Solutions API
        try:
            import mediapipe.solutions.face_detection as mp_face_detection
            self._detector = mp_face_detection.FaceDetection(
                min_detection_confidence=config.min_detection_confidence,
                model_selection=config.model_selection,
            )
            self._name = "MediaPipe Solutions (CPU)"
            logger.info("MediaPipe Legacy Solutions FaceDetection initialized.")
            return
        except Exception as e:
            logger.debug(f"MediaPipe Solutions API init skipped: {e}")

        raise RuntimeError("No working MediaPipe implementation available on system.")

    @property
    def name(self) -> str:
        return self._name

    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        if self._detector is None:
            return []

        h, w, _ = rgb_frame.shape
        frame_area = w * h
        boxes: List[FaceBox] = []

        if self._is_tasks:
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            res = self._detector.detect(mp_img)
            if res and res.detections:
                for det in res.detections:
                    score = float(det.categories[0].score) if det.categories else 0.8
                    if score < config.min_detection_confidence:
                        continue
                    bbox = det.bounding_box
                    px = int(np.clip(bbox.origin_x, 0, w - 1))
                    py = int(np.clip(bbox.origin_y, 0, h - 1))
                    pw = int(np.clip(bbox.width, 1, w - px))
                    ph = int(np.clip(bbox.height, 1, h - py))

                    if (pw * ph) / float(frame_area) < config.min_face_area_ratio:
                        continue

                    boxes.append(FaceBox(x=px, y=py, w=pw, h=ph, confidence=score))
        else:
            results = self._detector.process(rgb_frame)
            if results and results.detections:
                for det in results.detections:
                    score = float(det.score[0])
                    if score < config.min_detection_confidence:
                        continue
                    bbox = det.location_data.relative_bounding_box
                    rel_w, rel_h = bbox.width, bbox.height
                    if (rel_w * rel_h) < config.min_face_area_ratio:
                        continue

                    px = int(np.clip(bbox.xmin * w, 0, w - 1))
                    py = int(np.clip(bbox.ymin * h, 0, h - 1))
                    pw = int(np.clip(rel_w * w, 1, w - px))
                    ph = int(np.clip(rel_h * h, 1, h - py))

                    boxes.append(FaceBox(x=px, y=py, w=pw, h=ph, confidence=score))

        return boxes

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None


class OpenCVCascadeBackend(FaceDetectorBackend):
    """OpenCV Haar Cascade Classifier fallback backend."""

    def __init__(self, config: EngineConfig) -> None:
        self._name = "OpenCV Cascade (CPU Fallback)"
        cascade_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "haarcascade_frontalface_default.xml"
        )
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise FileNotFoundError("OpenCV Haar cascade XML model file not found.")

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar Cascade Classifier.")

        logger.info("OpenCV Haar Cascade Classifier backend loaded.")

    @property
    def name(self) -> str:
        return self._name

    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        frame_area = w * h
        detected = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        boxes: List[FaceBox] = []

        for (fx, fy, fw, fh) in detected:
            if (fw * fh) / float(frame_area) < config.min_face_area_ratio:
                continue
            boxes.append(FaceBox(x=int(fx), y=int(fy), w=int(fw), h=int(fh), confidence=0.85))

        return boxes

    def close(self) -> None:
        self.cascade = None


# ==============================================================================
# Simple Centroid & IoU Face Tracker
# ==============================================================================

class SimpleFaceTracker:
    """Tracks face bounding boxes across frames to maintain stable track IDs."""

    def __init__(self, max_disappeared: int = 5, distance_threshold: float = 80.0) -> None:
        self.next_track_id = 1
        self.tracked_faces: Dict[int, Tuple[int, int, int, int]] = {}  # id -> (x, y, w, h)
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def update(self, detected_boxes: List[FaceBox]) -> List[FaceBox]:
        if not detected_boxes:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracked_faces[tid]
                    del self.disappeared[tid]
            return []

        if not self.tracked_faces:
            for box in detected_boxes:
                box.track_id = self.next_track_id
                self.tracked_faces[self.next_track_id] = (box.x, box.y, box.w, box.h)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1
            return detected_boxes

        # Match new boxes with tracked centroids
        track_ids = list(self.tracked_faces.keys())
        track_centers = [
            (x + w / 2.0, y + h / 2.0) for (x, y, w, h) in self.tracked_faces.values()
        ]

        assigned_tracks = set()
        for box in detected_boxes:
            bx, by = box.x + box.w / 2.0, box.y + box.h / 2.0
            best_dist = float("inf")
            best_tid = -1

            for idx, tid in enumerate(track_ids):
                if tid in assigned_tracks:
                    continue
                tcx, tcy = track_centers[idx]
                dist = np.hypot(bx - tcx, by - tcy)
                if dist < best_dist and dist < self.distance_threshold:
                    best_dist = dist
                    best_tid = tid

            if best_tid != -1:
                box.track_id = best_tid
                self.tracked_faces[best_tid] = (box.x, box.y, box.w, box.h)
                self.disappeared[best_tid] = 0
                assigned_tracks.add(best_tid)
            else:
                box.track_id = self.next_track_id
                self.tracked_faces[self.next_track_id] = (box.x, box.y, box.w, box.h)
                self.disappeared[self.next_track_id] = 0
                self.next_track_id += 1

        # Mark disappeared tracks
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracked_faces[tid]
                    del self.disappeared[tid]

        return detected_boxes


# ==============================================================================
# Core Vision Engine Class
# ==============================================================================

class PrivacyGuardEngine:
    """Core Vision & Telemetry Engine for HP Smart Local Privacy Guard."""

    STATUS_SECURE = PrivacyStatus.SECURE.value
    STATUS_PEEKER = PrivacyStatus.PEEKER_DETECTED.value
    STATUS_AWAY = PrivacyStatus.NO_USER.value
    STATUS_UNKNOWN = PrivacyStatus.UNKNOWN_USER.value
    STATUS_ERROR = PrivacyStatus.ENGINE_ERROR.value

    def __init__(
        self,
        min_detection_confidence: float = 0.4,
        min_face_area_ratio: float = 0.01,
        debounce_frames: int = 3,
        release_delay_sec: float = 1.0,
        model_selection: int = 0,
        backend: Optional[FaceDetectorBackend] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes PrivacyGuardEngine with configuration parameters and detector backend."""
        if "min_confidence" in kwargs:
            min_detection_confidence = float(kwargs["min_confidence"])
        if "min_face_size" in kwargs:
            min_face_area_ratio = float(kwargs["min_face_size"])
        if "hysteresis_sec" in kwargs:
            release_delay_sec = float(kwargs["hysteresis_sec"])

        self.config = EngineConfig(
            min_detection_confidence=min_detection_confidence,
            min_face_area_ratio=min_face_area_ratio,
            peeker_relative_size_ratio=float(kwargs.get("peeker_relative_size_ratio", 0.15)),
            debounce_frames=max(1, debounce_frames),
            release_delay_sec=max(0.0, release_delay_sec),
            model_selection=model_selection,
        )

        self._tracker = SimpleFaceTracker()
        self._owner_histogram: Optional[np.ndarray] = None

        # Backend selection hierarchy
        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._auto_select_backend()

        # State machine and telemetry history
        self._debounce_buffer: deque = deque(maxlen=self.config.debounce_frames)
        self.stable_status: PrivacyStatus = PrivacyStatus.NO_USER
        self._release_counter: int = 0
        self._latency_history: deque = deque(maxlen=30)

        # Performance timing & process monitor
        self._last_frame_timestamp: float = time.perf_counter()
        self._current_fps: float = 0.0
        self._process = psutil.Process(os.getpid())
        # Call cpu_percent once during init to set baseline for psutil
        try:
            self._process.cpu_percent()
        except Exception:
            pass

        logger.info(f"PrivacyGuardEngine initialized successfully on {self.backend_name}.")

    def _auto_select_backend(self) -> FaceDetectorBackend:
        """Attempts to load backends in order: QNN ONNX -> MediaPipe -> OpenCV Cascade."""
        # 1. Try ONNX QNN / CPU Backend
        try:
            onnx_backend = OnnxQnnBackend()
            return onnx_backend
        except Exception as e:
            logger.debug(f"ONNX QNN backend unavailable: {e}")

        # 2. Try MediaPipe Backend
        try:
            mp_backend = MediaPipeBackend(self.config)
            return mp_backend
        except Exception as e:
            logger.debug(f"MediaPipe backend unavailable: {e}")

        # 3. Fallback to OpenCV Cascade Backend
        try:
            cascade_backend = OpenCVCascadeBackend(self.config)
            return cascade_backend
        except Exception as e:
            logger.error(f"OpenCV Cascade backend unavailable: {e}")
            raise RuntimeError("No valid computer vision face detection backend could be loaded!")

    @property
    def backend_name(self) -> str:
        """Returns the name of the active face detector backend."""
        return self.backend.name if self.backend else "Unknown"

    @property
    def is_npu(self) -> bool:
        """Returns True if the backend is offloaded to Hexagon NPU via QNN Execution Provider."""
        return "QNN" in self.backend_name or "NPU" in self.backend_name

    @property
    def power_estimate(self) -> str:
        """Returns estimated power consumption string."""
        return "Ultra-Low (~0.2W)" if self.is_npu else "Standard (~1.5W)"

    def update_config(self, cfg: EngineConfig) -> None:
        """Updates engine configuration parameters without resetting detector resources."""
        self.config = cfg
        if self.config.debounce_frames != self._debounce_buffer.maxlen:
            self._debounce_buffer = deque(maxlen=self.config.debounce_frames)
        logger.info("Engine configuration updated.")

    def enroll_owner(self, frame: np.ndarray, face_box: FaceBox) -> bool:
        """Enrolls owner's face color histogram in-memory for unknown user verification."""
        if frame is None or face_box is None:
            return False
        h, w, _ = frame.shape
        x1 = max(0, face_box.x)
        y1 = max(0, face_box.y)
        x2 = min(w, face_box.x + face_box.w)
        y2 = min(h, face_box.y + face_box.h)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        self._owner_histogram = hist
        logger.info("Primary user face profile enrolled successfully.")
        return True

    def clear_owner_enrollment(self) -> None:
        """Clears enrolled owner profile."""
        self._owner_histogram = None
        logger.info("Enrolled owner face profile cleared.")

    def __enter__(self) -> "PrivacyGuardEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Releases underlying model resources."""
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception as e:
                logger.warning(f"Error while closing backend: {e}")
            finally:
                self.backend = None
                logger.info("PrivacyGuardEngine resources released.")

    def _filter_and_select_primary(
        self, raw_boxes: List[FaceBox], frame_w: int, frame_h: int
    ) -> List[FaceBox]:
        """Filters faces using relative size threshold and assigns primary user flag."""
        if not raw_boxes:
            return []

        frame_area = float(frame_w * frame_h)

        # Calculate primary candidate score for each box: area ratio weighted by center proximity
        best_score = -1.0
        primary_idx = 0

        for idx, box in enumerate(raw_boxes):
            cx = (box.x + box.w / 2.0) / float(frame_w)
            cy = (box.y + box.h / 2.0) / float(frame_h)
            dist_from_center = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            proximity_factor = max(0.0, 1.0 - (dist_from_center / 0.7071))
            area_ratio = box.area / frame_area
            combined_score = area_ratio * (0.5 + 0.5 * proximity_factor)

            if combined_score > best_score:
                best_score = combined_score
                primary_idx = idx

        primary_box = raw_boxes[primary_idx]
        primary_box.is_primary = True
        primary_area = primary_box.area

        # Relative Peeker Filter: Non-primary face must be >= peeker_relative_size_ratio of primary face
        filtered_faces: List[FaceBox] = [primary_box]
        min_peeker_area = primary_area * self.config.peeker_relative_size_ratio

        for idx, box in enumerate(raw_boxes):
            if idx == primary_idx:
                continue
            if box.area >= min_peeker_area:
                box.is_primary = False
                filtered_faces.append(box)

        # Update face tracker for persistent track_ids
        return self._tracker.update(filtered_faces)

    def _verify_owner_similarity(self, frame: np.ndarray, primary_box: FaceBox) -> bool:
        """Verifies single primary face against enrolled owner histogram."""
        if self._owner_histogram is None:
            return True

        h, w, _ = frame.shape
        x1 = max(0, primary_box.x)
        y1 = max(0, primary_box.y)
        x2 = min(w, primary_box.x + primary_box.w)
        y2 = min(h, primary_box.y + primary_box.h)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        similarity = cv2.compareHist(self._owner_histogram, hist, cv2.HISTCMP_CORREL)
        return similarity >= 0.50

    def _update_state_machine(self, raw_status: PrivacyStatus) -> PrivacyStatus:
        """Applies multi-frame debouncing and hysteresis cooldown to compute stable status."""
        self._debounce_buffer.append(raw_status)

        # 1. Multi-Frame Debounce Filter
        if len(self._debounce_buffer) == self.config.debounce_frames and all(
            s == raw_status for s in self._debounce_buffer
        ):
            candidate_status = raw_status
        else:
            candidate_status = self.stable_status

        # Dynamic Hysteresis Cooldown Frame Count
        release_delay_frames = int(self.config.release_delay_sec * max(self._current_fps, 1.0))

        # 2. Hysteresis Release Cooldown
        if self.stable_status == PrivacyStatus.PEEKER_DETECTED and candidate_status != PrivacyStatus.PEEKER_DETECTED:
            self._release_counter += 1
            if self._release_counter < release_delay_frames:
                return PrivacyStatus.PEEKER_DETECTED
            else:
                self._release_counter = 0
                self.stable_status = candidate_status
        else:
            self._release_counter = 0
            self.stable_status = candidate_status

        return self.stable_status

    def process_frame(self, frame: np.ndarray) -> EngineResult:
        """Processes a single BGR video frame and returns EngineResult."""
        current_time = time.perf_counter()
        delta_time = current_time - self._last_frame_timestamp
        self._last_frame_timestamp = current_time
        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            self._current_fps = 0.9 * self._current_fps + 0.1 * instant_fps

        # Input Validation & Safe Exception Handling
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

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            t0 = time.perf_counter()
            raw_detected = self.backend.detect(rgb_frame, self.config)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            self._latency_history.append(latency_ms)

            # Apply relative size filtering, tracker, and primary user assignment
            faces = self._filter_and_select_primary(raw_detected, w, h)
            face_count = len(faces)

            # Determine instantaneous raw status
            if face_count == 0:
                raw_status = PrivacyStatus.NO_USER
            elif face_count == 1:
                # Check owner similarity if enrolled
                if self._owner_histogram is not None and not self._verify_owner_similarity(frame, faces[0]):
                    raw_status = PrivacyStatus.UNKNOWN_USER
                else:
                    raw_status = PrivacyStatus.SECURE
            else:
                raw_status = PrivacyStatus.PEEKER_DETECTED

            # Update debounced state machine
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

        except Exception as e:
            logger.error(f"Error encountered during process_frame execution: {e}", exc_info=True)
            return EngineResult(
                status=PrivacyStatus.ENGINE_ERROR,
                stable_status=PrivacyStatus.ENGINE_ERROR,
                face_count=0,
                faces=[],
                latency_ms=0.0,
                fps=round(self._current_fps, 1),
                backend_name=self.backend_name,
                frame_size=(w, h),
            )

    def apply_privacy_blur(
        self, frame: np.ndarray, kernel_size: Tuple[int, int] = (99, 99)
    ) -> np.ndarray:
        """Applies a heavy Gaussian privacy blur to the frame."""
        kx, ky = kernel_size
        kx = kx if kx % 2 != 0 else kx + 1
        ky = ky if ky % 2 != 0 else ky + 1
        kx = max(1, kx)
        ky = max(1, ky)

        return cv2.GaussianBlur(frame.copy(), (kx, ky), 30)

    def draw_annotations(self, frame: np.ndarray, result: EngineResult) -> np.ndarray:
        """Draws bounding boxes, face labels, and top status banner using ASCII-only text."""
        annotated = frame.copy()
        w, h = result.frame_size

        if w == 0 or h == 0:
            return annotated

        # 1. Draw Bounding Boxes and ASCII Labels
        for face in result.faces:
            if face.is_primary:
                color = COLOR_GREEN
                label = f"USER {face.confidence:.2f} [ID:{face.track_id}]"
            else:
                color = COLOR_RED
                label = f"PEEKER {face.confidence:.2f} [ID:{face.track_id}]"

            cv2.rectangle(
                annotated, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2
            )

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2
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
                0.50,
                COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Top Status Banner (ASCII-only strings)
        banner_h = 50
        if result.stable_status == PrivacyStatus.SECURE:
            banner_bg = COLOR_GREEN
            status_text = f"[SECURE] Primary User Active ({result.face_count} Face Detected)"
        elif result.stable_status == PrivacyStatus.PEEKER_DETECTED:
            banner_bg = COLOR_RED
            status_text = f"[PRIVACY BREACH] Peeker Detected! ({result.peeker_count} Shoulder Surfer)"
        elif result.stable_status == PrivacyStatus.UNKNOWN_USER:
            banner_bg = COLOR_PURPLE
            status_text = "[UNKNOWN USER] Unrecognized Face Detected!"
        elif result.stable_status == PrivacyStatus.ENGINE_ERROR:
            banner_bg = COLOR_AMBER
            status_text = "[ENGINE ERROR] Vision Pipeline Exception!"
        else:
            banner_bg = COLOR_AMBER
            status_text = "[NO USER] Workstation Unattended"

        cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_bg, -1)

        # Text shadow for crisp legibility
        cv2.putText(
            annotated,
            status_text,
            (21, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            status_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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

        if result.stable_status == PrivacyStatus.PEEKER_DETECTED and apply_filter:
            rendered = self.apply_privacy_blur(rendered)

        return self.draw_annotations(rendered, result)

    def get_measured_metrics(self) -> Dict[str, Any]:
        """Returns real measured performance telemetry metrics."""
        avg_latency = float(np.mean(self._latency_history)) if self._latency_history else 0.0
        try:
            cpu_val = round(self._process.cpu_percent(), 1)
        except Exception:
            cpu_val = 0.0

        return {
            "latency_ms": round(avg_latency, 2),
            "fps": round(self._current_fps, 1),
            "cpu_percent": cpu_val,
            "backend_name": self.backend_name,
        }


# Backward-compatible alias
PrivacyEngine = PrivacyGuardEngine


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=================================================================")
    print("  HP Smart Local Privacy Guard - Vision Engine Standard Harness")
    print("=================================================================")
    
    with PrivacyGuardEngine() as engine:
        print(f"[+] Loaded Backend: {engine.backend_name}")
        print(f"[+] NPU Acceleration: {engine.is_npu}")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = engine.process_frame(dummy_frame)
        print(f"[+] Test Frame Result: Status={res.status.value}, StableStatus={res.stable_status.value}, Latency={res.latency_ms}ms")
