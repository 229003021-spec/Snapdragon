"""HP Smart Local Privacy Guard - Deterministic Engine Test Suite.

Runs automated unit and integration tests for state machine debouncing,
hysteresis cooldown, relative peeker threshold filtering, face tracking,
owner enrollment unknown user detection, and robust engine exception handling.

Can be run directly or via pytest:
    python test_engine.py
    pytest -q test_engine.py
"""

import io
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

# Ensure local workspace import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from privacy_engine import (
    PrivacyGuardEngine,
    PrivacyStatus,
    FaceBox,
    EngineConfig,
    FaceDetectorBackend,
    EngineResult,
)


class ScriptedBackend(FaceDetectorBackend):
    """Scripted mock backend returning pre-determined sequences of FaceBox lists."""

    def __init__(self, scripted_frames: List[List[FaceBox]], backend_name: str = "Scripted Mock Backend") -> None:
        self.scripted_frames = scripted_frames
        self._name = backend_name
        self.frame_index = 0

    @property
    def name(self) -> str:
        return self._name

    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        if not self.scripted_frames:
            return []
        boxes = self.scripted_frames[self.frame_index % len(self.scripted_frames)]
        self.frame_index += 1
        # Return copies of FaceBox objects
        return [FaceBox(x=b.x, y=b.y, w=b.w, h=b.h, confidence=b.confidence) for b in boxes]


class ErrorBackend(FaceDetectorBackend):
    """Mock backend that intentionally raises an exception during detection."""

    @property
    def name(self) -> str:
        return "Error Simulation Backend"

    def detect(self, rgb_frame: np.ndarray, config: EngineConfig) -> List[FaceBox]:
        raise RuntimeError("Simulated Hardware Detection Failure")


def create_dummy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8) + 50


# ==============================================================================
# Unit Test Cases
# ==============================================================================

def test_state_machine_debouncing() -> None:
    """Verifies multi-frame debouncing prevents single-frame false alarms."""
    # 2 frames of peeker, then 1 face
    frame_sequence = [
        [FaceBox(100, 100, 100, 100, 0.9), FaceBox(400, 100, 100, 100, 0.85)],  # Frame 1
        [FaceBox(100, 100, 100, 100, 0.9), FaceBox(400, 100, 100, 100, 0.85)],  # Frame 2
        [FaceBox(100, 100, 100, 100, 0.9), FaceBox(400, 100, 100, 100, 0.85)],  # Frame 3
    ]
    scripted = ScriptedBackend(frame_sequence)
    engine = PrivacyGuardEngine(debounce_frames=3, release_delay_sec=1.0, backend=scripted)

    dummy = create_dummy_frame()

    # Frame 1: Raw status PEEKER_DETECTED, but debounced stable_status is NO_USER
    res1 = engine.process_frame(dummy)
    assert res1.status == PrivacyStatus.PEEKER_DETECTED
    assert res1.stable_status == PrivacyStatus.NO_USER

    # Frame 2: Still debouncing
    res2 = engine.process_frame(dummy)
    assert res2.status == PrivacyStatus.PEEKER_DETECTED
    assert res2.stable_status == PrivacyStatus.NO_USER

    # Frame 3: 3 consecutive peeker frames -> stable_status becomes PEEKER_DETECTED
    res3 = engine.process_frame(dummy)
    assert res3.status == PrivacyStatus.PEEKER_DETECTED
    assert res3.stable_status == PrivacyStatus.PEEKER_DETECTED


def test_hysteresis_release_cooldown() -> None:
    """Verifies hysteresis cooldown maintains shield during transient peeker disappearance."""
    # 3 peeker frames to trigger, then 1 face frame
    peeker_boxes = [FaceBox(100, 100, 100, 100, 0.9), FaceBox(400, 100, 100, 100, 0.85)]
    single_box = [FaceBox(100, 100, 100, 100, 0.9)]

    sequence = [peeker_boxes, peeker_boxes, peeker_boxes, single_box]
    scripted = ScriptedBackend(sequence)

    # Set hysteresis to 10.0 seconds so release delay is long
    engine = PrivacyGuardEngine(debounce_frames=3, release_delay_sec=10.0, backend=scripted)
    dummy = create_dummy_frame()

    for _ in range(3):
        engine.process_frame(dummy)

    assert engine.stable_status == PrivacyStatus.PEEKER_DETECTED

    # Frame 4 has single face, but hysteresis cooldown keeps stable status as PEEKER_DETECTED
    res4 = engine.process_frame(dummy)
    assert res4.status == PrivacyStatus.SECURE
    assert res4.stable_status == PrivacyStatus.PEEKER_DETECTED


def test_relative_peeker_size_filter() -> None:
    """Verifies non-primary faces below peeker_relative_size_ratio are ignored as background noise."""
    primary_user = FaceBox(x=200, y=100, w=200, h=200, confidence=0.95)  # Area = 40,000
    tiny_noise = FaceBox(x=500, y=50, w=30, h=30, confidence=0.80)       # Area = 900 (< 15% of 40,000 = 6,000)
    real_peeker = FaceBox(x=450, y=100, w=150, h=150, confidence=0.85)    # Area = 22,500 (>= 15%)

    # Test 1: Primary + Tiny Noise -> Should be filtered to 1 face (SECURE)
    scripted_noise = ScriptedBackend([[primary_user, tiny_noise]])
    engine_noise = PrivacyGuardEngine(
        debounce_frames=1, peeker_relative_size_ratio=0.15, backend=scripted_noise
    )
    res_noise = engine_noise.process_frame(create_dummy_frame())
    assert res_noise.face_count == 1
    assert res_noise.status == PrivacyStatus.SECURE

    # Test 2: Primary + Real Peeker -> Should count both faces (PEEKER_DETECTED)
    scripted_peeker = ScriptedBackend([[primary_user, real_peeker]])
    engine_peeker = PrivacyGuardEngine(
        debounce_frames=1, peeker_relative_size_ratio=0.15, backend=scripted_peeker
    )
    res_peeker = engine_peeker.process_frame(create_dummy_frame())
    assert res_peeker.face_count == 2
    assert res_peeker.status == PrivacyStatus.PEEKER_DETECTED


def test_owner_enrollment_and_unknown_user() -> None:
    """Verifies owner enrollment triggers UNKNOWN_USER for unrecognized faces."""
    single_face = FaceBox(x=100, y=100, w=100, h=100, confidence=0.90)
    scripted = ScriptedBackend([[single_face]])
    engine = PrivacyGuardEngine(debounce_frames=1, backend=scripted)

    frame_owner = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_owner[100:200, 100:200] = (0, 200, 0)  # Green owner face

    # Enroll owner
    assert engine.enroll_owner(frame_owner, single_face) is True

    # Same frame -> SECURE
    res_owner = engine.process_frame(frame_owner)
    assert res_owner.status == PrivacyStatus.SECURE

    # Unrecognized blue face frame -> UNKNOWN_USER
    frame_stranger = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_stranger[100:200, 100:200] = (200, 0, 0)  # Blue stranger face
    res_stranger = engine.process_frame(frame_stranger)
    assert res_stranger.status == PrivacyStatus.UNKNOWN_USER


def test_engine_error_handling() -> None:
    """Verifies exceptions during detection return ENGINE_ERROR status without crashing app."""
    err_backend = ErrorBackend()
    engine = PrivacyGuardEngine(backend=err_backend)

    res = engine.process_frame(create_dummy_frame())
    assert res.status == PrivacyStatus.ENGINE_ERROR
    assert res.stable_status == PrivacyStatus.ENGINE_ERROR


def test_update_config() -> None:
    """Verifies engine.update_config dynamic updates."""
    scripted = ScriptedBackend([[FaceBox(100, 100, 100, 100, 0.9)]])
    engine = PrivacyGuardEngine(debounce_frames=2, backend=scripted)

    new_cfg = EngineConfig(debounce_frames=5, min_detection_confidence=0.7)
    engine.update_config(new_cfg)

    assert engine.config.debounce_frames == 5
    assert engine.config.min_detection_confidence == 0.7


# ==============================================================================
# Direct Command-Line Test Suite Harness
# ==============================================================================

def run_all_tests() -> None:
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("==========================================================")
    print("  HP Smart Local Privacy Guard - Deterministic Test Suite")
    print("==========================================================")

    print("[1/6] Running test_state_machine_debouncing...")
    test_state_machine_debouncing()
    print("      ✅ PASSED!")

    print("[2/6] Running test_hysteresis_release_cooldown...")
    test_hysteresis_release_cooldown()
    print("      ✅ PASSED!")

    print("[3/6] Running test_relative_peeker_size_filter...")
    test_relative_peeker_size_filter()
    print("      ✅ PASSED!")

    print("[4/6] Running test_owner_enrollment_and_unknown_user...")
    test_owner_enrollment_and_unknown_user()
    print("      ✅ PASSED!")

    print("[5/6] Running test_engine_error_handling...")
    test_engine_error_handling()
    print("      ✅ PASSED!")

    print("[6/6] Running test_update_config...")
    test_update_config()
    print("      ✅ PASSED!")

    print("==========================================================")
    print("  🎉 ALL 6 DETERMINISTIC UNIT TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")


if __name__ == "__main__":
    run_all_tests()
