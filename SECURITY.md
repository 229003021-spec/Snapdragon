# Security & Privacy Policy — HP Smart Local Privacy Guard

## 🛡️ Threat Model & Security Scope

**HP Smart Local Privacy Guard** protects confidential workstation data against physical shoulder surfing ("peeking") in public and open-office environments. 

### What We Protect Against
- Unauthorized individuals standing behind or looking over the primary user's shoulder.
- Unattended laptop screen exposure when the primary user steps away.
- Unrecognized individuals using the workstation without authorization.

---

## 🔒 Local Privacy Guarantees

### 1. 100% On-Device Intelligence
- **Zero Network Traffic**: All computer vision model inference is executed locally using Qualcomm QNN Execution Provider (Snapdragon NPU) or CPU fallbacks.
- **Zero Disk Persistence**: Raw camera video frames are held transiently in memory for inference and immediately discarded. Frames are **never written to disk or recorded**.
- **No Telemetry Tracking**: Streamlit usage telemetry is explicitly disabled (`gatherUsageStats = false`).

### 2. Process Isolation & Desktop Shielding
- The OS Screen Shield (`screen_shield.py`) runs as an isolated OS process separated from the vision app thread.
- Inter-Process Communication (IPC) signals are stored in system temp (`tempfile.gettempdir()/hp_privacy_guard/`).
- Stale signal protection ensures that if the vision engine crashes, signal files older than 3 seconds are automatically ignored.

### 3. Fail-Safe User Override & Automatic Re-arm
- Pressing `<Escape>` or clicking the designated button immediately hides the screen shield to prevent user lockout.
- Overrides are single-episode: as soon as the peeker leaves (`SECURE` or `NO_USER` state), the override is automatically cleared and the shield is re-armed.

---

## 🧪 Verification & Audit

You can audit zero-network compliance at any time by running:
```powershell
python scripts/check_offline.py
```
This script scans the codebase to verify zero outbound network calls or external URL connections.
