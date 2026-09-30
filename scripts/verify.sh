#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  HP Smart Local Privacy Guard - Automated Verification"
echo "=========================================================="

echo ""
echo "[1/3] Compiling Python Source Files..."
python -m py_compile app.py privacy_engine.py screen_shield.py test_engine.py scripts/check_offline.py
echo "  --> Syntax Compilation PASSED!"

echo ""
echo "[2/3] Running Deterministic Unit Test Suite..."
python -m pytest -q test_engine.py
echo "  --> Pytest Suite PASSED!"

echo ""
echo "[3/3] Running Offline Network Audit..."
python scripts/check_offline.py
echo "  --> Offline Audit PASSED!"

echo ""
echo "=========================================================="
echo "  🎉 ALL VERIFICATION STAGES PASSED SUCCESSFULLY!"
echo "=========================================================="
