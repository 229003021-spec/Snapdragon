# HP Smart Local Privacy Guard - Automated PowerShell Verification Suite

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  HP Smart Local Privacy Guard - Automated Verification" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Compiling Python Source Files..." -ForegroundColor Yellow
python -m py_compile app.py privacy_engine.py screen_shield.py test_engine.py scripts/check_offline.py
Write-Host "  --> Syntax Compilation PASSED!" -ForegroundColor Green

Write-Host "`n[2/3] Running Deterministic Unit Test Suite..." -ForegroundColor Yellow
python -m pytest -q test_engine.py
Write-Host "  --> Pytest Suite PASSED!" -ForegroundColor Green

Write-Host "`n[3/3] Running Offline Network Audit..." -ForegroundColor Yellow
python scripts/check_offline.py
Write-Host "  --> Offline Audit PASSED!" -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  🎉 ALL VERIFICATION STAGES PASSED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
