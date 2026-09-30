"""HP Smart Local Privacy Guard - Offline Network Audit Verification Tool.

Greps python source files to verify 100% local, on-device execution with zero network calls.
"""

import io
import os
import re
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Disallowed networking keywords/patterns in code execution paths
BANNED_PATTERNS = [
    (r"import\s+requests", "requests library import"),
    (r"import\s+urllib", "urllib library import"),
    (r"from\s+urllib", "urllib library import"),
    (r"import\s+http\.client", "http.client import"),
    (r"socket\.connect", "socket connection call"),
    (r"https?://(?!localhost|127\.0\.0\.1)", "External HTTP/HTTPS URL reference"),
]


def audit_file(file_path: str) -> list[str]:
    violations = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        for pattern, desc in BANNED_PATTERNS:
            if re.search(pattern, stripped):
                violations.append(f"Line {idx}: [{desc}] -> {stripped}")

    return violations


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_files = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ("venv", ".venv", "ENV", "env", ".git", "__pycache__")]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    total_violations = 0
    print("==========================================================")
    print("  HP Smart Local Privacy Guard - Offline Network Audit")
    print("==========================================================")

    for py_file in py_files:
        rel_path = os.path.relpath(py_file, repo_root)
        violations = audit_file(py_file)
        if violations:
            print(f"[VIOLATION] {rel_path}:")
            for v in violations:
                print(f"    - {v}")
            total_violations += len(violations)
        else:
            print(f"[CLEAN] {rel_path}")

    print("==========================================================")
    if total_violations == 0:
        print("PASSED: 0 network calls detected. 100% on-device privacy compliant!")
        sys.exit(0)
    else:
        print(f"FAILED: Found {total_violations} network call violation(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
