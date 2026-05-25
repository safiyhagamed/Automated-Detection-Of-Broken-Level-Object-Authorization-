# run_pipeline.py
"""
Run the entire BOLA risk prediction pipeline with coloured progress.
All phases executed in order; stops on first error.
"""

import subprocess
import sys
from pathlib import Path

# ---------- colour helpers ----------
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

PHASES = [
    # ("Phase 1 – Parse",             "src/phase1_parse.py"),
    ("Phase 2 – Safe Features",     "src/phase2_features_safe.py"),
    ("Phase 3 – Split & Label",     "src/phase3_split.py"),
    ("Phase 4 – Train & Test",      "src/phase4_silver_train.py"),
    ("Phase 5 – Report",            "src/phase5_report.py"),
]

def run_phase(name, script):
    print(f"\n{CYAN}{'─'*64}{RESET}")
    print(f"  {BOLD}{CYAN}{name}{RESET}  ({script})")
    print(f"{CYAN}{'─'*64}{RESET}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"\n{RED}  ✗ {name} FAILED (exit code {result.returncode}){RESET}")
        sys.exit(1)
    print(f"  {GREEN}✓ {name} completed{RESET}")

def main():
    print(f"{BOLD}{YELLOW}═══════════════════════════════════════════{RESET}")
    print(f"{BOLD}{YELLOW}  BOLA Risk Prediction – Full Pipeline{RESET}")
    print(f"{BOLD}{YELLOW}═══════════════════════════════════════════{RESET}")
    for name, script in PHASES:
        if not Path(script).exists():
            print(f"{RED}ERROR: Script '{script}' not found. Aborting.{RESET}")
            sys.exit(1)
        run_phase(name, script)
    print(f"\n{BOLD}{GREEN}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{GREEN}║  All phases completed successfully!     ║{RESET}")
    print(f"{BOLD}{GREEN}║  Report: outputs/reports/evaluation_report.html ║{RESET}")
    print(f"{BOLD}{GREEN}╚══════════════════════════════════════════╝{RESET}")

if __name__ == "__main__":
    main()