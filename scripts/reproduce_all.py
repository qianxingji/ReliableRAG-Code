#!/usr/bin/env python3
"""Run the complete public, non-neural reproduction workflow."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> int:
    run("-m", "unittest", "discover", "-s", "tests", "-v")
    run("scripts/reproduce_paper_statistics.py")
    run("scripts/verify_cas_q3_claim_statistics.py")
    run("scripts/verify_repository.py")
    print("PASS_COMPLETE_PUBLIC_REPRODUCTION_WORKFLOW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
