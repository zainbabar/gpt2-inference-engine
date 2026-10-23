# run the checks in separate processes so hooks and cache state dont leak between scripts
import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description="Run the GPT-2 correctness checks")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--output", type=Path, default=Path("results/verification-cpu.txt"))
args = parser.parse_args()
checks = [
    "tests.test_attention",
    "tests.test_logits_against_hf",
    "tests.test_greedy_against_hf",
    "tests.test_sampling",
    "tests.test_kv_cache",
]
log = [
    f"UTC: {datetime.now(UTC).isoformat()}",
    f"Python: {sys.version.split()[0]}, torch: {torch.__version__}, device: {args.device}",
]
failed = []
for script in checks:
    print(f"running {script}...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", script, "--device", args.device],
        check=False,
        capture_output=True,
        text=True,
    )
    log.extend(
        [
            f"\n$ python -m {script} --device {args.device}",
            result.stdout,
            result.stderr,
            f"exit status: {result.returncode}",
        ]
    )
    if result.returncode != 0:
        failed.append(script)
    print("FAIL" if result.returncode else "PASS", flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(log) + "\n")
print(f"saved {args.output}")
if failed:
    raise SystemExit(f"failed checks: {', '.join(failed)}")
print("all checks pass")
