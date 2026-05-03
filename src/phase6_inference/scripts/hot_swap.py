"""
phase6_inference/scripts/hot_swap.py

Manual model hot-swap for the local vLLM server. The "primary → coder" swap
target is Qwen2.5-Coder-14B-Instruct at Q4_K_M (8.99 GB), used for compiled-
language code generation per the architecture doc.

Strategy: stop the running vllm process, start a new one with MODEL=<target>.
This is intentionally simple (file-based PID handling) — vLLM does not
natively support runtime model swap on a single GPU, so the swap takes
~3-10 seconds depending on NVMe speed.

Usage:
    python -m src.phase6_inference.scripts.hot_swap --model Qwen/Qwen2.5-Coder-14B-Instruct
    python -m src.phase6_inference.scripts.hot_swap --model Qwen/Qwen3.5-9B  # back to primary
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PIDFILE = Path("/tmp/rtpi-vllm.pid")
SERVE_SCRIPT = Path(__file__).resolve().parents[1] / "server" / "vllm_serve.sh"
HEALTH_URL = "http://localhost:8000/health"


def _stop_running():
    if not PIDFILE.exists():
        logger.info("No PID file at %s; assuming nothing to stop.", PIDFILE)
        return
    pid = int(PIDFILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to vLLM pid %d", pid)
    except ProcessLookupError:
        logger.info("Process %d already gone", pid)
    # Wait up to 30s for graceful shutdown
    for _ in range(30):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    PIDFILE.unlink(missing_ok=True)


def _start_with_model(model: str, port: int, max_model_len: int) -> int:
    env = {**os.environ, "MODEL": model, "PORT": str(port), "MAX_MODEL_LEN": str(max_model_len)}
    logger.info("Starting vLLM with MODEL=%s", model)
    proc = subprocess.Popen(
        ["bash", str(SERVE_SCRIPT)],
        env=env,
        start_new_session=True,
    )
    PIDFILE.write_text(str(proc.pid))
    logger.info("vLLM pid=%d (written to %s)", proc.pid, PIDFILE)
    return proc.pid


def _wait_healthy(timeout_s: int = 120) -> bool:
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description="Swap the model behind the local vLLM server")
    parser.add_argument("--model", required=True,
                        help="HF model id, e.g. Qwen/Qwen3.5-9B or Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--health-timeout", type=int, default=120)
    args = parser.parse_args()

    t0 = time.time()
    _stop_running()
    _start_with_model(args.model, args.port, args.max_model_len)

    if not _wait_healthy(args.health_timeout):
        logger.error("vLLM did not become healthy within %ds", args.health_timeout)
        sys.exit(1)
    logger.info("Hot-swap to %s complete in %.1fs", args.model, time.time() - t0)


if __name__ == "__main__":
    main()
