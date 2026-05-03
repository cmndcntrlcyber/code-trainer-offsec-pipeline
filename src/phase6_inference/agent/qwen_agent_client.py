"""
phase6_inference/agent/qwen_agent_client.py

Wire Qwen-Agent to the local vLLM endpoint with the documented MCP tool set.
Mirrors docs/plan/Inference-Agent-Architecture.md lines 95-115.

Run vLLM separately first (src/phase6_inference/server/vllm_serve.sh), then:

    python -m src.phase6_inference.agent.qwen_agent_client \
        --query "Summarize the README in two sentences."
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MCP_CONFIG = Path(__file__).parent / "mcp_servers.json"


def build_agent(
    model: str = "Qwen/Qwen3.5-9B",
    server: str = "http://localhost:8000/v1",
    mcp_config: Path = DEFAULT_MCP_CONFIG,
    enable_code_interpreter: bool = True,
):
    """Construct a Qwen-Agent Assistant wired to local vLLM + MCP servers.

    Imports happen inside the function so this module is importable without
    qwen-agent installed (which it isn't in the project's pyproject — Phase 6
    deps are managed separately, see scripts/setup.md).
    """
    from qwen_agent.agents import Assistant

    llm_cfg = {
        "model": model,
        "model_server": server,
        "api_key": os.environ.get("VLLM_API_KEY", "EMPTY"),
    }

    tools = []
    if mcp_config.exists():
        tools.append(json.loads(mcp_config.read_text()))
    if enable_code_interpreter:
        tools.append("code_interpreter")

    return Assistant(llm=llm_cfg, function_list=tools)


def main():
    parser = argparse.ArgumentParser(description="Qwen-Agent + vLLM + MCP smoke client")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--server", default="http://localhost:8000/v1")
    parser.add_argument("--mcp-config", default=str(DEFAULT_MCP_CONFIG))
    parser.add_argument("--no-code-interpreter", action="store_true")
    parser.add_argument("--query", required=True,
                        help="Single-turn user message to send to the agent")
    args = parser.parse_args()

    agent = build_agent(
        model=args.model,
        server=args.server,
        mcp_config=Path(args.mcp_config),
        enable_code_interpreter=not args.no_code_interpreter,
    )

    messages = [{"role": "user", "content": args.query}]
    for chunk in agent.run(messages):
        # qwen-agent yields incremental message lists; print the final assistant text
        if chunk and isinstance(chunk, list):
            last = chunk[-1]
            if last.get("role") == "assistant" and last.get("content"):
                sys.stdout.write(last["content"])
                sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
