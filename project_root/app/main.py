"""
Offline LLM agent entry point (Windows, no installation).

Setup:
1) Put model file at: project_root/models/model.gguf
2) Put llama executable at: project_root/bin/llama.exe
   Download prebuilt llama.cpp binaries:
   https://github.com/ggerganov/llama.cpp/releases
"""

from pathlib import Path

from agent import OfflineAgent


def main() -> int:
    # Resolve project root from this file location.
    project_root = Path(__file__).resolve().parents[1]

    # Initialize agent and validate runtime files.
    try:
        agent = OfflineAgent(project_root)
    except Exception as exc:
        print(f"[FATAL] {exc}")
        return 1

    print("Offline Agent ready. Type 'exit' to quit.")

    # Simple CLI loop.
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        try:
            answer = agent.ask(user_text)
        except Exception as exc:
            print(f"Agent> [ERROR] {exc}\n")
            continue

        print(f"Agent> {answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
