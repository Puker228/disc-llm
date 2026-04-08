"""
Simple offline ReAct-style agent.

Tools:
- file_read: reads local .txt files from project folder
- shell: runs simple safe allow-listed Windows commands
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from utils import call_llama, validate_runtime

MAX_STEPS = 4
MAX_HISTORY_TURNS = 4
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_FILE_READ_CHARS = 4000

FORBIDDEN_SHELL_CHARS = re.compile(r"[&|><;\r\n]")
ALLOWED_SHELL_COMMANDS = {
    "cd",
    "dir",
    "echo",
    "findstr",
    "hostname",
    "type",
    "ver",
    "where",
    "whoami",
}


class OfflineAgent:
    def __init__(self, project_root: Path) -> None:
        # Validate required runtime files once at startup.
        self.root = Path(project_root).resolve()
        validate_runtime(self.root)
        self.history: List[Tuple[str, str]] = []

    def ask(self, user_input: str) -> str:
        # ReAct loop: model -> optional tool -> observation -> model.
        scratchpad = ""

        for _ in range(MAX_STEPS):
            prompt = self._build_prompt(user_input, scratchpad)
            model_output = call_llama(prompt, root=self.root)

            parsed = self._parse_model_output(model_output)
            action = parsed["action"]
            action_input = parsed["action_input"]
            final_answer = parsed["final_answer"]

            if final_answer:
                self._remember(user_input, final_answer)
                return final_answer

            if action in {"file_read", "shell"}:
                observation = self._run_tool(action, action_input)
                scratchpad += (
                    f"Action: {action}\n"
                    f"Action Input: {action_input}\n"
                    f"Observation: {observation}\n\n"
                )
                continue

            # Fallback when model output is not structured.
            fallback = model_output.strip() or "No response generated."
            self._remember(user_input, fallback)
            return fallback

        timeout_message = "Tool-step limit reached. Please ask a more specific question."
        self._remember(user_input, timeout_message)
        return timeout_message

    def _build_prompt(self, user_input: str, scratchpad: str) -> str:
        history_text = self._format_history()

        return (
            "You are an offline assistant running on Windows.\n"
            "Use tools only when needed.\n\n"
            "Available tools:\n"
            "1) file_read\n"
            "   - Reads a local .txt file inside the project folder.\n"
            "2) shell\n"
            "   - Executes a simple safe Windows command.\n\n"
            "Output format rules:\n"
            "- To use a tool:\n"
            "  Action: file_read OR shell\n"
            "  Action Input: <single line input>\n"
            "- To answer user:\n"
            "  Final Answer: <your response>\n"
            "- Prefer Final Answer when tools are not required.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"User: {user_input}\n"
            f"{scratchpad}"
            "Assistant:\n"
        )

    def _parse_model_output(self, text: str) -> Dict[str, str]:
        action = ""
        action_input = ""
        final_answer = ""

        action_match = re.search(r"^\s*Action\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if action_match:
            action = action_match.group(1).strip().lower()

        action_input_match = re.search(
            r"^\s*Action\s*Input\s*:\s*(.+?)\s*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if action_input_match:
            action_input = action_input_match.group(1).strip().strip('"').strip("'")

        final_match = re.search(
            r"^\s*Final\s*Answer\s*:\s*(.*)$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if final_match:
            # Keep everything after "Final Answer:" to support multiline output.
            final_answer = text[final_match.start(1):].strip()

        return {
            "action": action,
            "action_input": action_input,
            "final_answer": final_answer,
        }

    def _run_tool(self, action: str, action_input: str) -> str:
        if action == "file_read":
            return self._tool_file_read(action_input)
        if action == "shell":
            return self._tool_shell(action_input)
        return f"ERROR: Unknown tool '{action}'."

    def _tool_file_read(self, file_path_arg: str) -> str:
        # Read .txt files only from inside project root.
        raw_path = file_path_arg.strip().strip('"').strip("'")
        if not raw_path:
            return "ERROR: file_read requires a path."

        requested = Path(raw_path)
        candidate = (self.root / requested).resolve() if not requested.is_absolute() else requested.resolve()

        if candidate.suffix.lower() != ".txt":
            return "ERROR: file_read supports only .txt files."
        if not self._is_within_root(candidate):
            return "ERROR: Access denied. File must be inside project folder."
        if not candidate.exists() or not candidate.is_file():
            return f"ERROR: File not found: {candidate}"

        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: Unable to read file: {exc}"

        return self._truncate(content, MAX_FILE_READ_CHARS)

    def _tool_shell(self, command: str) -> str:
        # Run allow-listed commands only, with blocked shell operators.
        cmd = command.strip()
        if not cmd:
            return "ERROR: shell requires a command."
        if FORBIDDEN_SHELL_CHARS.search(cmd):
            return "ERROR: Unsafe shell syntax detected."

        try:
            parts = shlex.split(cmd, posix=False)
        except ValueError as exc:
            return f"ERROR: Invalid command syntax: {exc}"

        if not parts:
            return "ERROR: Empty command."
        if parts[0].lower() not in ALLOWED_SHELL_COMMANDS:
            return f"ERROR: Command '{parts[0]}' is not allowed."

        try:
            result = subprocess.run(
                ["cmd.exe", "/C", cmd],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out (15 seconds)."
        except OSError as exc:
            return f"ERROR: Unable to execute command: {exc}"

        output = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if err:
            output = f"{output}\n{err}".strip()

        if not output:
            output = "(no output)"
        if result.returncode != 0:
            output = f"[exit code {result.returncode}] {output}"

        return self._truncate(output, MAX_TOOL_OUTPUT_CHARS)

    def _format_history(self) -> str:
        if not self.history:
            return "(empty)"

        lines: List[str] = []
        for user_text, answer in self.history[-MAX_HISTORY_TURNS:]:
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    def _remember(self, user_text: str, answer: str) -> None:
        self.history.append((user_text, answer))

    def _is_within_root(self, path: Path) -> bool:
        # Normalize case for Windows drive-letter comparisons.
        try:
            root_norm = os.path.normcase(str(self.root))
            path_norm = os.path.normcase(str(path))
            return os.path.commonpath([path_norm, root_norm]) == root_norm
        except ValueError:
            return False

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated to {limit} chars]"
