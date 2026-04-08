"""
Utility wrapper for calling llama.cpp as an external executable.

Setup:
1) Place GGUF model at: project_root/models/model.gguf
2) Place llama executable at: project_root/bin/llama.exe
   Prebuilt binaries: https://github.com/ggerganov/llama.cpp/releases
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_N_PREDICT = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONTEXT = 2048
DEFAULT_TIMEOUT_SEC = 180

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def get_project_root() -> Path:
    # /project_root/app/utils.py -> /project_root
    return Path(__file__).resolve().parents[1]


def get_llama_exe(root: Optional[Path] = None) -> Path:
    return (root or get_project_root()) / "bin" / "llama.exe"


def get_model_file(root: Optional[Path] = None) -> Path:
    return (root or get_project_root()) / "models" / "model.gguf"


def validate_runtime(root: Optional[Path] = None) -> None:
    # Validate critical files before running the agent loop.
    base = (root or get_project_root()).resolve()
    missing = []

    if not (base / "python" / "python.exe").exists():
        missing.append(str(base / "python" / "python.exe"))
    if not get_llama_exe(base).exists():
        missing.append(str(get_llama_exe(base)))
    if not get_model_file(base).exists():
        missing.append(str(get_model_file(base)))

    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing required files:\n{details}")


def _clean_llama_output(stdout: str, prompt: str) -> str:
    # Remove ANSI escapes and echoed prompt for clean display.
    text = _ANSI_RE.sub("", (stdout or "")).strip()
    if not text:
        return ""

    prompt_index = text.find(prompt)
    if prompt_index != -1:
        text = text[prompt_index + len(prompt):].lstrip()

    cleaned_lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("llama_print_timings"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def call_llama(
    prompt: str,
    root: Optional[Path] = None,
    n_predict: int = DEFAULT_N_PREDICT,
    temperature: float = DEFAULT_TEMPERATURE,
    context: int = DEFAULT_CONTEXT,
    threads: Optional[int] = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> str:
    # Call external llama.cpp executable using subprocess.
    base = (root or get_project_root()).resolve()
    exe_path = get_llama_exe(base)
    model_path = get_model_file(base)

    if not exe_path.exists():
        raise FileNotFoundError(f"llama executable not found: {exe_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    worker_threads = threads or max(1, (os.cpu_count() or 2) - 1)

    # Required invocation pattern:
    # llama.exe -m models/model.gguf -p "PROMPT" -n 512
    cmd = [
        str(exe_path),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(context),
        "-t",
        str(worker_threads),
        "--temp",
        str(temperature),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(base),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"llama.exe timed out after {timeout_sec} seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to start llama.exe: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "(no stderr output)"
        raise RuntimeError(f"llama.exe failed with exit code {result.returncode}: {stderr}")

    cleaned = _clean_llama_output(result.stdout, prompt)
    if cleaned:
        return cleaned

    fallback = (result.stderr or "").strip()
    return fallback or "No response generated."
