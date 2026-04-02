from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from textwrap import dedent


MODULE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = MODULE_DIR.parent.parent / "prompts"
_BAML_PROMPTS_FILE = PROMPTS_DIR / "orchestrator_prompts.baml"
_BAML_PROMPT_PATTERN = re.compile(
    r"function\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*->\s*[^{]+\{\s*prompt\s+#\"\"\"(?P<prompt>.*?)\"\"\"#",
    re.DOTALL,
)

_PROMPT_FILES = {
    "GeneratorAgent": PROMPTS_DIR / "generator_agent.md",
    "InvestigatorAgent": PROMPTS_DIR / "investigator_agent.md",
    "ReflectorAgent": PROMPTS_DIR / "reflector_agent.md",
}


def _clean_baml_prompt(raw: str) -> str:
    cleaned = dedent(raw).strip("\n")
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("{{") and "_.role" in stripped:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


@lru_cache()
def _load_prompt_file(name: str) -> str:
    path = _PROMPT_FILES.get(name)
    if not path:
        raise KeyError(name)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt file not found: {path}") from exc
    return dedent(text).strip()


@lru_cache()
def _load_baml_prompts() -> dict[str, str]:
    if not _BAML_PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {_BAML_PROMPTS_FILE}")

    text = _BAML_PROMPTS_FILE.read_text(encoding="utf-8")
    prompts: dict[str, str] = {}
    for match in _BAML_PROMPT_PATTERN.finditer(text):
        prompts[match.group("name")] = _clean_baml_prompt(match.group("prompt"))

    if not prompts:
        raise ValueError(f"No prompts parsed from {_BAML_PROMPTS_FILE}")
    return prompts


def get_agent_instructions(name: str, fallback: str) -> str:
    try:
        return _load_prompt_file(name)
    except Exception:
        pass
    try:
        prompt = _load_baml_prompts()[name]
        if not prompt:
            raise ValueError(f"Prompt '{name}' is empty")
        return prompt
    except Exception as exc:
        print(f"[warn] Failed to load BAML prompt '{name}': {exc}. Using fallback instructions.")
        return fallback


__all__ = ["PROMPTS_DIR", "get_agent_instructions"]
