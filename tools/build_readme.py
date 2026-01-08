from __future__ import annotations

import re
from pathlib import Path

INCLUDE_RE = re.compile(r"<!--\s*INCLUDE:\s*(.+?)\s*-->")


def build_readme() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    template_path = repo_root / "docs" / "README.template.md"
    output_path = repo_root / "README.md"

    template_text = template_path.read_text(encoding="utf-8")
    output_lines: list[str] = []

    for line in template_text.splitlines():
        match = INCLUDE_RE.search(line)
        if not match:
            output_lines.append(line)
            continue

        include_rel = match.group(1)
        include_path = (template_path.parent / include_rel).resolve()
        if not include_path.exists():
            raise FileNotFoundError(f"Include not found: {include_rel}")

        include_text = include_path.read_text(encoding="utf-8").rstrip()
        output_lines.append(include_text)

    output_text = "\n".join(output_lines).rstrip() + "\n"
    output_path.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    build_readme()
