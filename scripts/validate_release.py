"""Release validation helper for lite_llama_npu.

The script intentionally stays CPU-friendly so it can run on a laptop before a
branch is pushed and inside an Ascend container before server benchmarking.
"""

from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def check_utf8_markdown() -> None:
    bad: list[Path] = []
    for path in [*ROOT.glob("*.md"), *ROOT.joinpath("docs").rglob("*.md")]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\ufffd" in text:
            bad.append(path.relative_to(ROOT))
    if bad:
        raise SystemExit("Invalid UTF-8 replacement characters in: " + ", ".join(map(str, bad)))
    print("markdown utf-8 scan ok", flush=True)


def check_version_docs() -> None:
    version = ROOT.joinpath("VERSION").read_text(encoding="utf-8").strip()
    release_doc = ROOT / "docs" / "releases" / f"v{version}.md"
    if not release_doc.exists():
        raise SystemExit(f"missing release document: {release_doc.relative_to(ROOT)}")
    for readme in ("README.md", "README_CN.md"):
        text = ROOT.joinpath(readme).read_text(encoding="utf-8")
        badge = f"version-{version}-blue"
        if badge not in text:
            raise SystemExit(f"{readme} badge does not match VERSION={version}")
        if f"v{version}" not in text:
            raise SystemExit(f"{readme} does not link current release v{version}")
    changelog = ROOT.joinpath("CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        raise SystemExit(f"CHANGELOG.md is missing current VERSION={version}")
    docs_index = ROOT.joinpath("docs/README.md").read_text(encoding="utf-8")
    if f"releases/v{version}.md" not in docs_index:
        raise SystemExit(f"docs/README.md does not index v{version}")
    print(f"version docs ok: {version}", flush=True)


def main() -> int:
    check_utf8_markdown()
    check_version_docs()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])
    if not compileall.compile_dir(ROOT / "lite_llama", quiet=1):
        return 1
    for file_name in ("server.py",):
        if not compileall.compile_file(ROOT / file_name, quiet=1):
            return 1
    run(["git", "diff", "--check"])
    print("release validation ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
