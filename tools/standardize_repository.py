from __future__ import annotations

import ast
import io
import re
import shutil
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OLD_PACKAGE = ROOT / "simulator_v2"
NEW_PACKAGE = ROOT / "pu_dcgp_comsol"


def strip_docstrings(source: str) -> str:
    tree = ast.parse(source)
    ranges: list[tuple[int, int]] = []
    nodes = [tree]
    nodes.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    for node in nodes:
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ranges.append((first.lineno, first.end_lineno or first.lineno))
    lines = source.splitlines(keepends=True)
    for start, end in sorted(ranges, reverse=True):
        del lines[start - 1 : end]
    return "".join(lines)


def strip_comments(source: str) -> str:
    tokens = []
    stream = io.StringIO(source)
    for token in tokenize.generate_tokens(stream.readline):
        if token.type != tokenize.COMMENT:
            tokens.append(token)
    return tokenize.untokenize(tokens)


def clean_python(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = strip_docstrings(source)
    source = strip_comments(source)
    source = re.sub(r"[ \t]+\n", "\n", source)
    source = re.sub(r"\n{4,}", "\n\n\n", source)
    path.write_text(source.rstrip() + "\n", encoding="utf-8")


def rename_package() -> None:
    if OLD_PACKAGE.exists() and not NEW_PACKAGE.exists():
        OLD_PACKAGE.rename(NEW_PACKAGE)
    old_phase = NEW_PACKAGE / "phase_h"
    new_phase = NEW_PACKAGE / "comsol"
    if old_phase.exists() and not new_phase.exists():
        old_phase.rename(new_phase)


def rename_modules() -> None:
    module_dir = NEW_PACKAGE / "comsol"
    if not module_dir.exists():
        return
    for path in sorted(module_dir.glob("h11_*.py")):
        target = path.with_name(path.name.removeprefix("h11_"))
        if target.exists():
            raise FileExistsError(target)
        path.rename(target)


def update_text_files() -> None:
    replacements = {
        "simulator_v2.phase_h.h11_": "pu_dcgp_comsol.comsol.",
        "simulator_v2.phase_h": "pu_dcgp_comsol.comsol",
        "simulator_v2/phase_h/": "pu_dcgp_comsol/comsol/",
        "simulator_v2\\phase_h\\": "pu_dcgp_comsol\\comsol\\",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.resolve() == SCRIPT:
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".yml", ".yaml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def clean_all_python() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts and path.resolve() != SCRIPT:
            clean_python(path)


def ensure_packages() -> None:
    for directory in (NEW_PACKAGE, NEW_PACKAGE / "comsol"):
        directory.mkdir(parents=True, exist_ok=True)
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")


def remove_caches() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob("*.py[co]"):
        path.unlink()


def main() -> None:
    rename_package()
    rename_modules()
    ensure_packages()
    update_text_files()
    clean_all_python()
    remove_caches()


if __name__ == "__main__":
    main()
