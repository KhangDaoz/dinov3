import ast
from pathlib import Path


def test_entry_points_are_valid_python() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = sorted((root / "scripts").glob("*.py"))
    assert scripts
    for script in scripts:
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
