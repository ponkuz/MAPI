from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

from mapi.config import MapiConfig


class ConfigUsageTests(unittest.TestCase):
    def test_every_declared_config_field_is_consumed_outside_config_module(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [
            path
            for folder in (root / "mapi", root / "examples")
            for path in folder.rglob("*.py")
            if path.name != "config.py"
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        unused = [field.name for field in fields(MapiConfig) if field.name not in source]
        self.assertEqual(unused, [], f"Unused MapiConfig fields: {unused}")


if __name__ == "__main__":
    unittest.main()
