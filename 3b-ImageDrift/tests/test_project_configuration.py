"""Contract tests for the image MMD demo's uv project."""

from __future__ import annotations

import unittest
from pathlib import Path

import tomllib

DEMO_ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigurationTests(unittest.TestCase):
    """Verify that the demo exposes separate predictor and local runtimes."""

    def test_uv_project_separates_predictor_and_local_dependencies(self) -> None:
        """The base environment stays small while local extra enables the notebook."""
        project_path = DEMO_ROOT / "pyproject.toml"
        with project_path.open("rb") as project_file:
            project = tomllib.load(project_file)

        self.assertFalse((DEMO_ROOT / "predictor" / "requirements.txt").exists())
        self.assertEqual(
            project["project"]["dependencies"],
            ["numpy==2.5.3", "Pillow==12.3.0"],
        )
        self.assertCountEqual(
            project["project"]["optional-dependencies"]["local"],
            [
                "ipykernel",
                "jupyterlab",
                "matplotlib",
                "pandas>=3.0,<4",
                "nbconvert",
            ],
        )
        self.assertFalse(project["tool"]["uv"]["package"])


if __name__ == "__main__":
    unittest.main()
