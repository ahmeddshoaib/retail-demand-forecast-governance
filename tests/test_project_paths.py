import tempfile
import unittest
from pathlib import Path

from src.project_paths import ProjectPaths


class ProjectPathTests(unittest.TestCase):
    def test_grouped_m5_path_is_preferred_over_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            grouped = root / "data" / "raw" / "m5" / "calendar.csv"
            legacy = root / "data" / "raw" / "calendar.csv"
            grouped.parent.mkdir(parents=True)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            grouped.touch()
            legacy.touch()

            self.assertEqual(ProjectPaths(root).m5_file("calendar.csv"), grouped)

    def test_legacy_path_remains_a_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "data" / "raw" / "calendar.csv"
            legacy.parent.mkdir(parents=True)
            legacy.touch()

            self.assertEqual(ProjectPaths(root).m5_file("calendar.csv"), legacy)


if __name__ == "__main__":
    unittest.main()
