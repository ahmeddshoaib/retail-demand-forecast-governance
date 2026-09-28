"""Portable project and data paths for local, GitHub and Google Colab runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_FOLDER_NAME = "MSc_Dissertation_Forecasting_Final"


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository without depending on one machine-specific path."""
    configured = os.environ.get("MSC_SUPPLY_CHAIN_ROOT")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    if start is not None:
        start_path = Path(start).expanduser().resolve()
        candidates.extend([start_path, *start_path.parents])

    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    candidates.append(Path("/content/drive/MyDrive") / PROJECT_FOLDER_NAME)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate

    searched = "\n".join(f"  - {path}" for path in seen)
    raise FileNotFoundError(
        "Could not locate the dissertation project. Set MSC_SUPPLY_CHAIN_ROOT "
        f"or place the project at /content/drive/MyDrive/{PROJECT_FOLDER_NAME}.\n"
        f"Searched:\n{searched}"
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project locations with legacy raw-file compatibility."""

    root: Path

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        return cls(find_project_root(start))

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def metrics(self) -> Path:
        return self.outputs / "metrics"

    @property
    def powerbi(self) -> Path:
        return self.outputs / "powerbi"

    def m5_file(self, filename: str) -> Path:
        return self._resolve_raw(filename, "m5")

    def _resolve_raw(self, filename: str, group: str) -> Path:
        """Prefer the neat grouped layout but accept the existing flat layout."""
        grouped = self.raw / group / filename
        legacy = self.raw / filename
        if grouped.exists():
            return grouped
        if legacy.exists():
            return legacy
        return grouped

    def ensure_output_directories(self) -> None:
        for directory in (self.processed, self.metrics, self.powerbi):
            directory.mkdir(parents=True, exist_ok=True)
