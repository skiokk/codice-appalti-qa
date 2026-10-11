"""``refined.json``: what a refine established about the snapshot files of a directory.

Written by ``refine_snapshots`` and read by it and by ``download_snapshots``, so that intervals
already searched are not probed or downloaded again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "refined.json"


@dataclass(frozen=True, slots=True)
class Manifest:
    """What a previous refine established about the files in ``raw_dir``."""

    verified_through: date  # last day whose content is known
    boundaries: list[date]  # first day of each distinct content up to verified_through
    unresolved: list[tuple[date, date]] = field(default_factory=list)  # gaps left by the cap

    def is_exact(self, day: date) -> bool:
        """True when ``day`` is a boundary whose previous day is known to differ."""
        return day in self.boundaries and all(day != hi for _, hi in self.unresolved)


def read_manifest(raw_dir: Path) -> Manifest | None:
    path = Path(raw_dir) / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(
            verified_through=date.fromisoformat(data["verified_through"]),
            boundaries=[date.fromisoformat(day) for day in data["boundaries"]],
            unresolved=[
                (date.fromisoformat(lo), date.fromisoformat(hi)) for lo, hi in data["unresolved"]
            ],
        )
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("ignoring malformed %s: %s", path, exc)
        return None


def write_manifest(raw_dir: Path, manifest: Manifest) -> None:
    data = {
        "verified_through": manifest.verified_through.isoformat(),
        "boundaries": [day.isoformat() for day in manifest.boundaries],
        "unresolved": [[lo.isoformat(), hi.isoformat()] for lo, hi in manifest.unresolved],
    }
    path = Path(raw_dir) / MANIFEST_NAME
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
