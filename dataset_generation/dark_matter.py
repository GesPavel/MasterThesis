import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


DARK_MATTER_PATTERN = re.compile(r"^MSL41_(?P<label>.+?)_ms(?P<suffix>\.(?:pkl|pickle|fasta|fa|faa))$", re.IGNORECASE)


@dataclass(frozen=True)
class DarkMatterGroup:
    label: str
    pkl: Path | None = None
    fasta: Path | None = None
    faa: Path | None = None


def discover_dark_matter_groups(additional_dir: Path) -> List[DarkMatterGroup]:
    if not additional_dir.exists():
        raise FileNotFoundError(f"Missing dark-matter directory: {additional_dir}")

    logger.info("Scanning %s for dark-matter files", additional_dir)
    grouped: Dict[str, Dict[str, Path]] = {}

    for source_path in sorted(additional_dir.iterdir()):
        if not source_path.is_file():
            continue

        match = DARK_MATTER_PATTERN.match(source_path.name)
        if not match:
            continue

        label = match.group("label").lower()
        suffix = source_path.suffix.lower()
        grouped.setdefault(label, {})[suffix] = source_path

    if not grouped:
        raise ValueError(f"No dark-matter files found in {additional_dir}")

    groups: List[DarkMatterGroup] = []
    for label in sorted(grouped):
        files = grouped[label]
        groups.append(
            DarkMatterGroup(
                label=label,
                pkl=files.get(".pkl") or files.get(".pickle"),
                fasta=files.get(".fasta") or files.get(".fa"),
                faa=files.get(".faa"),
            )
        )
    logger.info("Discovered %d dark-matter group(s): %s", len(groups), ", ".join(g.label for g in groups))
    return groups
