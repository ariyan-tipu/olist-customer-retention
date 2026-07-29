"""Fetch the Olist dataset from Kaggle and record where it came from.

The dataset is public, so ``kagglehub`` can download it without an account.
Files land in ``data/raw`` together with a small provenance manifest
(sizes, SHA-256 digests, source, licence), which makes any later "works on
my machine" conversation short: compare manifests.

The data itself is never committed to the repository. It is redistributed
by Olist under CC BY-NC-SA 4.0, and a checked-in copy would bloat the repo
for no benefit when a reproducible download exists.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from olist_retention.data.schemas import EXPECTED_FILES
from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "manifest.json"
_DATA_LICENCE = "CC BY-NC-SA 4.0"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DatasetDownloader:
    """Idempotent downloader for the raw Olist CSVs."""

    def __init__(self, kaggle_handle: str, raw_dir: Path) -> None:
        self.kaggle_handle = kaggle_handle
        self.raw_dir = raw_dir

    def missing_files(self) -> list[str]:
        return [name for name in EXPECTED_FILES if not (self.raw_dir / name).is_file()]

    def fetch(self, force: bool = False) -> Path:
        """Download the dataset if needed and return the raw directory."""
        missing = self.missing_files()
        if not missing and not force:
            logger.info("Raw data already present in %s, skipping download", self.raw_dir)
            return self.raw_dir

        # Imported lazily so the rest of the package works offline.
        import kagglehub

        logger.info("Downloading %s via kagglehub", self.kaggle_handle)
        source_dir = Path(kagglehub.dataset_download(self.kaggle_handle))
        logger.info("Kaggle cache directory: %s", source_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for name in EXPECTED_FILES:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"Expected file {name!r} not found in the Kaggle download at {source_dir}"
                )
            shutil.copy2(source, self.raw_dir / name)

        self._write_manifest()
        logger.info("Raw data ready: %d files in %s", len(EXPECTED_FILES), self.raw_dir)
        return self.raw_dir

    def _write_manifest(self) -> None:
        manifest = {
            "dataset": self.kaggle_handle,
            "source": f"https://www.kaggle.com/datasets/{self.kaggle_handle}",
            "licence": _DATA_LICENCE,
            "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": {
                name: {
                    "bytes": (self.raw_dir / name).stat().st_size,
                    "sha256": _sha256(self.raw_dir / name),
                }
                for name in EXPECTED_FILES
            },
        }
        manifest_path = self.raw_dir / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote provenance manifest to %s", manifest_path)
