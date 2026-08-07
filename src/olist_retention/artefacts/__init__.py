"""Model artefact storage. S3 today; the interface is deliberately small."""

from __future__ import annotations

from olist_retention.artefacts.s3 import (
    ArtefactError,
    ArtefactStore,
    PublishedArtefact,
    parse_uri,
)

__all__ = ["ArtefactError", "ArtefactStore", "PublishedArtefact", "parse_uri"]
