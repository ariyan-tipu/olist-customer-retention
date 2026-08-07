"""Publish and fetch the trained model artefact through S3.

The deployed service holds no model of its own. It reads one from a
private bucket at boot, which keeps the container image the same for
every model and makes a rollback a matter of pointing at an earlier key.

Every upload is accompanied by a manifest carrying the SHA-256 of the
object. The service verifies that digest after downloading and refuses
to load anything that fails. The artefact is a joblib pickle, so it
executes code on load; a truncated download or a swapped object should
stop the process instead of reaching `joblib.load`. Write access to the
bucket is held by one deploy role, and the instance role can only read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from olist_retention.logging_utils import get_logger

logger = get_logger(__name__)

BOTO3_MISSING = (
    "boto3 is not installed. Install the optional extra with "
    'pip install -e ".[aws]" to use the S3 artefact store.'
)

MANIFEST_SUFFIX = ".manifest.json"
CHUNK_BYTES = 1024 * 1024


class ArtefactError(RuntimeError):
    """The artefact could not be published or could not be trusted."""


@dataclass(frozen=True)
class PublishedArtefact:
    uri: str
    sha256: str
    size_bytes: int
    model_name: str


def _client(profile: str | None, region: str | None) -> Any:
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(BOTO3_MISSING) from error
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_uri(uri: str) -> tuple[str, str]:
    """Split an s3:// URI into bucket and key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ArtefactError(f"Expected an s3://bucket/key URI, received {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class ArtefactStore:
    """Reads and writes model artefacts in one bucket and prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "models",
        profile: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        # The injection point exists so the tests can drive a mocked S3
        # without a session, and so a caller inside EC2 can hand in a
        # client already built from the instance role.
        self._client = client or _client(profile, region)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def publish(self, model_path: Path, name: str | None = None) -> PublishedArtefact:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ArtefactError(f"No artefact to publish at {model_path}")

        name = name or model_path.name
        digest = sha256_of(model_path)
        size = model_path.stat().st_size

        card_path = model_path.parent / "model_card.json"
        card: dict[str, Any] = {}
        if card_path.is_file():
            card = json.loads(card_path.read_text(encoding="utf-8"))

        key = self._key(name)
        self._client.upload_file(str(model_path), self.bucket, key)
        manifest = {
            "sha256": digest,
            "size_bytes": size,
            "published_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_name": card.get("model_name", "unknown"),
            "trained_at_utc": card.get("trained_at_utc", ""),
            "artefact_version": card.get("artefact_version"),
        }
        self._client.put_object(
            Bucket=self.bucket,
            Key=key + MANIFEST_SUFFIX,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uri = f"s3://{self.bucket}/{key}"
        logger.info("Published %s (%d bytes, sha256 %s)", uri, size, digest[:12])
        return PublishedArtefact(
            uri=uri, sha256=digest, size_bytes=size, model_name=manifest["model_name"]
        )

    def fetch(self, uri: str, destination: Path) -> Path:
        """Download an artefact and check it against its manifest."""
        bucket, key = parse_uri(uri)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            manifest_body = self._client.get_object(Bucket=bucket, Key=key + MANIFEST_SUFFIX)
            manifest = json.loads(manifest_body["Body"].read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - any failure here is fatal
            raise ArtefactError(
                f"No manifest beside {uri}. Publish with 'olist-retention publish' so the "
                "download can be verified before it is loaded."
            ) from error

        temporary = destination.with_suffix(destination.suffix + ".partial")
        self._client.download_file(bucket, key, str(temporary))
        actual = sha256_of(temporary)
        expected = manifest.get("sha256", "")
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise ArtefactError(
                f"Digest mismatch for {uri}: manifest says {expected[:12]}, "
                f"downloaded object is {actual[:12]}. Refusing to load it."
            )
        temporary.replace(destination)
        logger.info(
            "Fetched %s to %s (%s, sha256 verified)",
            uri,
            destination,
            manifest.get("model_name", "unknown model"),
        )
        return destination
