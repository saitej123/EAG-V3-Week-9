"""
Content-addressable artifact store.

Large tool payloads become ``art:<sha256-prefix>`` handles; metadata lives in a sidecar JSON.
Legacy ``art:*.txt`` blobs from earlier iterations are still readable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger

from .paths import STATE as STATE_DIR
from .schemas import ArtifactRecord
ARTIFACTS_DIR = STATE_DIR / "artifacts"


def ensure_state_dirs() -> None:
    """Create state/ and artifacts/ (safe after clean_state wipes disk)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_THRESHOLD_BYTES = 4096


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        ensure_state_dirs()
        self.root = root or ARTIFACTS_DIR

    def _stem(self, artifact_id: str) -> str:
        return artifact_id.replace(":", "_").replace("/", "_")

    def put(
        self,
        blob: bytes,
        *,
        content_type: str = "application/octet-stream",
        source: str = "",
        descriptor: str = "",
    ) -> str:
        digest = hashlib.sha256(blob).hexdigest()
        aid = f"art:{digest[:16]}"
        stem = self._stem(aid)
        bin_path = self.root / f"{stem}.bin"
        meta_path = self.root / f"{stem}.json"
        if bin_path.is_file():
            return aid
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            bin_path.write_bytes(blob)
            meta = ArtifactRecord(
                id=aid,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor,
            )
            meta_path.write_text(json.dumps(meta.model_dump(), indent=2), encoding="utf-8")
        except OSError as e:
            logger.error(f"ArtifactStore.put failed: {e}")
            return ""
        return aid

    def exists(self, artifact_id: str) -> bool:
        if not artifact_id:
            return False
        stem = self._stem(artifact_id)
        bin_path = self.root / f"{stem}.bin"
        if bin_path.is_file():
            return True
        legacy = self.root / f"{artifact_id}.txt"
        return legacy.is_file()

    def get_bytes(self, artifact_id: str) -> bytes:
        stem = self._stem(artifact_id)
        bin_path = self.root / f"{stem}.bin"
        if bin_path.is_file():
            return bin_path.read_bytes()
        legacy = self.root / f"{artifact_id}.txt"
        if legacy.is_file():
            return legacy.read_bytes()
        return b""

    def get_meta(self, artifact_id: str) -> ArtifactRecord | None:
        stem = self._stem(artifact_id)
        meta_path = self.root / f"{stem}.json"
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                return ArtifactRecord.model_validate(data)
            except Exception:
                return None
        if self.exists(artifact_id):
            b = self.get_bytes(artifact_id)
            return ArtifactRecord(
                id=artifact_id,
                content_type="text/plain; charset=utf-8",
                size_bytes=len(b),
                source="legacy_txt",
                descriptor="legacy artifact",
            )
        return None


# Flat export used by agent loop / memory paths
ARTIFACTS_PATH = ARTIFACTS_DIR
