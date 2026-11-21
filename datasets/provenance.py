from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from datasets.schemas import DatasetRecord


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_records(records: Iterable[DatasetRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        encoded = json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()
