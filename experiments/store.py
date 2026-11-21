from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from experiments.records import ExperimentRecord, RunManifest


class RunStore:
    """SQLite checkpoint store with a unique execution key and transactional claims."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    manifest_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS executions (
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claim_token TEXT,
                    updated_at_utc TEXT NOT NULL,
                    record_json TEXT,
                    PRIMARY KEY (run_id, sample_id, model_tier),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def ensure_run(self, manifest: RunManifest, manifest_hash: str) -> None:
        manifest_json = json.dumps(
            manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT manifest_hash FROM runs WHERE run_id=?", (manifest.run_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO runs(run_id,manifest_hash,manifest_json) VALUES(?,?,?)",
                    (manifest.run_id, manifest_hash, manifest_json),
                )
            elif row["manifest_hash"] != manifest_hash:
                connection.rollback()
                raise ValueError("run ID already exists with a mismatched manifest")
            connection.commit()

    def claim(
        self,
        run_id: str,
        sample_id: str,
        model_tier: str,
        *,
        retry_interrupted: bool = False,
        retry_failed: bool = False,
        rerun_completed: bool = False,
    ) -> str | None:
        token = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM executions WHERE run_id=? AND sample_id=? AND model_tier=?",
                (run_id, sample_id, model_tier),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO executions VALUES(?,?,?,?,?,?,NULL)",
                    (run_id, sample_id, model_tier, "reserved", token, now),
                )
                connection.commit()
                return token
            status = row["status"]
            should_retry = (
                status == "interrupted"
                and retry_interrupted
                or status == "provider_failure"
                and retry_failed
                or status == "success"
                and rerun_completed
                or status == "reserved"
                and retry_interrupted
            )
            if not should_retry:
                connection.rollback()
                return None
            connection.execute(
                "UPDATE executions SET status='reserved',claim_token=?,updated_at_utc=?,record_json=NULL "
                "WHERE run_id=? AND sample_id=? AND model_tier=?",
                (token, now, run_id, sample_id, model_tier),
            )
            connection.commit()
            return token

    def save(self, record: ExperimentRecord, token: str) -> None:
        status = record.execution_status
        content = record.model_dump_json()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET status=?,claim_token=NULL,updated_at_utc=?,record_json=? "
                "WHERE run_id=? AND sample_id=? AND model_tier=? AND claim_token=? AND status='reserved'",
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    content,
                    record.run_id,
                    record.sample_id,
                    record.model_tier,
                    token,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution claim was lost before its record could be saved")

    def list_records(self, run_id: str) -> list[ExperimentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM executions WHERE run_id=? AND record_json IS NOT NULL "
                "ORDER BY sample_id, model_tier",
                (run_id,),
            ).fetchall()
        return [ExperimentRecord.model_validate_json(row["record_json"]) for row in rows]

    def statuses(self, run_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status,COUNT(*) AS count FROM executions WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}
