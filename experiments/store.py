"""SQLite checkpointing with leases, fencing, and explicit uncertain outcomes."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from experiments.records import ExperimentRecord, RunManifest


@dataclass(frozen=True)
class ExecutionClaim:
    run_id: str
    sample_id: str
    model_tier: str
    owner_id: str
    token: str
    fencing_token: int
    lease_expires_at: float
    prior_status: str
    prior_uncertain_record: ExperimentRecord | None = None
    legacy_unknown_invocation: bool = False


class RunStore:
    """Durable task state; external provider calls are at-least-once at best.

    A `claimed` lease is known not to have started provider invocation and may be
    reclaimed after expiry. An `in_flight` lease that expires becomes `uncertain`:
    the provider may have processed and billed it. Reissuing an uncertain real
    request requires an explicit approval reference and always preserves the
    possible prior attempt. Fencing tokens reject stale writers.
    """

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
                    owner_id TEXT,
                    lease_expires_at REAL,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    request_metadata_json TEXT,
                legacy_unknown_invocation INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, sample_id, model_tier),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                """
            )
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(executions)")}
        migrations = {
            "owner_id": "ALTER TABLE executions ADD COLUMN owner_id TEXT",
            "lease_expires_at": "ALTER TABLE executions ADD COLUMN lease_expires_at REAL",
            "fencing_token": "ALTER TABLE executions ADD COLUMN fencing_token INTEGER NOT NULL DEFAULT 0",
            "request_metadata_json": "ALTER TABLE executions ADD COLUMN request_metadata_json TEXT",
            "legacy_unknown_invocation": "ALTER TABLE executions ADD COLUMN legacy_unknown_invocation INTEGER NOT NULL DEFAULT 0",
        }
        for name, statement in migrations.items():
            if name not in columns:
                connection.execute(statement)
        # A legacy reservation had no invocation marker; conservatively classify it
        # as uncertain because it may have crossed the provider boundary.
        connection.execute(
            "UPDATE executions SET status='uncertain',claim_token=NULL,owner_id=NULL,"
            "lease_expires_at=NULL,legacy_unknown_invocation=1 WHERE status='reserved'"
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

    def ensure_tasks(self, run_id: str, tasks: Sequence[tuple[str, str]]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT OR IGNORE INTO executions "
                "(run_id,sample_id,model_tier,status,updated_at_utc) VALUES(?,?,?,'pending',?)",
                [(run_id, sample_id, tier, now) for sample_id, tier in tasks],
            )
            connection.commit()

    def claim(
        self,
        run_id: str,
        sample_id: str,
        model_tier: str,
        *,
        owner_id: str,
        lease_seconds: float,
        now: float,
        retry_failed: bool = False,
        rerun_completed: bool = False,
        retry_interrupted: bool = False,
        allow_uncertain_reissue: bool = False,
        uncertain_reissue_approval_id: str | None = None,
        paid_retry_approval_id: str | None = None,
        provider_is_paid: bool = False,
    ) -> ExecutionClaim | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if provider_is_paid and allow_uncertain_reissue and not uncertain_reissue_approval_id:
            raise ValueError("uncertain paid reissue requires an approval reference")
        if provider_is_paid and (retry_failed or rerun_completed) and not paid_retry_approval_id:
            raise ValueError("paid retry or rerun requires an approval reference")
        token = str(uuid.uuid4())
        expires = now + lease_seconds
        now_iso = datetime.fromtimestamp(now, UTC).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM executions WHERE run_id=? AND sample_id=? AND model_tier=?",
                (run_id, sample_id, model_tier),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("task was not initialized")

            status = row["status"]
            if status == "claimed" and (row["lease_expires_at"] or 0) <= now:
                status = "pending"  # provider call is known not to have started
                connection.execute(
                    "UPDATE executions SET status='pending',claim_token=NULL,owner_id=NULL,"
                    "lease_expires_at=NULL WHERE run_id=? AND sample_id=? AND model_tier=?",
                    (run_id, sample_id, model_tier),
                )
            elif status == "in_flight" and (row["lease_expires_at"] or 0) <= now:
                status = "uncertain"
                connection.execute(
                    "UPDATE executions SET status='uncertain',claim_token=NULL,owner_id=NULL,"
                    "lease_expires_at=NULL WHERE run_id=? AND sample_id=? AND model_tier=?",
                    (run_id, sample_id, model_tier),
                )

            prior_uncertain = None
            legacy_unknown = False
            should_claim = status == "pending"
            if status == "uncertain":
                if not allow_uncertain_reissue:
                    connection.commit()
                    return None
                prior_uncertain = (
                    ExperimentRecord.model_validate_json(row["record_json"])
                    if row["record_json"]
                    else None
                )
                legacy_unknown = bool(row["legacy_unknown_invocation"])
                should_claim = True
            if status == "provider_failure":
                should_claim = retry_failed
                if should_claim and row["record_json"]:
                    prior_uncertain = ExperimentRecord.model_validate_json(row["record_json"])
            if status == "success":
                should_claim = rerun_completed
                if should_claim and row["record_json"]:
                    prior_uncertain = ExperimentRecord.model_validate_json(row["record_json"])
            if status == "interrupted":
                should_claim = retry_interrupted and not provider_is_paid
                if should_claim and row["record_json"]:
                    prior_uncertain = ExperimentRecord.model_validate_json(row["record_json"])
            if status in {"claimed", "in_flight"}:
                should_claim = False
            if not should_claim:
                connection.commit()
                return None

            fence = int(row["fencing_token"] or 0) + 1
            connection.execute(
                "UPDATE executions SET status='claimed',claim_token=?,owner_id=?,lease_expires_at=?,"
                "fencing_token=?,updated_at_utc=? WHERE run_id=? AND sample_id=? AND model_tier=?",
                (
                    token,
                    owner_id,
                    expires,
                    fence,
                    now_iso,
                    run_id,
                    sample_id,
                    model_tier,
                ),
            )
            connection.commit()
            return ExecutionClaim(
                run_id,
                sample_id,
                model_tier,
                owner_id,
                token,
                fence,
                expires,
                status,
                prior_uncertain,
                legacy_unknown,
            )

    def renew(self, claim: ExecutionClaim, *, now: float, lease_seconds: float) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET lease_expires_at=?,updated_at_utc=? "
                "WHERE run_id=? AND sample_id=? AND model_tier=? AND claim_token=? "
                "AND owner_id=? AND fencing_token=? AND lease_expires_at>? "
                "AND status IN ('claimed','in_flight')",
                (
                    now + lease_seconds,
                    datetime.fromtimestamp(now, UTC).isoformat(),
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def mark_invoked(
        self,
        claim: ExecutionClaim,
        record: ExperimentRecord,
        request_metadata: dict,
        *,
        now: float,
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET status='in_flight',record_json=?,request_metadata_json=?,"
                "updated_at_utc=? WHERE run_id=? AND sample_id=? AND model_tier=? "
                "AND claim_token=? AND owner_id=? AND fencing_token=? AND status='claimed' "
                "AND lease_expires_at>?",
                (
                    record.model_dump_json(),
                    json.dumps(request_metadata, sort_keys=True),
                    datetime.fromtimestamp(now, UTC).isoformat(),
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution claim was lost before provider invocation")

    def release_unstarted(self, claim: ExecutionClaim, *, now: float) -> None:
        prior_record_json = (
            claim.prior_uncertain_record.model_dump_json()
            if claim.prior_uncertain_record is not None
            else None
        )
        prior_status = claim.prior_status
        with self._connection() as connection:
            connection.execute(
                "UPDATE executions SET status=?,claim_token=NULL,owner_id=NULL,"
                "lease_expires_at=NULL,record_json=?,updated_at_utc=? WHERE run_id=? "
                "AND sample_id=? AND model_tier=? AND claim_token=? AND owner_id=? "
                "AND fencing_token=? AND status='claimed'",
                (
                    prior_status,
                    prior_record_json,
                    datetime.fromtimestamp(now, UTC).isoformat(),
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                ),
            )

    def mark_uncertain(self, claim: ExecutionClaim, *, now: float) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET status='uncertain',claim_token=NULL,owner_id=NULL,"
                "lease_expires_at=NULL,updated_at_utc=? WHERE run_id=? AND sample_id=? "
                "AND model_tier=? AND claim_token=? AND owner_id=? AND fencing_token=? "
                "AND status='in_flight'",
                (
                    datetime.fromtimestamp(now, UTC).isoformat(),
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("execution claim was lost while marking uncertain outcome")

    def save(self, record: ExperimentRecord, claim: ExecutionClaim) -> None:
        status = record.execution_status
        content = record.model_dump_json()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET status=?,claim_token=NULL,owner_id=NULL,lease_expires_at=NULL,"
                "updated_at_utc=?,record_json=? WHERE run_id=? AND sample_id=? AND model_tier=? "
                "AND claim_token=? AND owner_id=? AND fencing_token=? AND status='in_flight' "
                "AND lease_expires_at>?",
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    content,
                    record.run_id,
                    record.sample_id,
                    record.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    time.time(),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stale execution claim rejected by fencing token")

    def save_uncertain(self, record: ExperimentRecord, claim: ExecutionClaim) -> None:
        """Persist a returned-but-nondefinitive provider outcome for explicit recovery."""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE executions SET status='uncertain',claim_token=NULL,owner_id=NULL,"
                "lease_expires_at=NULL,updated_at_utc=?,record_json=? WHERE run_id=? "
                "AND sample_id=? AND model_tier=? AND claim_token=? AND owner_id=? "
                "AND fencing_token=? AND status='in_flight' AND lease_expires_at>?",
                (
                    datetime.now(UTC).isoformat(),
                    record.model_dump_json(),
                    record.run_id,
                    record.sample_id,
                    record.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    time.time(),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stale execution claim rejected while saving uncertainty")

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

    def execution_state(self, run_id: str, sample_id: str, model_tier: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM executions WHERE run_id=? AND sample_id=? AND model_tier=?",
                (run_id, sample_id, model_tier),
            ).fetchone()
        if row is None:
            raise KeyError("task was not initialized")
        return str(row["status"])
