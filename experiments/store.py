"""SQLite checkpointing with leases, fencing, and explicit uncertain outcomes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
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
                    claim_prior_status TEXT,
                    claim_prior_record_json TEXT,
                    PRIMARY KEY (run_id, sample_id, model_tier),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS request_reservations (
                    claim_token TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    reserved_requests INTEGER NOT NULL CHECK(reserved_requests > 0),
                    status TEXT NOT NULL CHECK(status IN ('held','consumed')),
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS budget_reservations (
                    claim_token TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    reserved_micros INTEGER NOT NULL CHECK(reserved_micros >= 0),
                    exposure_micros INTEGER NOT NULL CHECK(exposure_micros >= 0),
                    observed_micros INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('held','uncertain','observed','released')),
                    pricing_provenance_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS recovery_authorizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    model_tier TEXT NOT NULL,
                    prior_status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    claim_token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    approval_id TEXT,
                    uncertain_reissue_approval_id TEXT,
                    reserved_requests INTEGER NOT NULL,
                    git_commit_sha TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'authorized_claimed',
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS request_budget_migrations (
                    run_id TEXT PRIMARY KEY,
                    migrated_at_utc TEXT NOT NULL,
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
            "claim_prior_status": "ALTER TABLE executions ADD COLUMN claim_prior_status TEXT",
            "claim_prior_record_json": "ALTER TABLE executions ADD COLUMN claim_prior_record_json TEXT",
        }
        for name, statement in migrations.items():
            if name not in columns:
                connection.execute(statement)
        audit_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(recovery_authorizations)")
        }
        if "state" not in audit_columns:
            connection.execute(
                "ALTER TABLE recovery_authorizations ADD COLUMN state TEXT NOT NULL "
                "DEFAULT 'authorized_claimed'"
            )
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
                "SELECT manifest_hash,manifest_json FROM runs WHERE run_id=?", (manifest.run_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO runs(run_id,manifest_hash,manifest_json) VALUES(?,?,?)",
                    (manifest.run_id, manifest_hash, manifest_json),
                )
            elif row["manifest_hash"] != manifest_hash:
                previous = json.loads(row["manifest_json"])
                current = manifest.model_dump(mode="json")
                if (
                    manifest_hash != _legacy_manifest_hash_without_commit(current)
                    or row["manifest_hash"]
                    not in {
                        _legacy_manifest_hash(previous),
                        _legacy_manifest_hash_without_commit(previous),
                    }
                    or _manifest_compatibility_hash(previous)
                    != _manifest_compatibility_hash(current)
                ):
                    connection.rollback()
                    raise ValueError("run ID already exists with a mismatched manifest")
                connection.execute(
                    "UPDATE runs SET manifest_hash=? WHERE run_id=?",
                    (manifest_hash, manifest.run_id),
                )
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
        allow_paid_retries: bool = False,
        paid_retry_approval_id: str | None = None,
        provider_is_paid: bool = False,
        request_attempt_limit: int = 1,
        git_commit_sha: str = "unavailable",
    ) -> ExecutionClaim | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if request_attempt_limit <= 0:
            raise ValueError("request_attempt_limit must be positive")
        if provider_is_paid and allow_uncertain_reissue and not uncertain_reissue_approval_id:
            raise ValueError("uncertain paid reissue requires an approval reference")
        if (
            provider_is_paid
            and (retry_failed or rerun_completed)
            and (not allow_paid_retries or not paid_retry_approval_id)
        ):
            raise ValueError("paid retry or rerun requires explicit authorization and approval")
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
                status = row["claim_prior_status"]
                restored_record = row["claim_prior_record_json"]
                if status is None:
                    # Older checkpoints did not save the claimed-from status.
                    # Their record_json still holds a terminal record on retries.
                    legacy_record = (
                        ExperimentRecord.model_validate_json(row["record_json"])
                        if row["record_json"]
                        else None
                    )
                    if legacy_record and legacy_record.execution_status in {
                        "success",
                        "provider_failure",
                        "interrupted",
                        "outcome_unknown",
                    }:
                        status = legacy_record.execution_status
                        restored_record = row["record_json"]
                    else:
                        status = "pending"
                connection.execute(
                    "UPDATE recovery_authorizations SET state='lease_expired_before_invoke' "
                    "WHERE claim_token=? AND state='authorized_claimed'",
                    (row["claim_token"],),
                )
                connection.execute(
                    "DELETE FROM request_reservations WHERE claim_token=? AND status='held'",
                    (row["claim_token"],),
                )
                connection.execute(
                    "UPDATE budget_reservations SET status='released',exposure_micros=0,"
                    "updated_at_utc=? WHERE claim_token=? AND status='held'",
                    (now_iso, row["claim_token"]),
                )
                connection.execute(
                    "UPDATE executions SET status=?,record_json=?,claim_token=NULL,owner_id=NULL,"
                    "lease_expires_at=NULL,claim_prior_status=NULL,claim_prior_record_json=NULL "
                    "WHERE run_id=? AND sample_id=? AND model_tier=?",
                    (status, restored_record, run_id, sample_id, model_tier),
                )
                row = connection.execute(
                    "SELECT * FROM executions WHERE run_id=? AND sample_id=? AND model_tier=?",
                    (run_id, sample_id, model_tier),
                ).fetchone()
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

            run = connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if run is None:
                connection.rollback()
                raise KeyError("run manifest was not initialized")
            max_requests = int(json.loads(run["manifest_json"])["max_requests"])
            self._migrate_request_budget_history(connection, run_id, run["manifest_json"], now_iso)
            used_requests = int(
                connection.execute(
                    "SELECT COALESCE(SUM(reserved_requests),0) FROM request_reservations "
                    "WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            if used_requests + request_attempt_limit > max_requests:
                connection.commit()
                return None

            fence = int(row["fencing_token"] or 0) + 1
            connection.execute(
                "UPDATE executions SET status='claimed',claim_token=?,owner_id=?,lease_expires_at=?,"
                "fencing_token=?,updated_at_utc=?,claim_prior_status=?,claim_prior_record_json=? "
                "WHERE run_id=? AND sample_id=? AND model_tier=?",
                (
                    token,
                    owner_id,
                    expires,
                    fence,
                    now_iso,
                    status,
                    prior_uncertain.model_dump_json() if prior_uncertain is not None else None,
                    run_id,
                    sample_id,
                    model_tier,
                ),
            )
            connection.execute(
                "INSERT INTO request_reservations "
                "(claim_token,run_id,sample_id,model_tier,reserved_requests,status,created_at_utc) "
                "VALUES(?,?,?,?,?,'held',?)",
                (
                    token,
                    run_id,
                    sample_id,
                    model_tier,
                    request_attempt_limit,
                    now_iso,
                ),
            )
            if provider_is_paid and status in {"provider_failure", "uncertain", "success"}:
                action = {
                    "provider_failure": "retry_failed",
                    "uncertain": "reissue_uncertain",
                    "success": "rerun_completed",
                }[status]
                connection.execute(
                    "INSERT INTO recovery_authorizations "
                    "(run_id,sample_id,model_tier,prior_status,action,owner_id,claim_token,"
                    "fencing_token,approval_id,uncertain_reissue_approval_id,reserved_requests,"
                    "git_commit_sha,created_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        sample_id,
                        model_tier,
                        status,
                        action,
                        owner_id,
                        token,
                        fence,
                        paid_retry_approval_id,
                        uncertain_reissue_approval_id,
                        request_attempt_limit,
                        git_commit_sha,
                        now_iso,
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
            connection.execute("BEGIN IMMEDIATE")
            active_claim = connection.execute(
                "SELECT 1 FROM executions WHERE run_id=? AND sample_id=? AND model_tier=? "
                "AND claim_token=? AND owner_id=? AND fencing_token=? AND status='claimed' "
                "AND lease_expires_at>?",
                (
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    now,
                ),
            ).fetchone()
            if active_claim is None:
                connection.rollback()
                raise RuntimeError("execution claim was lost before provider invocation")
            reservation = connection.execute(
                "SELECT status FROM request_reservations WHERE claim_token=? AND run_id=?",
                (claim.token, claim.run_id),
            ).fetchone()
            if reservation is None or reservation["status"] != "held":
                connection.rollback()
                raise RuntimeError("request budget reservation was lost before invocation")
            if record.execution_mode == "real":
                monetary = connection.execute(
                    "SELECT status FROM budget_reservations WHERE claim_token=? AND run_id=?",
                    (claim.token, claim.run_id),
                ).fetchone()
                if monetary is None or monetary["status"] != "held":
                    connection.rollback()
                    raise RuntimeError(
                        "monetary budget reservation is required before paid invocation"
                    )
            cursor = connection.execute(
                "UPDATE executions SET status='in_flight',record_json=?,request_metadata_json=?,"
                "updated_at_utc=?,claim_prior_status=NULL,claim_prior_record_json=NULL "
                "WHERE run_id=? AND sample_id=? AND model_tier=? "
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
                connection.rollback()
                raise RuntimeError("execution claim was lost before provider invocation")
            connection.execute(
                "UPDATE request_reservations SET status='consumed' WHERE claim_token=?",
                (claim.token,),
            )
            connection.execute(
                "UPDATE budget_reservations SET status='uncertain',updated_at_utc=? "
                "WHERE claim_token=? AND status='held'",
                (datetime.fromtimestamp(now, UTC).isoformat(), claim.token),
            )
            connection.execute(
                "UPDATE recovery_authorizations SET state='invocation_marked' "
                "WHERE claim_token=? AND state='authorized_claimed'",
                (claim.token,),
            )
            connection.commit()

    def release_unstarted(self, claim: ExecutionClaim, *, now: float) -> None:
        prior_record_json = (
            claim.prior_uncertain_record.model_dump_json()
            if claim.prior_uncertain_record is not None
            else None
        )
        prior_status = claim.prior_status
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE executions SET status=?,claim_token=NULL,owner_id=NULL,"
                "lease_expires_at=NULL,record_json=?,updated_at_utc=?,claim_prior_status=NULL,"
                "claim_prior_record_json=NULL WHERE run_id=? "
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
            if cursor.rowcount == 1:
                connection.execute(
                    "DELETE FROM request_reservations WHERE claim_token=? AND status='held'",
                    (claim.token,),
                )
                connection.execute(
                    "UPDATE budget_reservations SET status='released',exposure_micros=0,"
                    "updated_at_utc=? WHERE claim_token=? AND status='held'",
                    (datetime.now(UTC).isoformat(), claim.token),
                )
                connection.execute(
                    "UPDATE recovery_authorizations SET state='not_invoked' "
                    "WHERE claim_token=? AND state='authorized_claimed'",
                    (claim.token,),
                )
                connection.commit()
            else:
                connection.rollback()

    def mark_uncertain(self, claim: ExecutionClaim, *, now: float) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                connection.rollback()
                raise RuntimeError("execution claim was lost while marking uncertain outcome")
            connection.execute(
                "UPDATE recovery_authorizations SET state='outcome_unknown' "
                "WHERE claim_token=? AND state='invocation_marked'",
                (claim.token,),
            )
            connection.commit()

    def save(self, record: ExperimentRecord, claim: ExecutionClaim) -> None:
        status = record.execution_status
        content = record.model_dump_json()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                connection.rollback()
                raise RuntimeError("stale execution claim rejected by fencing token")
            self._reconcile_budget_row(connection, claim.token, record)
            connection.execute(
                "UPDATE recovery_authorizations SET state=? "
                "WHERE claim_token=? AND state='invocation_marked'",
                (status, claim.token),
            )
            connection.commit()

    def save_uncertain(self, record: ExperimentRecord, claim: ExecutionClaim) -> None:
        """Persist a returned-but-nondefinitive provider outcome for explicit recovery."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                connection.rollback()
                raise RuntimeError("stale execution claim rejected while saving uncertainty")
            connection.execute(
                "UPDATE budget_reservations SET status='uncertain',updated_at_utc=? "
                "WHERE claim_token=? AND status IN ('held','uncertain')",
                (datetime.now(UTC).isoformat(), claim.token),
            )
            connection.execute(
                "UPDATE recovery_authorizations SET state='outcome_unknown' "
                "WHERE claim_token=? AND state='invocation_marked'",
                (claim.token,),
            )
            connection.commit()

    def reserve_budget(
        self,
        claim: ExecutionClaim,
        amount_micros: int,
        *,
        pricing_provenance: dict,
        now: float,
    ) -> None:
        """Atomically reserve exposure against the immutable manifest run cap.

        The caller must reserve a defensible maximum for the complete invocation
        envelope. This transaction enforces local arithmetic; it does not prove
        that provider billing itself is bounded by that amount.
        """
        if amount_micros < 0:
            raise ValueError("budget reservation must be non-negative")
        now_iso = datetime.fromtimestamp(now, UTC).isoformat()
        provenance_json = json.dumps(pricing_provenance, sort_keys=True, separators=(",", ":"))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT 1 FROM executions WHERE run_id=? AND sample_id=? AND model_tier=? "
                "AND claim_token=? AND owner_id=? AND fencing_token=? AND status='claimed' "
                "AND lease_expires_at>?",
                (
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    claim.token,
                    claim.owner_id,
                    claim.fencing_token,
                    now,
                ),
            ).fetchone()
            if active is None:
                connection.rollback()
                raise RuntimeError("stale execution claim cannot reserve budget")
            existing = connection.execute(
                "SELECT reserved_micros,pricing_provenance_json FROM budget_reservations "
                "WHERE claim_token=?",
                (claim.token,),
            ).fetchone()
            if existing:
                if (existing["reserved_micros"], existing["pricing_provenance_json"]) != (
                    amount_micros,
                    provenance_json,
                ):
                    connection.rollback()
                    raise ValueError("budget reservation cannot be changed for an existing claim")
                connection.commit()
                return
            run = connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id=?", (claim.run_id,)
            ).fetchone()
            manifest = json.loads(run["manifest_json"])
            request_cap = _usd_to_micros(manifest.get("budget_limit_usd"))
            run_cap = _usd_to_micros(manifest.get("max_run_budget_usd"))
            if request_cap is None or run_cap is None:
                connection.rollback()
                raise ValueError("budget policy is missing from the immutable run manifest")
            if amount_micros > request_cap:
                connection.rollback()
                raise ValueError("reservation exceeds the per-request budget")
            exposure = int(
                connection.execute(
                    "SELECT COALESCE(SUM(exposure_micros),0) FROM budget_reservations "
                    "WHERE run_id=? AND status!='released'",
                    (claim.run_id,),
                ).fetchone()[0]
            )
            if exposure + amount_micros > run_cap:
                connection.rollback()
                raise ValueError("reservation would exceed the cumulative run budget")
            connection.execute(
                "INSERT INTO budget_reservations(claim_token,run_id,sample_id,model_tier,"
                "reserved_micros,exposure_micros,status,pricing_provenance_json,created_at_utc,"
                "updated_at_utc) VALUES(?,?,?,?,?,?,'held',?,?,?)",
                (
                    claim.token,
                    claim.run_id,
                    claim.sample_id,
                    claim.model_tier,
                    amount_micros,
                    amount_micros,
                    provenance_json,
                    now_iso,
                    now_iso,
                ),
            )
            connection.commit()

    def budget_reservations(self, run_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT claim_token,run_id,sample_id,model_tier,reserved_micros,exposure_micros,"
                "observed_micros,status,pricing_provenance_json,created_at_utc,updated_at_utc "
                "FROM budget_reservations WHERE run_id=? ORDER BY created_at_utc,claim_token",
                (run_id,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["pricing_provenance"] = json.loads(item.pop("pricing_provenance_json"))
            results.append(item)
        return results

    @staticmethod
    def _reconcile_budget_row(
        connection: sqlite3.Connection, claim_token: str, record: ExperimentRecord
    ) -> None:
        row = connection.execute(
            "SELECT reserved_micros,exposure_micros FROM budget_reservations "
            "WHERE claim_token=? AND status IN ('uncertain','held')",
            (claim_token,),
        ).fetchone()
        if row is None:
            return
        known = record.cost.known_calculated_usd
        if known is None:
            connection.execute(
                "UPDATE budget_reservations SET status='uncertain',updated_at_utc=? "
                "WHERE claim_token=?",
                (datetime.now(UTC).isoformat(), claim_token),
            )
            return
        observed = _usd_to_micros(known)
        # Usage-derived cost is an observation, not a provider invoice. Keep at
        # least the original reservation until independent billing reconciliation.
        exposure = max(int(row["exposure_micros"]), observed or 0)
        connection.execute(
            "UPDATE budget_reservations SET status='observed',observed_micros=?,"
            "exposure_micros=?,updated_at_utc=? WHERE claim_token=?",
            (observed, exposure, datetime.now(UTC).isoformat(), claim_token),
        )

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

    def recovery_audit(self, run_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id,sample_id,model_tier,prior_status,action,owner_id,claim_token,"
                "fencing_token,approval_id,uncertain_reissue_approval_id,reserved_requests,"
                "git_commit_sha,state,"
                "created_at_utc FROM recovery_authorizations WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def request_reservations(self, run_id: str) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT claim_token,run_id,sample_id,model_tier,reserved_requests,status,"
                "created_at_utc FROM request_reservations WHERE run_id=? ORDER BY created_at_utc,claim_token",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stored_manifest(self, run_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError("run manifest was not initialized")
        return json.loads(row["manifest_json"])

    @staticmethod
    def _migrate_request_budget_history(
        connection: sqlite3.Connection, run_id: str, manifest_json: str, now_iso: str
    ) -> None:
        migrated = connection.execute(
            "SELECT 1 FROM request_budget_migrations WHERE run_id=?", (run_id,)
        ).fetchone()
        if migrated:
            return
        manifest = json.loads(manifest_json)
        fallback_count = 1 if manifest.get("fallback_policy") == "operational" else 0
        attempt_envelope = int(manifest.get("max_retries", 0)) + 1 + fallback_count
        rows = connection.execute(
            "SELECT sample_id,model_tier,status,record_json,request_metadata_json,"
            "legacy_unknown_invocation FROM executions WHERE run_id=?",
            (run_id,),
        ).fetchall()
        for row in rows:
            if row["status"] in {"pending", "claimed"}:
                continue
            record = (
                ExperimentRecord.model_validate_json(row["record_json"])
                if row["record_json"]
                else None
            )
            if row["status"] == "uncertain" or row["legacy_unknown_invocation"]:
                reserved = attempt_envelope
            elif record is not None and record.attempts:
                reserved = len(record.attempts)
            elif row["request_metadata_json"] or row["status"] in {
                "success",
                "provider_failure",
                "interrupted",
            }:
                reserved = 1
            else:
                continue
            synthetic_token = (
                "legacy-"
                + hashlib.sha256(
                    f"{run_id}\0{row['sample_id']}\0{row['model_tier']}".encode()
                ).hexdigest()
            )
            connection.execute(
                "INSERT OR IGNORE INTO request_reservations "
                "(claim_token,run_id,sample_id,model_tier,reserved_requests,status,created_at_utc) "
                "VALUES(?,?,?,?,?,'consumed',?)",
                (
                    synthetic_token,
                    run_id,
                    row["sample_id"],
                    row["model_tier"],
                    reserved,
                    now_iso,
                ),
            )
        connection.execute(
            "INSERT INTO request_budget_migrations(run_id,migrated_at_utc) VALUES(?,?)",
            (run_id, now_iso),
        )


def _manifest_compatibility_hash(material: dict) -> str:
    compatible = dict(material)
    compatible.pop("created_at_utc", None)
    # Recovery grants affect operations, not the frozen experiment protocol.
    compatible.pop("allow_paid_retries", None)
    compatible.pop("paid_retry_approval_id", None)
    compatible.pop("git_commit_sha", None)
    # Normalize fields added with plan binding so pre-plan Phase 2 manifests
    # remain compatible with their explicit legacy selection semantics.
    compatible.setdefault("source_dataset_content_sha256", None)
    compatible.setdefault("collection_plan_schema_version", None)
    compatible.setdefault("collection_plan_sha256", None)
    compatible.setdefault("collection_plan_sample_ids", [])
    compatible.setdefault("collection_plan_sample_ids_sha256", None)
    compatible.setdefault("collection_plan_split", None)
    compatible.setdefault("selection_policy", "legacy_sorted_prefix_v1")
    compatible.setdefault("max_run_budget_usd", None)
    compatible.setdefault("approval_artifact_sha256", None)
    compatible.setdefault("approval_config_sha256", None)
    return hashlib.sha256(
        json.dumps(compatible, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _usd_to_micros(value) -> int | None:
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("USD budget must be a decimal value") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD budget must be finite and non-negative")
    micros = amount * Decimal(1_000_000)
    if micros != micros.to_integral_value():
        raise ValueError("USD amounts must be expressible in integer micro-USD")
    return int(micros)


def _legacy_manifest_hash(material: dict) -> str:
    legacy = dict(material)
    legacy.pop("created_at_utc", None)
    return hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _legacy_manifest_hash_without_commit(material: dict) -> str:
    legacy = dict(material)
    legacy.pop("created_at_utc", None)
    legacy.pop("git_commit_sha", None)
    return hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
