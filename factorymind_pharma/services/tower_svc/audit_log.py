"""
FactoryMind AI (Pharma) - Tamper-Evident Audit Log (§10.5)
Implements an append-only cryptographic hash chain:
  hash_n = sha256(hash_{n-1} || canonical_json(record_n))
Provides verify_chain() to detect tampering and locate the exact corrupted index.
NOTE: This is an engineering integrity control; it does NOT constitute 21 CFR Part 11 compliance.
"""
from __future__ import annotations
import sqlite3
import hashlib
import json
import contextlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    timestamp: str
    event_type: str
    payload: Dict[str, Any]
    prev_hash: str
    record_hash: str


class AuditVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_valid: bool
    total_records: int
    broken_index: Optional[int] = None
    latest_hash: Optional[str] = None
    message: str

    @property
    def first_broken_index(self) -> Optional[int]:
        return self.broken_index

    @property
    def total_entries(self) -> int:
        return self.total_records

    @property
    def chain_head_hash(self) -> str:
        return self.latest_hash or ""


class AuditLog:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parents[3]
            outputs_dir = base_dir.parent / "data" / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = outputs_dir / "audit_chain.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    record_index INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                )
            """)
            conn.commit()

    @staticmethod
    def canonical_json(obj: Any) -> str:
        """Produces deterministic canonical JSON serialization with sorted keys."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    def append(self, event_type: str, payload: Dict[str, Any]) -> AuditEntry:
        """
        Appends a new event payload to the cryptographic hash chain.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_index, record_hash FROM audit_log ORDER BY record_index DESC LIMIT 1")
            last_row = cursor.fetchone()

            if last_row is None:
                new_index = 0
                prev_hash = GENESIS_HASH
            else:
                new_index = last_row["record_index"] + 1
                prev_hash = last_row["record_hash"]

            now_iso = datetime.now(timezone.utc).isoformat()
            canon_payload = self.canonical_json(payload)

            # Compute SHA-256 hash
            to_hash = f"{prev_hash}||{canon_payload}".encode("utf-8")
            record_hash = hashlib.sha256(to_hash).hexdigest()

            cursor.execute("""
                INSERT INTO audit_log (record_index, timestamp, event_type, payload_json, prev_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (new_index, now_iso, event_type, canon_payload, prev_hash, record_hash))
            conn.commit()

            return AuditEntry(
                index=new_index,
                timestamp=now_iso,
                event_type=event_type,
                payload=payload,
                prev_hash=prev_hash,
                record_hash=record_hash,
            )

    def verify_chain(self) -> AuditVerificationResult:
        """
        Recomputes the full hash chain from genesis to head.
        Returns first broken index if tampering is detected.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_index, timestamp, event_type, payload_json, prev_hash, record_hash FROM audit_log ORDER BY record_index ASC")
            rows = cursor.fetchall()

            if not rows:
                return AuditVerificationResult(
                    is_valid=True,
                    total_records=0,
                    broken_index=None,
                    latest_hash=GENESIS_HASH,
                    message="Audit log is empty; chain is valid.",
                )

            expected_prev_hash = GENESIS_HASH
            for row in rows:
                idx = row["record_index"]
                prev_hash = row["prev_hash"]
                stored_hash = row["record_hash"]
                payload_json = row["payload_json"]

                # 1. Verify prev_hash matches the previous record's hash
                if prev_hash != expected_prev_hash:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=len(rows),
                        broken_index=idx,
                        latest_hash=None,
                        message=f"Hash chain broken at index {idx}: expected prev_hash {expected_prev_hash}, found {prev_hash}.",
                    )

                # 2. Recompute expected record_hash
                to_hash = f"{prev_hash}||{payload_json}".encode("utf-8")
                computed_hash = hashlib.sha256(to_hash).hexdigest()

                if stored_hash != computed_hash:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=len(rows),
                        broken_index=idx,
                        latest_hash=None,
                        message=f"Payload tampering detected at index {idx}: expected hash {computed_hash}, found {stored_hash}.",
                    )

                expected_prev_hash = stored_hash

            return AuditVerificationResult(
                is_valid=True,
                total_records=len(rows),
                broken_index=None,
                latest_hash=expected_prev_hash,
                message=f"Audit chain intact across all {len(rows)} records.",
            )

    def tamper_for_test(self, index: int, modified_payload: Dict[str, Any]):
        """Injects a payload mutation into SQLite without updating the hash chain (for testing)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            canon = self.canonical_json(modified_payload)
            cursor.execute("UPDATE audit_log SET payload_json = ? WHERE record_index = ?", (canon, index))
            conn.commit()

    def get_entries(self, limit: int = 50) -> List[AuditEntry]:
        """Returns the latest audit entries."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_index, timestamp, event_type, payload_json, prev_hash, record_hash FROM audit_log ORDER BY record_index DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            entries = []
            for r in rows:
                entries.append(AuditEntry(
                    index=r["record_index"],
                    timestamp=r["timestamp"],
                    event_type=r["event_type"],
                    payload=json.loads(r["payload_json"]),
                    prev_hash=r["prev_hash"],
                    record_hash=r["record_hash"],
                ))
            return entries
