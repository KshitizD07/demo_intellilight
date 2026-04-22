"""
Override Audit Log
===================

Tracks all admin override actions for accountability and analysis.

The web team should integrate this into their FastAPI server to persist
override history.  The RL backend populates it; the dashboard reads it.

Usage:
    from models.override_log import OverrideAuditLog

    log = OverrideAuditLog()
    log.record(override_result)
    recent = log.get_recent(n=10)
    stats = log.get_stats()
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import json
from pathlib import Path

from models.decision_event import OverrideResult


class OverrideAuditLog:
    """
    In-memory override audit log with optional file persistence.

    The web team can extend this with database storage (SQLite, Postgres)
    for production deployment.

    Attributes:
        entries:    List of OverrideResult records
        max_size:   Maximum entries to keep in memory (FIFO)
    """

    def __init__(self, max_size: int = 1000, persist_path: Optional[str] = None):
        """
        Initialize audit log.

        Args:
            max_size:     Max entries in memory
            persist_path: Optional file path for JSON persistence
        """
        self.entries: List[OverrideResult] = []
        self.max_size = max_size
        self.persist_path = persist_path

        # Counters
        self.total_overrides = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def record(self, result: OverrideResult):
        """
        Record an override result.

        Args:
            result: OverrideResult from applying/rejecting an override
        """
        self.entries.append(result)
        self.total_overrides += 1

        if result.accepted:
            self.total_accepted += 1
        else:
            self.total_rejected += 1

        # FIFO eviction
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

        # Persist if configured
        if self.persist_path:
            self._persist()

    def get_recent(self, n: int = 10) -> List[Dict]:
        """
        Get the N most recent override entries.

        Args:
            n: Number of entries to return

        Returns:
            List of override result dicts
        """
        return [e.to_dict() for e in self.entries[-n:]]

    def get_by_junction(self, junction_id: str) -> List[Dict]:
        """
        Get all overrides for a specific junction.

        Args:
            junction_id: Junction ID (e.g. "J1")

        Returns:
            List of override result dicts for that junction
        """
        return [
            e.to_dict() for e in self.entries
            if e.request.junction_id == junction_id
        ]

    def get_by_admin(self, admin_id: str) -> List[Dict]:
        """
        Get all overrides by a specific admin.

        Args:
            admin_id: Admin user ID

        Returns:
            List of override result dicts by that admin
        """
        return [
            e.to_dict() for e in self.entries
            if e.request.admin_id == admin_id
        ]

    def get_stats(self) -> Dict:
        """
        Get aggregate override statistics.

        Returns:
            Dict with total counts, acceptance rate, avg delta, etc.
        """
        if self.total_overrides == 0:
            return {
                "total_overrides": 0,
                "accepted": 0,
                "rejected": 0,
                "acceptance_rate": 0.0,
                "avg_delta": 0.0,
                "per_junction": {},
                "per_admin": {},
            }

        accepted_deltas = [
            e.request.delta for e in self.entries if e.accepted
        ]

        # Per-junction counts
        junction_counts: Dict[str, int] = {}
        for e in self.entries:
            jid = e.request.junction_id
            junction_counts[jid] = junction_counts.get(jid, 0) + 1

        # Per-admin counts
        admin_counts: Dict[str, int] = {}
        for e in self.entries:
            aid = e.request.admin_id
            admin_counts[aid] = admin_counts.get(aid, 0) + 1

        return {
            "total_overrides": self.total_overrides,
            "accepted": self.total_accepted,
            "rejected": self.total_rejected,
            "acceptance_rate": self.total_accepted / self.total_overrides,
            "avg_delta": (
                sum(accepted_deltas) / len(accepted_deltas)
                if accepted_deltas else 0.0
            ),
            "per_junction": junction_counts,
            "per_admin": admin_counts,
        }

    def clear(self):
        """Clear all entries."""
        self.entries.clear()
        self.total_overrides = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def _persist(self):
        """Write current log to JSON file."""
        if not self.persist_path:
            return

        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "stats": self.get_stats(),
                "entries": [e.to_dict() for e in self.entries],
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception:
            pass  # Silent fail — audit is non-critical

    def to_dashboard_view(self, n: int = 20) -> Dict:
        """
        Format for dashboard display.

        Returns the data contract the web team should render in the
        override history panel.

        Args:
            n: Number of recent entries to include

        Returns:
            Dict suitable for JSON serialization to the dashboard
        """
        return {
            "type": "override_history",
            "stats": self.get_stats(),
            "recent": self.get_recent(n),
        }
