"""
Web-Integration Data Models
=============================

Pure Python data classes defining the interface contract between the
RL backend (PerPhaseCorridorEnv) and a future web frontend.

These models are consumed by:
  1. The PerPhaseCorridorEnv (emits DecisionEvent after each RL decision)
  2. A future FastAPI/WebSocket server (broadcasts events to admin panel)
  3. The admin panel (sends OverrideRequest back to the backend)
  4. The GreenWaveController (activates corridor coordination)

NO web server code lives here — only data structures and validation.

Usage by the web team:
    from models.decision_event import (
        DecisionEvent, OverrideRequest, OverrideResult,
        GreenWaveRequest, StateSnapshot, SystemMode
    )
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List


# ── System Modes ─────────────────────────────────────────────────────────

class SystemMode(str, Enum):
    """Operating mode of the IntelliLight system."""
    TRAINING = "TRAINING"       # Pure RL, no overrides, no web
    DEMO = "DEMO"               # RL + override window + dashboard
    DEPLOYMENT = "DEPLOYMENT"   # RL + override + full audit logging


# ── Phase Names ──────────────────────────────────────────────────────────

PHASE_NAMES = {
    0: "EW-Through",
    1: "EW-Left",
    2: "NS-Through",
    3: "NS-Left",
}


# ── Decision Event ────────────────────────────────────────────────────────

@dataclass
class DecisionEvent:
    """
    Emitted by PerPhaseCorridorEnv after each RL decision.

    The web layer should broadcast this over WebSocket to all connected
    admin panels.  The admin has `override_deadline` seconds to submit
    an OverrideRequest.

    Attributes:
        junction_id:        Junction identifier (e.g. "J1")
        phase:              Phase index (0-3)
        phase_name:         Human-readable phase name
        rl_duration:        Duration chosen by the RL agent (seconds)
        final_duration:     Actual duration after override (= rl_duration initially)
        sim_time:           Simulation time when decision was made
        cycle:              Current cycle count
        timestamp:          Wall-clock timestamp
        override_deadline:  Seconds remaining for admin to override
        override_applied:   Whether an admin override was applied
        override_by:        Admin user ID (if overridden)
    """
    junction_id: str
    phase: int
    phase_name: str
    rl_duration: int
    final_duration: int
    sim_time: int
    cycle: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    override_deadline: float = 5.0
    override_applied: bool = False
    override_by: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON/WebSocket transmission."""
        return asdict(self)

    def to_websocket_message(self) -> dict:
        """Format for WebSocket broadcast."""
        return {
            "type": "decision",
            "payload": self.to_dict(),
        }


# ── Override Request ──────────────────────────────────────────────────────

@dataclass
class OverrideRequest:
    """
    Submitted by an admin to adjust an RL decision.

    Constraints (enforced by the web layer):
      - Must arrive within `AdminPanelConfig.OVERRIDE_WINDOW_SECONDS` (5s)
      - Delta must be in [-MAX_OVERRIDE_DELTA, +MAX_OVERRIDE_DELTA] (±10s)
      - Final duration must respect MIN_GREEN (15s)

    Attributes:
        junction_id:  Target junction (e.g. "J2")
        phase:        Target phase (0-3)
        delta:        Duration adjustment in seconds (+10 = extend, -10 = shorten)
        admin_id:     ID of the admin user making the override
        reason:       Optional human-readable reason
        timestamp:    When the override was submitted
    """
    junction_id: str
    phase: int
    delta: int
    admin_id: str
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate(self, max_delta: int = 10, min_green: int = 15) -> bool:
        """
        Validate override constraints.

        Args:
            max_delta: Maximum allowed adjustment (±seconds)
            min_green: Minimum green time (safety floor)

        Returns:
            True if override is valid
        """
        if abs(self.delta) > max_delta:
            return False
        if self.phase < 0 or self.phase > 3:
            return False
        return True

    def to_dict(self) -> dict:
        return asdict(self)


# ── Override Result ───────────────────────────────────────────────────────

@dataclass
class OverrideResult:
    """
    Result of applying (or rejecting) an OverrideRequest.

    Stored in the audit log for accountability.
    """
    request: OverrideRequest
    accepted: bool
    original_duration: int
    final_duration: int
    rejection_reason: str = ""
    applied_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "request": self.request.to_dict(),
            "accepted": self.accepted,
            "original_duration": self.original_duration,
            "final_duration": self.final_duration,
            "rejection_reason": self.rejection_reason,
            "applied_at": self.applied_at,
        }


# ── Green-Wave Request ────────────────────────────────────────────────────

@dataclass
class GreenWaveRequest:
    """
    Admin request to activate a green-wave corridor.

    The GreenWaveController calculates phase offsets based on
    inter-junction distance and speed.

    Attributes:
        junction_ids:  Ordered list of junctions in the corridor
        direction:     "eastbound" or "westbound"
        speed_mps:     Assumed vehicle speed (m/s) for offset calculation
        admin_id:      Admin who activated the green wave
        active:        Whether the green wave is currently active
    """
    junction_ids: List[str]
    direction: str  # "eastbound" or "westbound"
    speed_mps: float = 15.0
    admin_id: str = ""
    active: bool = True
    activated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


# ── State Snapshot ────────────────────────────────────────────────────────

@dataclass
class JunctionSnapshot:
    """Per-junction state for dashboard display."""
    junction_id: str
    current_phase: int
    phase_name: str
    queues: Dict[str, float]        # {"N": 5.0, "S": 3.0, ...}
    wait_times: Dict[str, float]    # {"N": 12.5, "S": 8.0, ...}
    emergency_flag: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StateSnapshot:
    """
    Full system state snapshot for dashboard display.

    The web layer should request this at `AdminPanelConfig.METRICS_UPDATE_RATE_HZ`
    (default: 1 Hz) and render it on the admin dashboard.
    """
    junctions: List[JunctionSnapshot]
    system_mode: str = "TRAINING"
    system_status: str = "NORMAL"   # NORMAL, WARNING, EMERGENCY
    sim_time: int = 0
    cycle: int = 0
    total_arrived: int = 0
    active_green_wave: Optional[GreenWaveRequest] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        result = {
            "system_mode": self.system_mode,
            "system_status": self.system_status,
            "sim_time": self.sim_time,
            "cycle": self.cycle,
            "total_arrived": self.total_arrived,
            "timestamp": self.timestamp,
            "junctions": [j.to_dict() for j in self.junctions],
            "active_green_wave": (
                self.active_green_wave.to_dict()
                if self.active_green_wave else None
            ),
        }
        return result

    def to_websocket_message(self) -> dict:
        """Format for periodic WebSocket broadcast."""
        return {
            "type": "state_snapshot",
            "payload": self.to_dict(),
        }
