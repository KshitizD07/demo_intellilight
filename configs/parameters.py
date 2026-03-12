"""
IntelliLight Configuration Parameters
======================================

Centralized configuration for 4-phase production system.

UPDATES:
- 4-phase signal parameters
- Enhanced safety settings
- Realistic traffic configuration
- Production reward weights
"""

from dataclasses import dataclass
from typing import List


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """SUMO simulation settings."""
    
    # Episode settings
    EPISODE_LENGTH = 1800  # 30 minutes in seconds
    STEP_LENGTH = 1.0      # SUMO time step (seconds)
    
    # Network
    SUMO_CONFIG_FILE = "intersection.sumocfg"
    TRAFFIC_LIGHT_ID = "J1"
    
    # GUI
    USE_GUI = False  # Set True for visualization
    GUI_DELAY = 0    # ms delay between steps (0 = max speed)


# ============================================================================
# 4-PHASE SIGNAL PARAMETERS
# ============================================================================

class SignalConfig:
    """4-phase traffic signal settings."""
    
    # Number of phases
    N_PHASES = 4
    
    # Phase definitions
    PHASE_NAMES = [
        "EW Through + Right",
        "EW Protected Left",
        "NS Through + Right", 
        "NS Protected Left"
    ]
    
    # SUMO phase IDs (must match your .net.xml)
    SUMO_PHASE_MAP = {
        0: 0,  # EW through
        1: 1,  # EW left
        2: 2,  # NS through
        3: 3   # NS left
    }
    
    # Green duration options (seconds)
    GREEN_DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45]
    
    # Minimum timings (safety constraints)
    MIN_GREEN = 10      # Minimum green time
    ALL_RED = 4         # Clearance interval between phases
    YELLOW_TIME = 3     # Yellow light duration (if defined in network)
    
    # Cyclic enforcement
    CYCLIC_MODE = True  # Enforce 0→1→2→3→0 cycle


# ============================================================================
# TRAFFIC GENERATION PARAMETERS
# ============================================================================

class TrafficConfig:
    """Realistic traffic generation settings."""
    
    # Base demand (vehicles per hour)
    BASE_DEMAND = 800
    
    # Volatility (random variance)
    VOLATILITY = 0.3  # ±30% realistic uncertainty
    
    # Random events
    ENABLE_EVENTS = True
    EVENT_PROBABILITY = 0.10  # 10% chance per episode
    
    # Time-of-day curves
    ENABLE_TIME_CURVES = True
    
    # Scenario definitions
    SCENARIOS = {
        "WEEKEND": {
            "base_demand": 600,
            "volatility": 0.25,
            "time_of_day": 14.0  # 2 PM
        },
        "MORNING_RUSH": {
            "base_demand": 1200,
            "volatility": 0.35,
            "time_of_day": 8.0   # 8 AM
        },
        "EVENING_RUSH": {
            "base_demand": 1100,
            "volatility": 0.35,
            "time_of_day": 18.0  # 6 PM
        },
        "NIGHT": {
            "base_demand": 300,
            "volatility": 0.40,
            "time_of_day": 2.0   # 2 AM
        }
    }


# ============================================================================
# SAFETY PARAMETERS
# ============================================================================

class SafetyConfig:
    """Production safety constraints."""
    
    # Maximum acceptable wait times
    MAX_ACCEPTABLE_WAIT = 90  # seconds (HARD LIMIT)
    STARVATION_THRESHOLD = 60  # Start penalizing after 60s
    
    # Emergency vehicles
    EMERGENCY_MAX_WAIT = 30   # Max acceptable emergency wait
    EMERGENCY_PRIORITY = True  # Enable emergency override
    
    # Queue limits
    MAX_QUEUE_WARNING = 40    # Warn at this queue length
    MAX_QUEUE_CRITICAL = 60   # Critical queue length
    
    # Failsafe triggers
    ENABLE_FAILSAFE = True
    FAILSAFE_TRIGGER_WAIT = 100  # Revert to fixed-timer if wait > 100s
    FAILSAFE_TRIGGER_QUEUE = 70  # Revert if queue > 70 vehicles


# ============================================================================
# REWARD FUNCTION PARAMETERS
# ============================================================================

@dataclass
class RewardWeights:
    """Enhanced reward component weights."""
    
    # Primary objectives
    throughput: float = 1.0      # Vehicles served
    wait_time: float = -0.08     # Average wait penalty
    queue: float = -0.04         # Queue length penalty
    
    # Secondary objectives  
    fairness: float = -0.15      # Direction imbalance penalty
    efficiency: float = 0.4      # Throughput/queue ratio
    pressure: float = 0.6        # Pressure balance (helps learning)
    
    # Safety (CRITICAL)
    starvation: float = -8.0     # Exponential penalty (STRONG!)
    emergency: float = 100.0     # Emergency vehicle bonus (HUGE!)


class RewardConfig:
    """Reward calculation settings."""
    
    # Weights
    WEIGHTS = RewardWeights()
    
    # Safety thresholds
    MAX_ACCEPTABLE_WAIT = SafetyConfig.MAX_ACCEPTABLE_WAIT
    EMERGENCY_MAX_WAIT = SafetyConfig.EMERGENCY_MAX_WAIT
    
    # Curriculum learning
    ENABLE_CURRICULUM = True
    CURRICULUM_STAGES = [
        {"stage": 0, "scenario": "WEEKEND", "until_step": 66000},
        {"stage": 1, "scenario": "EVENING_RUSH", "until_step": 133000},
        {"stage": 2, "scenario": "MORNING_RUSH", "until_step": 200000}
    ]


# ============================================================================
# OBSERVATION SPACE PARAMETERS
# ============================================================================

class ObservationConfig:
    """Observation space normalization."""
    
    # Normalization constants
    MAX_QUEUE = 50.0      # Expected max queue length
    MAX_WAIT = 120.0      # Expected max wait time
    
    # Observation components (4-phase)
    OBS_SIZE = 14  # queues(4) + delta(4) + waits(4) + emergency(1) + phase(1)
    
    # Feature indices (for debugging)
    QUEUE_START = 0
    QUEUE_END = 4
    DELTA_START = 4
    DELTA_END = 8
    WAIT_START = 8
    WAIT_END = 12
    EMERGENCY_IDX = 12
    PHASE_IDX = 13


# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

class TrainingConfig:
    """PPO training hyperparameters."""
    
    # Training duration
    TOTAL_TIMESTEPS = 200000
    
    # PPO hyperparameters
    LEARNING_RATE = 3e-4
    N_STEPS = 2048          # Steps per environment per update
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99            # Discount factor
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01         # Entropy coefficient
    VF_COEF = 0.5           # Value function coefficient
    MAX_GRAD_NORM = 0.5
    
    # Parallel environments
    N_ENVS = 4  # Good for RTX 3050
    
    # Checkpointing
    SAVE_FREQ = 25000       # Save every 25K steps
    EVAL_FREQ = 10000       # Evaluate every 10K steps
    
    # Logging
    LOG_INTERVAL = 1        # Log every update
    VERBOSE = 1


# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

class EvaluationConfig:
    """Model evaluation settings."""
    
    # Episode count
    N_EVAL_EPISODES = 20
    
    # Scenarios to test
    EVAL_SCENARIOS = ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND"]
    
    # Baselines
    BASELINES = [
        "Fixed-Timer",      # 30s fixed cycles
        "Max-Pressure",     # Adaptive pressure-based
        "IntelliLight-RL"   # Your model
    ]
    
    # Metrics to track
    METRICS = [
        "avg_wait_time",
        "max_wait_time",
        "throughput",
        "avg_queue_length",
        "max_queue_length",
        "intersection_utilization",
        "fairness_score",
        "phase_switches",
        "phase_switch_frequency",
        "starvation_events",
        "max_wait_per_direction"
    ]


# ============================================================================
# DEPLOYMENT PARAMETERS
# ============================================================================

class DeploymentConfig:
    """Production deployment settings."""
    
    # Failsafe mode
    ENABLE_FAILSAFE = SafetyConfig.ENABLE_FAILSAFE
    FAILSAFE_FALLBACK = "fixed-timer"  # or "max-pressure"
    
    # Monitoring
    LOG_SAFETY_VIOLATIONS = True
    ALERT_ON_STARVATION = True
    
    # Performance targets (for monitoring)
    TARGET_AVG_WAIT = 30.0      # seconds
    TARGET_THROUGHPUT = 600     # vehicles/hour
    MAX_STARVATION_EVENTS = 0   # ZERO tolerance in production


# ============================================================================
# PATHS
# ============================================================================

class PathConfig:
    """File paths."""
    
    # Models
    MODEL_DIR = "models"
    CHECKPOINT_DIR = "models/checkpoints"
    FINAL_MODEL = "models/checkpoints/intellilight_4phase_final.zip"
    
    # Logs
    LOG_DIR = "logs"
    TENSORBOARD_LOG = "logs/tensorboard"
    
    # Data
    ROUTES_DIR = "routes"
    RESULTS_DIR = "results"
    
    # SUMO network
    NETWORK_DIR = "."
    NETWORK_FILE = "intersection_net.xml"


# ============================================================================
# EXPORT ALL CONFIGS
# ============================================================================

# Create instances for easy import
simulation = SimulationConfig()
signal = SignalConfig()
traffic = TrafficConfig()
safety = SafetyConfig()
reward = RewardConfig()
observation = ObservationConfig()
training = TrainingConfig()
evaluation = EvaluationConfig()
deployment = DeploymentConfig()
paths = PathConfig()


# Convenience function
def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("INTELLILIGHT 4-PHASE CONFIGURATION")
    print("=" * 70)
    
    print("\n🚦 SIGNAL:")
    print(f"   Phases: {signal.N_PHASES}")
    print(f"   Green options: {signal.GREEN_DURATIONS}")
    print(f"   Min green: {signal.MIN_GREEN}s")
    print(f"   All-red: {signal.ALL_RED}s")
    
    print("\n🚗 TRAFFIC:")
    print(f"   Base demand: {traffic.BASE_DEMAND} veh/hour")
    print(f"   Volatility: ±{traffic.VOLATILITY*100:.0f}%")
    print(f"   Events enabled: {traffic.ENABLE_EVENTS}")
    
    print("\n🚨 SAFETY:")
    print(f"   Max wait: {safety.MAX_ACCEPTABLE_WAIT}s")
    print(f"   Emergency max: {safety.EMERGENCY_MAX_WAIT}s")
    print(f"   Failsafe: {safety.ENABLE_FAILSAFE}")
    
    print("\n💰 REWARD:")
    print(f"   Starvation penalty: {reward.WEIGHTS.starvation}")
    print(f"   Emergency bonus: {reward.WEIGHTS.emergency}")
    print(f"   Throughput weight: {reward.WEIGHTS.throughput}")
    
    print("\n🎓 TRAINING:")
    print(f"   Total steps: {training.TOTAL_TIMESTEPS:,}")
    print(f"   Learning rate: {training.LEARNING_RATE}")
    print(f"   Parallel envs: {training.N_ENVS}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config()