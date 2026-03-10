"""
IntelliLight Configuration Parameters
======================================

Centralized configuration management for the IntelliLight traffic control system.
This file contains all system parameters organized by category for easy tuning and deployment.

Design Philosophy:
- All magic numbers are defined here
- Parameters are grouped by functional area
- Documentation explains the purpose and constraints of each parameter
- Real-world deployment settings are clearly marked
"""

import os

# =============================================================================
# SUMO SIMULATION CONFIGURATION
# =============================================================================

class SUMOConfig:
    """SUMO simulation engine configuration."""
    
    # Binary and file paths
    BINARY = os.environ.get("SUMO_BINARY", "sumo")  # Use "sumo-gui" for visualization
    CONFIG_FILE = "intersection.sumocfg"
    
    # Directory structure
    ROUTE_DIR = "data/generated_routes"  # Auto-generated traffic scenarios
    SUMO_CONFIG_DIR = "configs/sumo"     # Static SUMO network files
    
    # Performance optimization flags
    FLAGS = [
        "--no-step-log",                      # Disable step logging (faster)
        "--no-warnings", "true",              # Suppress warnings (cleaner logs)
        "--duration-log.disable",             # Disable duration logging
        "--tripinfo-output.write-unfinished", # Handle incomplete trips gracefully
        "--time-to-teleport", "300"           # Teleport stuck vehicles after 5 min
    ]
    
    # Operational parameters
    TIMEOUT = 30  # Seconds to wait for SUMO operations before failing
    

class NetworkTopology:
    """
    Network topology configuration matching SUMO network definition.
    
    These must match the actual network file (intersection_net.xml).
    Modify these if you change the intersection layout.
    """
    
    # Induction loop detectors by direction
    DETECTORS = {
        "W": ["det_W_0", "det_W_1"],  # West approach
        "E": ["det_E_0", "det_E_1"],  # East approach
        "N": ["det_N_0", "det_N_1"],  # North approach
        "S": ["det_S_0", "det_S_1"]   # South approach
    }
    
    # Lane identifiers by direction
    LANES = {
        "W": ["W1_to_J1_0", "W1_to_J1_1"],  # West to Junction
        "E": ["E1_to_J1_0", "E1_to_J1_1"],  # East to Junction
        "N": ["N1_to_J1_0", "N1_to_J1_1"],  # North to Junction
        "S": ["S1_to_J1_0", "S1_to_J1_1"]   # South to Junction
    }
    
    # Outgoing edges from intersection
    OUTGOING_EDGES = ["J1_to_N1", "J1_to_S1", "J1_to_E1", "J1_to_W1"]
    
    # Traffic light program ID
    TRAFFIC_LIGHT_ID = "J1"  # Junction 1 traffic light


# =============================================================================
# TRAFFIC SIGNAL TIMING PARAMETERS
# =============================================================================

class SignalTiming:
    """
    Traffic signal timing parameters.
    
    These values directly affect safety and traffic flow.
    Modifications should consider local traffic regulations.
    """
    
    # Green light duration options (seconds)
    MIN_GREEN = 15             # Minimum safe green time (safety requirement)
    GREEN_OPTIONS = [15, 18,21,24,27, 30,33,36,39,42,45,48,51,54,57,60]  # Available green durations for RL agent
    
    # Safety intervals
    ALL_RED = 4                 # All-red clearance interval (safety critical)
    YELLOW_TIME = 3             # Yellow (amber) warning time
    
    # Fairness constraints
    MAX_WAIT = 60               # Maximum acceptable wait time before penalty (seconds)
    
    # Phase definitions (which directions get green simultaneously)
    PHASES = {
        "EW": ["E", "W"],       # East-West green
        "NS": ["N", "S"]        # North-South green
    }


# =============================================================================
# SIMULATION SCENARIO PARAMETERS
# =============================================================================

class ScenarioConfig:
    """Traffic scenario generation parameters."""
    
    # Realistic sensor modeling
    DETECTION_MISS_PROB = 0.1   # 10% probability of missed vehicle detection
    
    # Emergency vehicle simulation
    EMERGENCY_PROB_EP = 0.5     # 50% of episodes include emergency vehicle
    EMERGENCY_DEPART_RANGE = (300, 1200)  # Emergency appears between 5-20 min
    
    # Available traffic scenarios
    SCENARIOS = ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND"]
    
    # Traffic volume definitions (vehicles per hour)
    TRAFFIC_VOLUMES = {
        "MORNING_RUSH": {
            "major_in": (600, 800),   # Heavy inbound traffic
            "major_out": (300, 400),  # Light outbound traffic
            "minor": (100, 200)       # Low cross traffic
        },
        "EVENING_RUSH": {
            "major_in": (300, 400),   # Light inbound traffic
            "major_out": (600, 800),  # Heavy outbound traffic
            "minor": (100, 200)       # Low cross traffic
        },
        "WEEKEND": {
            "major_in": (250, 350),   # Balanced traffic
            "major_out": (250, 350),  # Balanced traffic
            "minor": (150, 250)       # Moderate cross traffic
        }
    }
    
    # Vehicle type distribution (must sum to 1.0)
    VEHICLE_DISTRIBUTION = {
        "car": 0.70,        # 70% cars
        "2-wheeler": 0.20,  # 20% two-wheelers
        "bus": 0.10         # 10% buses
    }


# =============================================================================
# EPISODE AND TRAINING PARAMETERS
# =============================================================================

class EpisodeConfig:
    """Simulation episode configuration."""
    
    LENGTH = 1800              # Episode duration in simulation seconds (30 minutes)
    SIM_STEP = 1               # Simulation time step (1 second)
    MAX_STEPS = LENGTH // SIM_STEP  # Maximum steps per episode


class TrainingConfig:
    """Reinforcement learning training parameters."""
    
    # PPO hyperparameters (tuned for traffic control)
    LEARNING_RATE = 1e-4
    N_STEPS = 2048              # Steps per update
    BATCH_SIZE = 64
    N_EPOCHS = 5
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95          # GAE parameter
    CLIP_RANGE = 0.1           # PPO clip range
    ENT_COEF = 0.01            # Entropy coefficient (exploration)
    VF_COEF = 0.5              # Value function coefficient
    MAX_GRAD_NORM = 0.5        # Gradient clipping
    
    # Training schedule
    TOTAL_TIMESTEPS = 300000    # Default training duration
    EVAL_FREQUENCY = 10000      # Evaluate every N timesteps
    SAVE_FREQUENCY = 50000      # Save model every N timesteps
    
    # Vectorized training
    DEFAULT_N_ENVS = 4          # Number of parallel environments
    USE_VECENV = True           # Enable vectorized training by default


# =============================================================================
# OBSERVATION AND ACTION SPACE PARAMETERS
# =============================================================================

class ObservationConfig:
    """Observation space configuration."""
    
    # Observation vector structure: [queue_W, queue_E, queue_N, queue_S,
    #                                wait_W, wait_E, wait_N, wait_S,
    #                                emergency_flag]
    SHAPE = (9,)
    
    # Normalization bounds
    MAX_CARS_ON_LANE = 50.0     # Maximum expected vehicles per lane
    MAX_WAIT_TIME = EpisodeConfig.LENGTH  # Maximum possible wait time
    
    # Observation indices (for readability in code)
    QUEUE_INDICES = slice(0, 4)      # Indices 0-3: queue lengths
    WAIT_INDICES = slice(4, 8)       # Indices 4-7: wait times
    EMERGENCY_INDEX = 8              # Index 8: emergency flag


class ActionConfig:
    """Action space configuration."""
    
    # Action structure: [direction, duration]
    # direction: 0 = EW, 1 = NS
    # duration: index into SignalTiming.GREEN_OPTIONS
    N_DIRECTIONS = 2
    # N_DURATIONS = len(SignalTiming.GREEN_OPTIONS)
    N_DURATIONS=16


# =============================================================================
# REWARD FUNCTION PARAMETERS
# =============================================================================

class RewardConfig:
    """
    Multi-objective reward function weights.
    
    The total reward is a weighted combination of multiple objectives.
    Adjust these to prioritize different goals (efficiency vs fairness vs emissions).
    """
    
    # Component weights (must be tuned together)
    WAIT_TIME_WEIGHT = -0.05      # Minimize total waiting time
    THROUGHPUT_WEIGHT = 1.0     # Maximize vehicles served
    FAIRNESS_WEIGHT = -0.1       # Penalize unequal treatment
    EMERGENCY_WEIGHT = 5.0      # Prioritize emergency vehicles
    
    # Penalty thresholds
    EXCESSIVE_WAIT_THRESHOLD = SignalTiming.MAX_WAIT  # When to apply fairness penalty
    QUEUE_LENGTH_THRESHOLD = 30   # When queue is considered critical
    
    # Normalization factors (for reward scaling)
    BASELINE_WAIT_TIME = 100      # Expected average wait time
    BASELINE_THROUGHPUT = 200     # Expected vehicles per episode


# =============================================================================
# CURRICULUM LEARNING PARAMETERS
# =============================================================================

class CurriculumConfig:
    """
    Progressive difficulty scaling for training.
    
    The agent starts with simple scenarios and gradually faces more complex traffic.
    """
    
    # Curriculum stages definition
    STAGES = {
        0: {"max_flow": 400, "complexity": "LOW"},      # Light traffic
        1: {"max_flow": 600, "complexity": "MEDIUM"},   # Moderate traffic
        2: {"max_flow": 800, "complexity": "HIGH"}      # Heavy traffic
    }
    
    # Stage transition thresholds (timesteps)
    TRANSITIONS = [50000, 150000, 300000]
    
    # Whether to enable curriculum learning
    ENABLED = True


# =============================================================================
# RESOURCE MANAGEMENT PARAMETERS
# =============================================================================

class ResourceConfig:
    """System resource management settings."""
    
    # Route file cleanup
    MAX_ROUTE_FILES = 50        # Maximum route files before cleanup
    CLEANUP_INTERVAL = 100      # Cleanup every N episodes
    
    # Memory management
    CACHE_SIZE = 1000           # Observation cache size
    
    # Process management
    SUMO_PROCESS_CHECK_INTERVAL = 5  # Check for zombie processes every N episodes


# =============================================================================
# LOGGING AND MONITORING PARAMETERS
# =============================================================================

class LoggingConfig:
    """Logging and monitoring configuration."""
    
    # Log levels
    DEFAULT_LEVEL = "INFO"      # Options: DEBUG, INFO, WARNING, ERROR
    
    # Log format
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance metrics
    TRACK_PERFORMANCE = True    # Enable performance monitoring
    METRICS_WINDOW = 100        # Moving average window for metrics


# =============================================================================
# DEPLOYMENT PARAMETERS
# =============================================================================

class DeploymentConfig:
    """Real-world deployment settings."""
    
    # Failsafe configuration
    ENABLE_FAILSAFE = True      # Always enable in production
    FAILSAFE_TIMEOUT = 5.0      # Switch to failsafe if RL takes > 5 seconds
    
    # Health monitoring
    HEALTH_CHECK_INTERVAL = 60  # Check system health every minute
    MAX_CONSECUTIVE_ERRORS = 3  # Switch to failsafe after N errors
    
    # Model serving
    MODEL_UPDATE_INTERVAL = 3600  # Check for model updates every hour (production)
    
    # Edge device settings (for future deployment)
    EDGE_DEVICE_MODE = False    # Enable for edge deployment
    LOW_POWER_MODE = False      # Reduce computation for battery operation


# =============================================================================
# FILE PATHS (Auto-generated, don't modify)
# =============================================================================

class Paths:
    """Auto-generated paths based on project structure."""
    
    # Project root
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directories
    DATA_DIR = os.path.join(ROOT, "data")
    ROUTE_DIR = os.path.join(DATA_DIR, SUMOConfig.ROUTE_DIR)
    
    # Config directories
    CONFIG_DIR = os.path.join(ROOT, "configs")
    SUMO_CONFIG_DIR = os.path.join(CONFIG_DIR, "sumo")
    
    # Model directories
    MODELS_DIR = os.path.join(ROOT, "models")
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
    
    # Log directories
    LOGS_DIR = os.path.join(ROOT, "logs")
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.ROUTE_DIR,
            cls.CONFIG_DIR,
            cls.SUMO_CONFIG_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_configuration():
    """
    Validates configuration parameters for consistency.
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate green light durations
    if SignalTiming.MIN_GREEN not in SignalTiming.GREEN_OPTIONS:
        raise ValueError(f"MIN_GREEN ({SignalTiming.MIN_GREEN}) must be in GREEN_OPTIONS")
    
    # Validate vehicle distribution
    total_dist = sum(ScenarioConfig.VEHICLE_DISTRIBUTION.values())
    if not (0.99 <= total_dist <= 1.01):  # Allow floating point error
        raise ValueError(f"Vehicle distribution must sum to 1.0, got {total_dist}")
    
    # Validate observation shape
    expected_shape = (4 + 4 + 1,)  # queues + waits + emergency
    if ObservationConfig.SHAPE != expected_shape:
        raise ValueError(f"Observation shape mismatch: {ObservationConfig.SHAPE} != {expected_shape}")
    
    # Validate curriculum stages
    if len(CurriculumConfig.TRANSITIONS) != len(CurriculumConfig.STAGES):
        raise ValueError("Number of curriculum transitions must match number of stages")
    
    print("✓ Configuration validation passed")


# =============================================================================
# INITIALIZATION
# =============================================================================

# Create directories on import
Paths.create_directories()

# Optionally validate on import (comment out for production)
if __name__ == "__main__":
    validate_configuration()
    print("\n=== IntelliLight Configuration Summary ===")
    print(f"SUMO Binary: {SUMOConfig.BINARY}")
    print(f"Episode Length: {EpisodeConfig.LENGTH}s")
    print(f"Observation Shape: {ObservationConfig.SHAPE}")
    print(f"Green Options: {SignalTiming.GREEN_OPTIONS}")
    print(f"Curriculum Enabled: {CurriculumConfig.ENABLED}")
    print(f"Route Directory: {Paths.ROUTE_DIR}")
    print("==========================================\n")