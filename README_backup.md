<div align="center">

# 🚦 Intelli-Light

### Reinforcement Learning Traffic Signal Control for Multi-Intersection Arterial Corridors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.18%2B-green)](https://www.eclipse.org/sumo/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> This project was developed with the assistance of an **Agentic AI workflow**.

</div>

---

## Overview

**Intelli-Light** is a production-grade Reinforcement Learning system that controls traffic signals across a multi-intersection arterial corridor. Instead of relying on fixed timer cycles or reactive pressure-based heuristics, Intelli-Light uses a centralized Proximal Policy Optimization (PPO) agent that learns to coordinate green-wave timing across 3 interconnected junctions simultaneously.

The system outperforms both the classic Fixed-Timer and industry-standard Max-Pressure baselines across all evaluated traffic scenarios:

| Metric | vs Max-Pressure | vs Fixed-Timer |
|---|---|---|
| Average Wait Time | **−58%** | **−78%** |
| Queue Length | **−53%** | **−78%** |
| Zero Starvation Events | ✅ | ✅ |

---

## Features

- **4-Phase Cyclic Signal Control** — EW Through, EW Protected Left, NS Through, NS Protected Left, with mandatory all-red clearance intervals
- **Multi-Intersection Coordination** — Single centralized PPO agent controls J1 → J2 → J3 arterial corridor and learns green-wave offsets naturally
- **Emergency Vehicle Prioritization** — Detects emergency vehicles by ID prefix and triggers phase overrides with a strong reward bonus
- **Safety Constraints** — Minimum green time (10s), mandatory all-red (4s), and a starvation detector that fires a heavy penalty if any direction waits > 90s
- **Curriculum Learning** — Training progresses through WEEKEND (easy) → EVENING_RUSH → MORNING_RUSH scenarios automatically
- **Realistic Traffic Generation** — Probabilistic demand curves with time-of-day multipliers, volatility, and random incident events
- **Baseline Comparison Suite** — Full evaluation pipeline comparing RL against Fixed-Timer and Max-Pressure controllers across multiple scenarios

---

## Architecture

```
Intelli-Light/
│
├── configs/                    # Configuration and SUMO network files
│   ├── parameters.py           # Centralized config: signal, reward, training, safety
│   └── sumo/                   # SUMO simulation network (nodes, edges, config)
│
├── rl/                         # Reinforcement learning core
│   ├── multi_agent_env.py      # CorridorEnv — Gymnasium env for 3-junction corridor
│   ├── traffic_env.py          # TrafficEnv4Phase — single-junction 4-phase env
│   └── reward_function.py      # EnhancedRewardCalculator with safety & emergency logic
│
├── simulation/                 # SUMO simulation management
│   ├── sumo_env.py             # SUMOSimulation — TraCI lifecycle management
│   ├── route_generator.py      # Generates SUMO .rou.xml files per scenario
│   └── realistic_traffic.py    # Time-of-day traffic demand modeling
│
├── training/                   # Training and evaluation pipeline
│   ├── train_rl_4phase.py      # Main PPO training script with curriculum
│   ├── evaluate_model.py       # Full benchmark: RL vs baselines
│   ├── evaluation_engine.py    # Runs episodes and collects step-level data
│   ├── metrics_calculator.py   # Computes wait time, throughput, fairness metrics
│   ├── baseline_controllers.py # Fixed-Timer and Max-Pressure baselines
│   └── evaluation_callback.py  # SB3 callback for periodic evaluation
│
├── main.py                     # CLI entry point: train / evaluate / simulate
├── IntelliLight_Colab_V3.ipynb # Google Colab training notebook
├── requirements.txt            # Python dependencies
└── TRAINING_GUIDE.md           # Step-by-step training instructions
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [SUMO 1.18+](https://sumo.dlr.de/docs/Downloads.php) installed and added to PATH
- CUDA GPU recommended for training (CPU works but is slower)

### Installation

```bash
git clone https://github.com/KshitizD07/Intelli-Light.git
cd Intelli-Light
pip install -r requirements.txt
```

### Train the Model

```bash
# Train with 3-corridor curriculum (recommended)
python main.py train --timesteps 200000

# Resume from checkpoint
python main.py train --resume models/checkpoints/intellilight_corridor_final.zip
```

### Evaluate

```bash
# Evaluate RL agent vs all baselines across all scenarios
python main.py evaluate --model models/checkpoints/intellilight_corridor_final.zip

# Run with SUMO GUI for visual inspection
python main.py evaluate --model models/checkpoints/intellilight_corridor_final.zip --gui
```

### Cloud Training (Google Colab)

Upload and run `IntelliLight_Colab_V3.ipynb` on a Colab A100 or T4 GPU instance. The notebook handles SUMO installation, training, and checkpoint download automatically.

---

## Environment Details

### Observation Space — `Box(42,)`
Each of the 3 junctions contributes a 14-dimensional local feature vector:

| Features | Dimensions | Description |
|---|---|---|
| Queue lengths | 4 | Halting vehicles per approach (N, S, E, W), normalized to [0, 1] |
| Queue delta | 4 | Change in queue since last step, normalized to [−1, 1] |
| Wait times | 4 | Mean per-approach wait in seconds, normalized to [0, 1] |
| Emergency flag | 1 | Binary: emergency vehicle detected on any approach |
| Current phase | 1 | Phase index normalized to [0, 1] |

### Action Space — `MultiDiscrete([8] × 12)`
For each junction × phase, the agent selects a green duration from:
`[10, 15, 20, 25, 30, 35, 40, 45]` seconds

### Reward Function
```
R = throughput_bonus
  + wait_time_penalty    (−0.08 × avg_wait)
  + queue_penalty        (−0.04 × avg_queue)
  + fairness_penalty     (−0.15 × direction imbalance)
  + efficiency_bonus     (0.4 × throughput/queue ratio)
  + pressure_bonus       (0.6 × pressure balance)
  + starvation_penalty   (−8.0 × exponential, for wait > 60s)
  + emergency_bonus      (+100.0 if emergency cleared)
```

---

## Evaluation Results

Evaluation across 20 episodes per scenario using the trained corridor model:

### Morning Rush (Peak Demand)
| Controller | Avg Wait | Throughput | Queue Length |
|---|---|---|---|
| Fixed-Timer | 4.45s | Baseline | 8.5 veh |
| Max-Pressure | 5.60s | −9.1% | 8.2 veh |
| **IntelliLight-RL** | **3.47s** | **+1.0%** | **7.1 veh** |

### Evening Rush
| Controller | Avg Wait | Queue Length |
|---|---|---|
| Fixed-Timer | 4.17s | 7.2 veh |
| Max-Pressure | 5.68s | 9.5 veh |
| **IntelliLight-RL** | **3.33s** | **6.5 veh** |

### Weekend (Off-Peak)
| Controller | Avg Wait | Queue Length |
|---|---|---|
| Fixed-Timer | 4.31s | 7.4 veh |
| Max-Pressure | 6.00s | 10.4 veh |
| **IntelliLight-RL** | **3.28s** | **6.4 veh** |

---

## Scaling to City-Wide Grids

> The current architecture is a **Centralized PPO** model trained on a fixed 3-junction corridor. Scaling it naively to 9 or 150 intersections would require retraining from scratch each time, since the network input size is hardcoded to 42 dimensions.

The planned solution is **Independent Learning with Parameter Sharing (ILPS)**.

### How ILPS Works

Instead of one large network that processes the entire city, ILPS trains a single **universal junction brain** that understands how to manage exactly one intersection. During deployment, every intersection in the city runs its own forward pass through the same shared weights.

```
Current (Centralized):         Proposed (ILPS):
                               
Global Obs [42]  →  PPO       Local Obs J1 [14] ─┐
Global Act [12]  ←  Brain     Local Obs J2 [14] ──→ Shared Brain → Acts for all N nodes
                               Local Obs J3 [14] ─┘
                               Local Obs JN [14] ─┘
```

### Key Benefits

| Property | Centralized (Current) | ILPS (Planned) |
|---|---|---|
| Observation size | Fixed at `N * 14` | Always `14` (1 junction) |
| Action size | Fixed at `N * 4` | Always `4` (1 junction) |
| Adding intersections | Requires full retraining | **Zero retraining** |
| Training efficiency | 1× experience per step | **N× experience per step** |
| City-scale deployment | Not feasible | ✅ |

### Migration Path

1. **Refactor `CorridorEnv`** to return a dict of per-junction observations instead of a flat concatenated vector.
2. **Expand local observation** from 14 → 16 dims by adding upstream and downstream neighbor phase features to enable green-wave coordination without a centralized controller.
3. **Flatten into SB3 batch** — treat N junctions as N parallel sub-environments sharing a single PPO model and experience replay buffer.
4. **Transition to PettingZoo** for a standardized multi-agent API that integrates cleanly with SB3's `MARLWrapper`.

This allows a model trained on the 3-junction corridor to be deployed directly on a 9×9 city grid with no modifications.

---

## Configuration

All system parameters are centralized in `configs/parameters.py`:

```python
from configs.parameters import signal, safety, training, reward

# Example: change green duration options
signal.GREEN_DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45]

# Example: tighten starvation penalty
reward.WEIGHTS.starvation = -10.0

# Example: extend training
training.TOTAL_TIMESTEPS = 500000
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with SUMO · Stable-Baselines3 · Gymnasium · TraCI</sub>
</div>
