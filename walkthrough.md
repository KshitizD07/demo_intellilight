# Intelli-Light — Walkthrough & Handoff Guide

All four phases of the corridor optimization plan are **complete**.

---

## What Was Done

### Phase 1 — Codebase Cleanup ✅
| File | Action |
|---|---|
| `.gitignore` | Removed broad `*.xml` glob that was accidentally ignoring SUMO network files |
| `simulation/route_generator.py` | Removed broken `ResourceConfig` import — replaced with hardcoded default of 50 |
| `training/evaluation_callback.py` | Fixed `TrafficEnv` → `TrafficEnv4Phase` import |
| `training/train_rl.py` | **Deleted** — legacy broken script; all training now via `train_rl_4phase.py` |
| `train.py`, `test_model.py`, `watch_model.py` | **Deleted** — replaced by unified `main.py` CLI |
| `main.py` | **Implemented** from scratch — full CLI with `train`, `evaluate`, `test-gui` sub-commands |

---

### Phase 2 — Corridor SUMO Network ✅
| File | Action |
|---|---|
| `configs/sumo/corridor.nod.xml` | **New** — 3 traffic-light junctions (J1, J2, J3) + 9 peripheral nodes |
| `configs/sumo/corridor.edg.xml` | **New** — arterial links J1↔J2↔J3 and side-street stubs for each junction |
| `configs/sumo/corridor.net.xml` | **Generated** via `netconvert` — compiled SUMO network |
| `configs/sumo/corridor.sumocfg` | **New** — SUMO config referencing corridor.net.xml |
| `configs/parameters.py` | Updated: `TRAFFIC_LIGHT_IDS = ["J1","J2","J3"]`, `OBS_SIZE = 42`, `N_ENVS = 2` |

---

### Phase 3 — Multi-Intersection RL Core ✅
| File | Action |
|---|---|
| `rl/reward_function.py` | Added `green_wave` weight + `_green_wave_reward()` method. Fully backward-compatible with single-junction mode. |
| `rl/multi_agent_env.py` | **New** — `CorridorEnv`: centralised Gymnasium env for 3 junctions. Obs `(42,)`, action `MultiDiscrete([8]*12)`. Passes neighbour phase context to activate green-wave bonus. |
| `training/train_rl_4phase.py` | Switched to `CorridorEnv`, removed duplicate print statements, updated checkpoint prefix to `intellilight_corridor`. |

**Sanity test results (local, no SUMO launch):**
```
Obs space : Box(0.0, 1.0, (42,), float32)   ✅
Action    : MultiDiscrete([8 8 8 8 8 8 8 8 8 8 8 8])  ✅
Green wave reward with sync neighbours: +2.93  (vs base -1.07)  ✅
```

---

### Phase 4 — Colab Handoff ✅
| File | Action |
|---|---|
| `requirements.txt` | **New** — all Python dependencies with notes on SUMO apt install |
| `colab_training.ipynb` | **New** — step-by-step Colab notebook (9 cells: install → train → download) |
| `intelli_light_colab.zip` | **Built** — 0.08 MB clean package of all source files (no `models/`, `logs/`, `.git/`) |

---

## 🚀 Colab Training Instructions

### Step 1 — Upload the zip
Open [Google Colab](https://colab.research.google.com), create a **T4 GPU** runtime,
and upload the file:
```
c:\Users\kshit\cs\project\intelli_light_colab.zip
```

### Step 2 — Open the notebook
Upload `colab_training.ipynb` OR copy each cell from the notebook into Colab.

### Step 3 — Run in order
| Cell | What it does |
|---|---|
| 1 | Installs SUMO via `apt` (~2 min) |
| 2 | Installs Python packages |
| 3 | Extracts the zip to `/content/Intelli-Light` |
| 4 | Sets `SUMO_HOME` + enables `LIBSUMO_AS_TRACI` for 3× speed |
| 5 | Import sanity check |
| 6 | Starts TensorBoard **live monitoring** |
| 7 | **TRAINS** for 1 M steps (~2–4 h on T4) |
| 8 | Downloads `intellilight_corridor_final.zip` to your machine |
| 9 | (Info only) Local test commands |

### After Training — Test Locally
```bash
# Place the downloaded file in models/checkpoints/ then:

# Visual evaluation in SUMO GUI
python main.py test-gui --model models/checkpoints/intellilight_corridor_final.zip --plot

# Quantitative evaluation vs Fixed-Timer and Max-Pressure
python main.py evaluate
```

---

## Hardware Constraints Respected
- `N_ENVS = 2` in `configs/parameters.py` — safe for RTX 3050 (4 GB VRAM / 16 GB RAM)
- `main.py train` defaults to `--envs 2`
- Training script never opens GUI unless `--gui` flag is explicitly passed
- Colab runs with `N_ENVS = 4` (Colab has ~13 GB RAM per session)
