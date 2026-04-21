# Corridor Optimization & Bug Fix Tasks

## Phase 1: Codebase Cleanup and Bug Fixes
- `[x]` Fix `.gitignore` to preserve `*.net.xml` and `*.sumocfg`
- `[x]` Fix `simulation/route_generator.py` (remove `ResourceConfig`)
- `[x]` Fix `training/evaluation_callback.py` (`TrafficEnv` -> `TrafficEnv4Phase`)
- `[x]` Delete `training/train_rl.py`
- `[x]` Unify `test_model.py` and `watch_model.py` dictionary keys and model paths
- `[x]` Implement unified `main.py` CLI

## Phase 2: SUMO Network & Simulation (Corridor Layout)
- `[x]` Generate 3-intersection arterial corridor (`configs/sumo/corridor.net.xml`)
- `[x]` Update `configs/parameters.py` for multiple intersections (`TRAFFIC_LIGHT_IDS`)
- `[x]` Modify `simulation/sumo_env.py` to control multi-lights

## Phase 3: Multi-Intersection RL Core
- `[ ]` Implement `rl/multi_agent_env.py` (`CorridorEnv`) with Centralized Action Space
- `[ ]` Update Reward Function for green wave/coordination bonus
- `[ ]` Integrate `CorridorEnv` into `training/train_rl_4phase.py`
- `[ ]` Conduct brief local compilation & sanity test locally

## Phase 4: Training & Colab Handoff
- `[ ]` Package codebase for Google Colab upload (exclude logs/models/git)
- `[ ]` Hand over to the User for Colab training (~1M steps)
