# Git Commit Plan — Intelli-Light

This file maps each uncommitted file to a recommended git commit message.
Run these commits in the order listed below. `__pycache__` files should be excluded via `.gitignore`.

---

## Staged (Ready to Commit)

```
git add Intelli-Light_doc.pdf
git commit -m "docs: add project documentation PDF"
```

---

## Modified Source Files

### configs/parameters.py
```
git add configs/parameters.py
git commit -m "refactor(config): clean up module docstring and remove device-specific N_ENVS comment"
```

### rl/multi_agent_env.py
```
git add rl/multi_agent_env.py
git commit -m "feat(env): add cycle_time to step info dict and fix phase tracking for multi-intersection corridor

- Record simulation steps elapsed per cycle and expose as info['cycle_time']
- Snapshot current_phases as an immutable tuple to correctly track phase switches
- Update deployment note in module docstring to be hardware-agnostic"
```

### rl/reward_function.py
```
git add rl/reward_function.py
git commit -m "refactor(reward): clean up reward function implementation and docstrings"
```

### rl/traffic_env.py
```
git add rl/traffic_env.py
git commit -m "refactor(env): remove commented-out debug code from TrafficEnv4Phase

- Remove leftover commented-out cumulative arrival tracking block
- Remove debug print statement in lane vehicle iteration
- No functional changes"
```

### simulation/sumo_env.py
```
git add simulation/sumo_env.py
git commit -m "refactor(simulation): clean up SUMOSimulation docstrings and remove internal debug logging

- Remove FIXED tag from module docstring
- Remove verbose debug log from close() method
- No functional changes"
```

### simulation/route_generator.py
```
git add simulation/route_generator.py
git commit -m "refactor(simulation): clean up RouteGenerator implementation and comments"
```

### simulation/realistic_traffic.py
```
git add simulation/realistic_traffic.py
git commit -m "refactor(simulation): clean up RealisticTrafficModel and remove stale comments"
```

### main.py
```
git add main.py
git commit -m "feat(main): integrate multi-agent corridor environment into evaluation pipeline"
```

### training/baseline_controllers.py
```
git add training/baseline_controllers.py
git commit -m "fix(evaluation): update baseline controllers to support multi-intersection observation vectors

- MaxPressureController and FixedTimerController now dynamically detect
  number of intersections from observation shape (obs.size // 14)
- Aggregate queue and wait pressures across all local junction slices
- Output action tuples replicated for N intersections to match CorridorEnv action space
- Fix wait-time denormalization index (was using wrong slice 4:8, now uses 8:12)"
```

### training/evaluate_model.py
```
git add training/evaluate_model.py
git commit -m "fix(evaluation): run all three controllers and compute real improvement metrics

- Previously only the RL controller was evaluated; Fixed-Timer and Max-Pressure
  were skipped and improvements were hardcoded to zero
- Now evaluates all three controllers per scenario and uses MetricsCalculator
  to compute genuine percentage improvements
- Retain per-controller AggregatedMetrics objects to pass into compare_metrics()"
```

### training/evaluation_engine.py
```
git add training/evaluation_engine.py
git commit -m "fix(evaluation): correct phase snapshot logic and throughput key lookup

- Store phase as immutable tuple to enable correct change detection
- Fix throughput extraction: check info['throughput'] before info['total_arrived']
  to match key emitted by CorridorEnv step()"
```

### training/evaluation_callback.py
```
git add training/evaluation_callback.py
git commit -m "refactor(training): clean up EvaluationCallback implementation"
```

### training/metrics_calculator.py
```
git add training/metrics_calculator.py
git commit -m "refactor(evaluation): clean up MetricsCalculator docstrings and remove stale code"
```

### training/train_rl.py
```
git add training/train_rl.py
git commit -m "refactor(training): clean up single-intersection PPO training script"
```

### training/train_rl_4phase.py
```
git add training/train_rl_4phase.py
git commit -m "refactor(training): clean up 4-phase corridor PPO training script"
```

### .gitignore
```
git add .gitignore
git commit -m "chore: update .gitignore to exclude __pycache__ and route temp files"
```

---

## Deleted Files

```
git add -u test_model.py train.py watch_model.py
git commit -m "chore: remove obsolete standalone scripts

- test_model.py: superseded by training/evaluate_model.py
- train.py: superseded by training/train_rl_4phase.py
- watch_model.py: superseded by --gui flag in main.py evaluate command"
```

---

## New Untracked Files

### SUMO Network Configuration
```
git add configs/sumo/
git commit -m "feat(config): add SUMO corridor network configuration files

- corridor.nod.xml: junction node definitions for J1, J2, J3
- corridor.edg.xml: arterial and cross-street edge definitions
- corridor.net.xml: compiled SUMO network file
- corridor.sumocfg: simulation configuration entrypoint"
```

### Colab Notebook
```
git add IntelliLight_Colab_V3.ipynb
git commit -m "feat(training): add Google Colab training notebook v3 for cloud GPU training"
```

### Requirement File
```
git add requirements.txt
git commit -m "chore: add requirements.txt with pinned Python dependencies"
```

### Documentation Files
```
git add TRAINING_GUIDE.md DEBUG_CORRECTIONS.md intelli_light_analysis.md
git commit -m "docs: add training guide, debug corrections log, and system analysis"
```

### Architecture Documentation
```
git add parameter_sharing_architecture.md
git commit -m "docs: add ILPS scaling architecture design document

Documents how to transition from the current centralized PPO approach to
Independent Learning with Parameter Sharing for zero-shot N-intersection
scalability, including architecture diagrams and SB3 pseudocode."
```

---

## One-Shot Combined Commit (Alternative)

If you prefer a single commit for all source changes:
```
git add -A -- ':!*.pyc' ':!*/__pycache__/*'
git commit -m "feat: multi-agent corridor evaluation pipeline with real baselines and bug fixes

- Evaluate Fixed-Timer, Max-Pressure, and IntelliLight-RL on all scenarios
- Fix throughput key mismatch in evaluation engine
- Fix phase snapshot to correctly measure phase switches
- Baseline controllers now support N-intersection observation vectors
- Add cycle_time to CorridorEnv step info
- Add SUMO network configs, training notebook, requirements and documentation"
```
