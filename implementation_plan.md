# 3x3 Multi-Agent RL Grid Implementation Plan

This document outlines the proposed transition from the current 3-intersection centralised corridor to a 3x3 True Multi-Agent Grid (9 intersections) using Reinforcement Learning, capable of behaving like decentralised scheduling systems (e.g., SURTRAC). 

## User Review Required
> [!IMPORTANT]
> Please review the design tradeoffs below, particularly regarding how cycles and lock-step constraints are modelled across 9 intersections, as this impacts both the emergency vehicle priority behaviour and the complexity.

## Proposed Changes

### 1. Grid Network & Scenario Generation (Simulation)
**Files affected:** `simulation/route_generator.py`, `configs/sumo/*`
- Generate `grid3x3.net.xml` comprising 9 intersections (e.g., J0_0 to J2_2) using SUMO's `netgenerate` tool (or automated XML generation) to ensure uniform distances.
- Update `route_generator.py` to route traffic organically from North/South/East/West entry points straight through the grid, adding turning probabilities at intersections.
- **Emergency Vehicles (EV):** EVs will spawn at fringes. We will broadcast their presence to *all* intersections along their intended shortest-path route, allowing the grid to proactively clear queues ("green wave" for ambulances).

### 2. Multi-Agent Reinforcement Learning (MARL) Architecture
**Files affected:** `rl/marl_env.py` (New), `rl/agent_communication.py`
- Move from Centralised PPO action-spaces to **Independent PPO (IPPO) with Parameter Sharing**.
- **Observation Space:** Each agent views its own intersection (queue, wait, current phase, EV incoming) **AND** the condensed state (queue lengths, current phase) of its 4 immediate neighbours (N, S, E, W).
- **Coordination Mechanism:** Giving agents their neighbours' states is how they learn to anticipate incoming platoons and "coordinate" without arbitrary rule-engine overrides.
- **Action Space:** Discrete duration choices `[10, 15, ..., 45]s` for the current phase only.

---

## 🏗️ Trade-offs for User Approval

> [!WARNING]
> Moving to a 3x3 grid introduces fundamental design decisions. Please approve or modify the following trade-offs:

1. **Strict Lock-step vs. Asynchronous Scheduling**
   - *Current Corridor:* All intersections advance phases simultaneously. The slowest agent holds up the cycle. This forces an easy-to-learn green wave.
   - *Proposed 3x3 Grid:* **Asynchronous Cycles**. Agents run independently. When J0_0 finishes Phase 0, it moves to Phase 1 immediately, regardless of what J1_1 is doing. 
   - *Trade-off:* Asynchronous allows much higher throughput (like Surtrac) but breaks the guarantee of perfect parallel cycles. *I recommend Asynchronous to challenge Surtrac.*
2. **Cyclic vs. Acyclic Phase Rotation**
   - *Current:* Phases run strictly 0->1->2->3->0 (Cyclic).
   - *Proposed 3x3:* Maintain strict Cyclic rotation to ensure "safety constraints" and prevent starvation (no phase is skipped). The RL agent only decides the *duration* of the phase, not the order. *This preserves your safety constraint.*
3. **Reward Sharing**
   - *Trade-off:* Giving an agent reward solely for its own intersection causes selfish behaviour (dumping traffic onto neighbors). We will use a **blended reward**: `0.7 * (Local Reward) + 0.3 * (Avg Neighbor Reward)`. This forces cooperation.

---

## 📡 Edge Device & Latency Simulation

You asked: *"In real time we'll have to use an edge device at each intersection for low latency, can we simulate that part as well?"*

**Yes! We will introduce native "Constraint Simulation" into the RL Environment:**

1. **Simulated Communication Latency / Jitter:**
   - When Agent A requests the queue status of Neighbor B, the environment will artificially provide the state from $t-N$ seconds ago (simulating V2I or Fiber latency). We can randomize this delay to train the RL to be robust to network drops.
2. **Simulated Edge Compute Delay (Action Delay):**
   - We will enforce an `action_delay` penalty. The model uses observations at step $t$ but the action physicalizes in SUMO at step $t + K$ (e.g., $K=1$ second) simulating model inference time on low-power ARM/Edge TPU devices.
3. **Model Quantization & FLOPS Tracking:**
   - We will purposefully restrict the MLP size (e.g., lightweight `[64, 64]` feature extractors) and track the PyTorch execution time per step. If an action trace takes > 100ms, the simulation logs an "Edge Compute Violation".

## Verification Plan

### Automated Tests
- Run `evaluate_model.py` against the 3x3 grid using fixed-timer baselines to establish a control floor.
- Confirm EVs lower standard traffic throughput temporarily but achieve 0 queue stops when crossing the grid compared to the control.

### Analytical Verification
- Output tensorboard correlation charts assessing if neighbouring agents "sync" their green phases when inter-intersection traffic density is high.
