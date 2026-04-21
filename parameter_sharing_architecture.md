# Scaling Intelli-Light: Independent Learning with Parameter Sharing (ILPS)

To scale the Intelli-Light reinforcement learning system to manage large-scale grids (e.g., 50 to 150+ intersections) without retraining, we must transition from the current **Centralized Architecture** to **Independent Learning with Parameter Sharing (ILPS)**. 

This document breaks down the concepts, outlines the architectural shift, and provides a concrete implementation guide.

---

## 1. Core Concept: What is Parameter Sharing?

In a traffic light network, most intersections follow the exact same physics and logic: they have queued vehicles, wait times, and traffic phases. Instead of treating the entire city grid as a single unique problem, we can treat *every intersection as an independent experience* to teach a single, universal traffic controller.

**Independent Learning with Parameter Sharing** means:
1. We instantiate **one** neural network "Brain".
2. This brain is designed strictly to control **just one intersection** (taking local queues as inputs and outputting green times for that local node).
3. During the simulation, all 3 (or 9, or 100) intersections query this *exact same network* for their next move.
4. All intersections gather their rewards and state transitions and pool them together into a massive shared buffer.
5. The network is then trained on this pooled data.

When you want to scale up from a 3-intersection corridor to a 9-intersection grid, you simply spawn 9 instances of the same Brain. The network size doesn't change, meaning **zero-shot scalability without retraining.**

---

## 2. Architecture Comparison

### Current Approach: Centralized 
Currently, Intelli-Light uses a generalized PPO model where everything goes into one giant pipe. If you add one intersection, the input size changes from 42 to 56, instantly breaking the neural network.

```mermaid
graph TD
    classDef brain fill:#ff9999,stroke:#333,stroke-width:2px;
    classDef node fill:#99ccff,stroke:#333,stroke-width:2px;

    subgraph Centralized Architecture (Current)
        O[Global Obs: 42 dims] --> NN[Centralized PPO Brain]:::brain
        NN --> A[Global Action: 12 dims]
        
        A -.-> I1[Intersection 1]:::node
        A -.-> I2[Intersection 2]:::node
        A -.-> I3[Intersection 3]:::node
    end
```

### Proposed Approach: Parameter Sharing
Using Parameter Sharing, the brain processes localized chunks. The intersections act locally but share their knowledge globally.

```mermaid
graph TD
    classDef brain fill:#99ff99,stroke:#333,stroke-width:2px;
    classDef node fill:#99ccff,stroke:#333,stroke-width:2px;

    subgraph Parameter Sharing Architecture (Proposed)
        NN((Shared PPO Brain)):::brain
        
        O1[Obs 1: 14 dims] --> NN
        O2[Obs 2: 14 dims] --> NN
        O3[Obs 3: 14 dims] --> NN
        
        NN --> A1[Action 1] -.-> I1[Intersection 1]:::node
        NN --> A2[Action 2] -.-> I2[Intersection 2]:::node
        NN --> A3[Action 3] -.-> I3[Intersection 3]:::node
        
        %% Training Feedback Loop
        I1 -. Data Pool .-> NN
        I2 -. Data Pool .-> NN
        I3 -. Data Pool .-> NN
    end
```

---

## 3. Implementation Guide: Migrating Intelli-Light

To implement ILPS in your existing codebase, you will need to update three main layers: The Environment, The Neural Network Wrapper, and the Reward Structure. 

### Step 1: Converting the Gym Environment to a Multi-Agent Format
Instead of outputting a single flat array of size `N * 14`, the environment must output a batch (or array) of arrays. If you use the highly compatible library like `PettingZoo` or Subproc VecEnvs, you batch local states.

**Changes required in `rl/multi_agent_env.py`:**
- **Observation Space**: Revert it from `Box(14 * n)` back to `Box(14)`.
- **Action Space**: Revert it from `MultiDiscrete([8]*12)` back to `MultiDiscrete([8]*4)`.
- **Step Function Return**: Instead of `obs` being size `42`, it should return a dictionary of local observations:
```python
# Rather than combining them:
# obs = np.concatenate(obs_parts) 

# Return independent agent views:
obs = {
    "J1": self._build_obs_for_junction("J1", states["J1"]),
    "J2": self._build_obs_for_junction("J2", states["J2"]),
    "J3": self._build_obs_for_junction("J3", states["J3"])
}
```

### Step 2: Incorporating Neighbor Communication (Green Waves)
In an isolated setup, an intersection doesn't know what the neighboring node is doing, which breaks "green wave" traffic coordination. 

To fix this, you expand the `14-dimension` input space slightly locally (e.g. to `16`), taking in the current phase of the upstream and downstream neighbors.
* **New Observation Input:** Local Queue [4], Local Wait [4], Local Delta [4], **Upstream Phase [1]**, **Downstream Phase [1]**, Local Phase [1].

### Step 3: Modifying the SB3 PPO Training Loop
Standard Stable Baselines 3 (SB3) models only accept flat vectors. Since you are using SB3, the cleanest way to do ILPS is to **flatten the multi-agent problem along the environment batch dimension**. 

When stepping the model, instead of feeding 1 environment with 1 agent, you feed 1 environment as if it is $N$ separate environments.

```python
# Pseudocode for Parameter Sharing with SB3
for step in range(MAX_STEPS):
    # 'obs_dict' contains states for J1, J2, and J3
    # Flatten the dict into a batch of size [3, 14]
    batched_obs = np.array([obs_dict["J1"], obs_dict["J2"], obs_dict["J3"]])
    
    # Pass the batch of 3 intersections through the SAME Brain model 
    batched_actions, _ = shared_ppo_model.predict(batched_obs)
    
    # Map actions back to their respective intersections
    action_dict = {
        "J1": batched_actions[0],
        "J2": batched_actions[1],
        "J3": batched_actions[2],
    }
    
    # Execute step over all 3
    obs_dict, rewards_dict, done, info = env.step(action_dict)
    
    # The PPO roll-out buffer accepts these 3 experiences as 3 separate 
    # steps learned by the same generic intersection model.
```

---

## 4. Key Advantages of the ILPS Model

1. **Zero-Shot Transfer:** Once training is completed on a 3-intersection corridor, you can deploy the trained model directly on an unlimited `N x N` grid without running the training script again.
2. **Speed Multiplier:** Because every step in your simulation aggregates data from 3 intersections into the buffer, the PPO algorithm learns literally **3x faster**. Simulating a 150-grid junction means one single simulation second generates 150 seconds worth of training experience.
3. **Robust Generalization:** The brain learns universal truths. If 'Junction A' sees a traffic pattern that 'Junction B' doesn't, they BOTH still learn how to handle it because they share the same neural weights.

## What's Next?
If you'd like to implement this approach, we will transition your existing `CorridorEnv` into an AEC (Agent-Environment-Cycle) or Parallel format utilizing the industry-standard `PettingZoo` library, which seamlessly acts as an adapter into Stable Baselines 3.
