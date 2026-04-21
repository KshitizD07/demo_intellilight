# Intelli-Light: Step-by-Step Training Guide
**For: Local machine (Asus TUF A15 / RTX 3050 / 16 GB RAM)**

---

## Prerequisites

Make sure you are in the project directory:
```
cd c:\Users\kshit\cs\project\Intelli-Light
```

All commands use `main.py` as the single entry point.

---

## Training Philosophy

Each "step" in training is one full corridor cycle (~100-200 simulation seconds).
At ~8 FPS with SubprocVecEnv, you get roughly 480 steps/minute.

The guide below splits training into **5 phases** of increasing length.
After each phase, you can evaluate results and decide whether to continue.

---

## Phase 1: Smoke Test (2 minutes)

Verify everything works. This trains just enough to run one PPO update.

```bash
python main.py train --steps 2048 --envs 2 --subproc --device cpu
```

**What to check:**
- Should complete without errors
- Should print `TRAINING COMPLETE` and `No safety violations detected!`
- Model saved to `models/checkpoints/intellilight_corridor_final.zip`

**Expected output:** `ep_rew_mean` around 1000-1500 (random policy baseline)

---

## Phase 2: Short Training (10 minutes)

Train enough to see the reward start moving. Uses `--resume` to continue
from where Phase 1 left off.

```bash
python main.py train --steps 10000 --envs 2 --subproc --device cpu
```

**What to check:**
- `ep_rew_mean` should be changing (up or down — the agent is exploring)
- `fps` should be ~8 with 2 envs

**Optional — speed up with 4 envs** (uses more RAM but ~2x faster):
```bash
python main.py train --steps 10000 --envs 4 --subproc --device cpu
```

---

## Phase 3: Medium Training (30 minutes)

This is where real learning starts. The curriculum will still be on Stage 0
(WEEKEND — easy traffic).

```bash
python main.py train --steps 50000 --envs 4 --subproc --device cpu
```

**What to check:**
- `ep_rew_mean` should be trending upward
- Checkpoints saved at 25K and 50K steps in `models/checkpoints/`
- Look for `intellilight_corridor_25000_steps.zip`

---

## Phase 4: Curriculum Training (1–2 hours)

This trains through all three curriculum stages:
- 0-66K: WEEKEND (light traffic)
- 66K-133K: EVENING_RUSH (moderate)
- 133K-200K: MORNING_RUSH (heavy)

```bash
python main.py train --steps 200000 --envs 4 --subproc --device cpu
```

**What to check:**
- You should see `CURRICULUM STAGE 1: EVENING_RUSH` print at ~66K steps
- And `CURRICULUM STAGE 2: MORNING_RUSH` at ~133K steps
- Checkpoints saved every 25K steps
- Final model: `intellilight_corridor_final.zip`

---

## Phase 5: Extended Training (4–6 hours, optional)

For maximum performance. Only do this if Phase 4 results look promising.

```bash
python main.py train --steps 1000000 --envs 4 --subproc --device cpu
```

**Tip:** You can safely Ctrl+C at any time — the final model is saved
in the `finally` block even on interruption.

---

## Evaluating Results

### Quick visual check (SUMO GUI)

```bash
python main.py test-gui --model models/checkpoints/intellilight_corridor_final.zip
```

This opens SUMO's GUI and runs one episode with the trained model.
Watch for green-wave coordination along the corridor.

Add `--plot` for matplotlib graphs:
```bash
python main.py test-gui --model models/checkpoints/intellilight_corridor_final.zip --plot
```

### Compare checkpoints

You can test any checkpoint saved during training:
```bash
python main.py test-gui --model models/checkpoints/intellilight_corridor_25000_steps.zip
python main.py test-gui --model models/checkpoints/intellilight_corridor_50000_steps.zip
```

### TensorBoard monitoring

To monitor training in real-time (run in a separate terminal):
```bash
tensorboard --logdir logs/tensorboard
```
Then open http://localhost:6006 in your browser.

---

## Quick Reference

| Goal | Command | Time |
|------|---------|------|
| Smoke test | `python main.py train --steps 2048 --envs 2 --subproc --device cpu` | ~2 min |
| Quick train | `python main.py train --steps 10000 --envs 4 --subproc --device cpu` | ~10 min |
| Medium train | `python main.py train --steps 50000 --envs 4 --subproc --device cpu` | ~30 min |
| Full curriculum | `python main.py train --steps 200000 --envs 4 --subproc --device cpu` | ~1-2 hr |
| Max performance | `python main.py train --steps 1000000 --envs 4 --subproc --device cpu` | ~4-6 hr |
| Visualize | `python main.py test-gui --plot` | ~2 min |

---

## Notes

- **`--subproc`** enables true multiprocessing. Remove it if you get pipe errors
  (unlikely on local Windows — this is mainly a Colab issue).
- **`--device cpu`** is recommended. GPU (cuda) does NOT help for small MLP policies
  and adds IPC overhead.
- **`--envs 4`** is the sweet spot for your TUF A15. Going to 8 will thrash RAM.
- Each training run **overwrites** `intellilight_corridor_final.zip`. Numbered
  checkpoints (`_25000_steps.zip`, etc.) are preserved.
- Training is **not** resumable across runs by default. Each `train` command
  starts a fresh model. To resume, modify `main.py` to pass `resume_from=`.
