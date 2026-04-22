# Intelli-Light / CorridorEnv Redis Integration Snippets

You will need to install Redis for Python via `pip install redis`.
Open `rl/multi_agent_env.py` and add the following snippets to integrate your RL training pipeline with the Web Layer in a non-blocking way.

### 1. Initialise the Redis Client
Add this near the end of your `CorridorEnv.__init__` method.

```python
import redis
import json

# Non-blocking Redis connection
try:
    # Use a fast timeout to never hang the RL loop if Redis drops
    self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_timeout=0.05)
    self.redis_client.ping()
except Exception as e:
    self.redis_client = None
    print(f"Failed to connect to Redis middleman: {e}")
```

---

### 2. The Command Queue Polling
Inject this snippet at the top of your `CorridorEnv.step(self, action)` method, directly before you decode `action` into durations.

```python
        # ── Poll Web layer commands (Inbound) ─────────────────────────────────
        if self.redis_client:
            try:
                # Process all commands currently in the queue
                while True:
                    cmd_str = self.redis_client.lpop('intellilight_commands')
                    if not cmd_str:
                        break
                    
                    cmd = json.loads(cmd_str)
                    cmd_type = cmd.get("type")
                    cmd_val = cmd.get("value")
                    
                    if cmd_type == "SCENARIO":
                        print(f"[Web Command] Scenario Hot-Swap requested: {cmd_val}")
                        # Example: Override the next reset scenario curriculum
                        # self.curriculum_stage = {"WEEKEND": 0, "EVENING_RUSH": 1, "MORNING_RUSH": 2}[cmd_val]
                        
                    elif cmd_type == "EVENT":
                        print(f"[Web Command] Event Injection: {cmd_val}")
                        # Example logic to immediately spawn an ambulance in SUMO
                        
                    elif cmd_type == "FAILSAFE":
                        print(f"[Web Command] Failsafe Triggered: override={cmd_val}")
                        # Example: flip a flag (self.failsafe_active = True), when True, ignore `action` and force a Fixed-Timer logic.
                        
                    elif cmd_type == "LOAD_MODEL":
                        print(f"[Web Command] Model swap requested to {cmd_val}")
                        
            except Exception as e:
                # Ignore transient Redis errors
                pass
```

---

### 3. The Telemetry Publisher
Inject this snippet at the end of your `CorridorEnv.step` method, right before you `return obs, avg_reward, terminated, False, info`.

```python
        # ── Publish Telemetry (Outbound) ──────────────────────────────────────
        if self.redis_client:
            try:
                # Calculate aggregated waiting time
                total_waits = [wait for s in states.values() for wait in s["wait_times"]]
                avg_wait = float(np.mean(total_waits)) if total_waits else 0.0
                
                # Calculate total queue length across all junctions
                total_queue = sum(sum(s["queues"]) for s in states.values())
                
                payload = {
                    "cycle": self.cycle_count,
                    "throughput": self.cumulative_arrived,
                    "total_queue": total_queue,
                    "avg_wait": avg_wait,
                    "active_phases": self.current_phases,
                    "intersections": states
                }
                
                # Publish JSON string to the telemetry channel
                self.redis_client.publish('intellilight_telemetry', json.dumps(payload))
            except Exception:
                pass
```
