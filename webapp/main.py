import sys
import json
import os
import asyncio
import glob
import subprocess
import time
from contextlib import asynccontextmanager
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import redis.asyncio as aioredis  # Using async Redis client to not block the event loop

# Optional GPU monitoring
try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Adjust path to point to root dir depending on where fastAPI is run from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "webapp", "static")

@asynccontextmanager
async def lifespan(app):
    """Application lifespan: start background tasks on startup."""
    asyncio.create_task(redis_reader())
    asyncio.create_task(hardware_poller())
    yield

app = FastAPI(title="Intelli-Light Web Architecture", lifespan=lifespan)

# Initialize Redis at app startup
redis_client = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Mount static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ── Connection Manager for WebSockets ──
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass # Client disconnected

manager = ConnectionManager()

# ── Training Management ──
class TrainingManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.start_time: float = 0
        self.status: str = "OFFLINE"

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    async def start(self, scenario: str = "WEEKEND"):
        if self.is_running():
            return False
        
        cmd = [sys.executable, os.path.join(ROOT_DIR, "training", "train_rl.py"), "--timesteps", "100000"]
        if scenario != "WEEKEND":
            # Map scenarios to stages if needed, for now just pass as example
            pass
            
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=ROOT_DIR
        )
        self.start_time = time.time()
        self.status = "TRAINING"
        asyncio.create_task(self._read_logs())
        return True

    async def stop(self):
        if not self.is_running():
            return False
        self.process.terminate()
        self.status = "STOPPED"
        return True

    async def _read_logs(self):
        """Forward subprocess logs to the dash via broadcast (non-blocking)."""
        def _blocking_readline():
            """Read one line from stdout in a thread."""
            return self.process.stdout.readline()

        while self.is_running():
            line = await asyncio.to_thread(_blocking_readline)
            if not line:
                break
            await manager.broadcast(json.dumps({
                "msg": line.strip(),
                "level": "info",
                "source": "TRAINING_CLI"
            }))
        
        if self.process and self.process.stdout:
            self.process.stdout.close()
        self.status = "COMPLETED"

training_manager = TrainingManager()

# ── Redis Reader Task ──
async def redis_reader():
    """Continuously poll the Redis Pub/Sub channel and broadcast to WS clients."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("intellilight_telemetry")
    
    print("[Intelli-Light Backend] Subscribed to intellilight_telemetry...")
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                await manager.broadcast(data)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Redis reader error: {e}")
    finally:
        await pubsub.unsubscribe()

# ── Hardware Polling Task ──
async def hardware_poller():
    """Poll system health and broadcast via WebSocket."""
    if HAS_GPU:
        try:
            pynvml.nvmlInit()
        except:
            pass

    while True:
        try:
            if HAS_PSUTIL:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
            else:
                # Mock values if psutil is missing
                cpu = 25.0 + (time.time() % 10) 
                ram = 45.0 + (time.time() % 5)

            hw_stats = {
                "cpu_usage": cpu,
                "ram_usage": ram,
                "uptime": f"{int((time.time() - start_time_app) / 3600)}h {int((time.time() - start_time_app) % 3600 / 60)}m",
                "latency": 12, # Placeholder for inference latency
                "gpu": None
            }

            if HAS_GPU:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    hw_stats["gpu"] = {
                        "load": util.gpu,
                        "temp": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                        "vram": int(info.used / info.total * 100)
                    }
                except:
                    pass
            
            await manager.broadcast(json.dumps({"hardware": hw_stats}))
        except Exception as e:
            print(f"Hardware poller error: {e}")
        
        await asyncio.sleep(2)

start_time_app = time.time()

# ── API Endpoints ──

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """Clients connect here to receive the telemetry stream."""
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect inbound messages on this socket (using REST for commands)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class CommandPayload(BaseModel):
    type: str
    value: str | bool | int

@app.post("/api/command")
async def push_command(cmd: CommandPayload):
    """API endpoint capturing UI interactions and pushing them to Redis queue."""
    try:
        # Handle specific Admin commands before pushing to Redis
        if cmd.type == "START_TRAINING":
            success = await training_manager.start(scenario=str(cmd.value))
            return {"status": "success" if success else "error", "msg": "Training session initiated" if success else "Already running"}
        
        if cmd.type == "PAUSE_TRAINING":
            success = await training_manager.stop()
            return {"status": "success" if success else "error", "msg": "Training stopped"}

        # For simulation-level overrides, push to Redis
        await redis_client.rpush('intellilight_commands', cmd.model_dump_json())
        return {"status": "success", "command": cmd.model_dump()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to execute command: {str(e)}"})

@app.get("/api/models")
async def list_models():
    """Finds all .zip files in the models directory for hot-swapping."""
    checkpoints_path = os.path.join(ROOT_DIR, "models", "checkpoints", "*.zip")
    files = glob.glob(checkpoints_path)
    # Extract just the basenames
    basenames = [os.path.basename(f) for f in files]
    return {"models": basenames}

@app.get("/api/results")
async def get_historical():
    """Read the final_results.json and format it for the frontend."""
    results_path = os.path.join(ROOT_DIR, "final_results.json")
    if not os.path.exists(results_path):
        return {"error": "final_results.json not found. Run training/evaluation mode first."}
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    # We will format this into a shape that is easy for Chart.JS to consume.
    # Specifically, returning avg_wait_time, throughput, max_queue_length for the 3 algorithms
    # under the current default 'WEEKEND' scenario. 
    
    # Alternatively, simply return the raw data and let the frontend do data-wrangling.
    return data

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this using `cd demo_intellilight && uvicorn webapp.main:app --reload`
    uvicorn.run("webapp.main:app", host="0.0.0.0", port=8000, reload=True)
