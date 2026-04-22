import json
import os
import asyncio
import glob
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import redis.asyncio as aioredis  # Using async Redis client to not block the event loop

# Adjust path to point to root dir depending on where fastAPI is run from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "webapp", "static")

app = FastAPI(title="Intelli-Light Web Architecture")

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

@app.on_event("startup")
async def startup_event():
    # Start the background task tracking Redis stream
    asyncio.create_task(redis_reader())

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
    # Push as a robust JSON string onto the end of the List
    try:
        await redis_client.rpush('intellilight_commands', cmd.json())
        return {"status": "success", "command": cmd.dict()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to reach Redis: {str(e)}"})

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
