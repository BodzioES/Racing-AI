"""FastAPI entrypoint.

Each browser tab opens one WebSocket connection to /ws. That connection
gets its own World (own track, own population, own generations) - two
people can use the demo at once without sharing state. The server ticks
the simulation on a fixed-rate loop and streams frames as JSON; the
browser only draws what it receives.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .config import TICK_HZ
from .world import World

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-racers")

app = FastAPI(title="AI Racers")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    world = World()
    speed_mult = 3.0
    paused = False
    tick_dt = 1.0 / TICK_HZ

    async def receive_loop():
        nonlocal speed_mult, paused
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "track":
                    points = np.asarray(msg["points"], dtype=np.float64)
                    width = float(msg.get("width", 70))
                    w = int(msg.get("w", 1280))
                    h = int(msg.get("h", 720))
                    if len(points) >= 24:
                        world.set_track(points, width, w, h)
                        await ws.send_text(json.dumps({
                            "type": "track_ready",
                            "centerline": world.track.centerline.tolist(),
                            "width": world.track.width,
                            "startAngle": world.track.start_angle,
                        }))
                    else:
                        await ws.send_text(json.dumps({"type": "track_too_short"}))
                elif kind == "speed":
                    speed_mult = max(1.0, min(20.0, float(msg.get("value", 3))))
                elif kind == "pause":
                    paused = bool(msg.get("value", False))
                elif kind == "reset":
                    if world.track is not None:
                        world.reset_evolution()
        except WebSocketDisconnect:
            pass

    receiver = asyncio.create_task(receive_loop())
    try:
        while True:
            await asyncio.sleep(tick_dt)
            if world.track is None or paused:
                continue
            steps_this_tick = 0
            budget = speed_mult
            while budget >= 1 and steps_this_tick < 80:
                world.tick()
                budget -= 1
                steps_this_tick += 1
            await ws.send_text(json.dumps(world.frame()))
    except WebSocketDisconnect:
        pass
    except Exception:  # keep one bad frame from killing the whole connection
        log.exception("ai-racers: error in simulation loop")
    finally:
        receiver.cancel()


# Serve the frontend last, so /ws and /health above take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
