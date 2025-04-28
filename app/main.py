from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import redis
import json
import asyncio

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Redis 연결
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("웹소켓 연결됨")

    last_sent = None

    try:
        while True:
            pose_data = r.get("latest_pose")
            if pose_data and pose_data != last_sent:
                await websocket.send_text(pose_data)
                last_sent = pose_data
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("웹소켓 연결 종료됨")