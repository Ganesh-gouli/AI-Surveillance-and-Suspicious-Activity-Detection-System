import json
from typing import List, Dict, Set
from fastapi import WebSocket

class WebSocketConnectionManager:
    def __init__(self):
        # Active camera live channels: {camera_id: set of WebSockets}
        self.camera_channels: Dict[str, Set[WebSocket]] = {}
        # Global alert channels: set of WebSockets
        self.alert_subscribers: Set[WebSocket] = set()
        # Active webcam channels: set of WebSockets
        self.webcam_subscribers: Set[WebSocket] = set()

    async def connect_camera(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.camera_channels:
            self.camera_channels[camera_id] = set()
        self.camera_channels[camera_id].add(websocket)

    async def connect(self, websocket: WebSocket, channel: str = "default"):
        await websocket.accept()
        if channel not in self.camera_channels:
            self.camera_channels[channel] = set()
        self.camera_channels[channel].add(websocket)

    def disconnect_camera(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.camera_channels:
            self.camera_channels[camera_id].discard(websocket)
            if not self.camera_channels[camera_id]:
                del self.camera_channels[camera_id]

    def disconnect(self, websocket: WebSocket, channel: str = "default"):
        if channel in self.camera_channels:
            self.camera_channels[channel].discard(websocket)
            if not self.camera_channels[channel]:
                del self.camera_channels[channel]

    async def broadcast_camera(self, camera_id: str, data: dict):
        if camera_id in self.camera_channels:
            dead_sockets = set()
            message = json.dumps(data)
            for ws in self.camera_channels[camera_id]:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead_sockets.add(ws)
            for ws in dead_sockets:
                self.camera_channels[camera_id].discard(ws)

    async def connect_alerts(self, websocket: WebSocket):
        await websocket.accept()
        self.alert_subscribers.add(websocket)

    def disconnect_alerts(self, websocket: WebSocket):
        self.alert_subscribers.discard(websocket)

    async def broadcast_alert(self, alert_data: dict):
        dead_sockets = set()
        message = json.dumps(alert_data)
        for ws in self.alert_subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead_sockets.add(ws)
        for ws in dead_sockets:
            self.alert_subscribers.discard(ws)

    async def connect_webcam(self, websocket: WebSocket):
        await websocket.accept()
        self.webcam_subscribers.add(websocket)

    def disconnect_webcam(self, websocket: WebSocket):
        self.webcam_subscribers.discard(websocket)

    def get_total_connections(self) -> int:
        camera_conns = sum(len(conns) for conns in self.camera_channels.values())
        return camera_conns + len(self.alert_subscribers) + len(self.webcam_subscribers)

manager = WebSocketConnectionManager()
ws_manager = manager
