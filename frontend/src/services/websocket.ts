type MessageCallback = (data: any) => void;

export class LiveCameraWebSocket {
  private ws: WebSocket | null = null;
  private camera_id: string;
  private onMessage: MessageCallback;
  private isClosedManually = false;
  private reconnectTimer: any = null;

  constructor(camera_id: string, onMessage: MessageCallback) {
    this.camera_id = camera_id;
    this.onMessage = onMessage;
    this.connect();
  }

  private connect() {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      this.ws = new WebSocket(`${protocol}//${host}/ws/live/${this.camera_id}`);

      this.ws.onopen = () => {
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.onMessage(data);
        } catch (e) {}
      };

      this.ws.onerror = () => {
        this.scheduleReconnect();
      };

      this.ws.onclose = () => {
        if (!this.isClosedManually) {
          this.scheduleReconnect();
        }
      };
    } catch (e) {
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer || this.isClosedManually) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 2000);
  }

  public close() {
    this.isClosedManually = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
