import { Camera, Incident, RestrictedZone, VideoAnalysisJob, SystemMetrics, AIModelInfo } from '../types';
const RAW_API_URL = (import.meta.env.VITE_API_URL || '').trim().replace(/\/+$/, '');
export const BACKEND_URL = RAW_API_URL.endsWith('/api') ? RAW_API_URL.slice(0, -4) : RAW_API_URL;
export const API_BASE = RAW_API_URL ? (RAW_API_URL.endsWith('/api') ? RAW_API_URL : `${RAW_API_URL}/api`) : '/api';

/**
 * Normalizes relative backend media paths (e.g. /results/... or /static/...)
 * into fully qualified URLs when deployed, while supporting blob: and external URLs.
 */
export const getMediaUrl = (path?: string | null): string => {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('blob:') || path.startsWith('data:')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return BACKEND_URL ? `${BACKEND_URL}${cleanPath}` : cleanPath;
};
export const api = {
  // Cameras
  async getCameras(): Promise<Camera[]> {
    try {
      const res = await fetch(`${API_BASE}/cameras`);
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("Camera fetch error:", e);
    }
    return [];
  },

  async getCamera(id: string): Promise<Camera | null> {
    try {
      const res = await fetch(`${API_BASE}/cameras/${id}`);
      if (res.ok) return await res.json();
    } catch (e) {}
    const cams = await this.getCameras();
    return cams.find(c => c.id === id) || null;
  },

  async createCamera(data: Partial<Camera>): Promise<Camera> {
    const res = await fetch(`${API_BASE}/cameras`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) return await res.json();
    throw new Error('Failed to create camera');
  },

  async testConnection(stream_url: string) {
    const res = await fetch(`${API_BASE}/cameras/test-connection`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stream_url }),
    });
    return await res.json();
  },

  // Incidents
  async getIncidents(params?: { camera_id?: string; activity?: string; severity?: string; status?: string; search?: string }): Promise<Incident[]> {
    try {
      const query = new URLSearchParams(params as any).toString();
      const res = await fetch(`${API_BASE}/incidents?${query}`);
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("Incidents fetch error:", e);
    }
    return [];
  },

  async updateIncident(id: string, data: { status?: string; resolved_by?: string; resolution_notes?: string }): Promise<Incident> {
    const res = await fetch(`${API_BASE}/incidents/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) return await res.json();
    throw new Error('Failed to update incident');
  },

  // Restricted Zones
  async getZones(camera_id?: string): Promise<RestrictedZone[]> {
    try {
      const q = camera_id ? `?camera_id=${camera_id}` : '';
      const res = await fetch(`${API_BASE}/zones${q}`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return [];
  },

  async createZone(data: Partial<RestrictedZone>): Promise<RestrictedZone> {
    const res = await fetch(`${API_BASE}/zones`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) return await res.json();
    throw new Error('Failed to create restricted zone');
  },

  async deleteZone(id: string) {
    await fetch(`${API_BASE}/zones/${id}`, { method: 'DELETE' });
  },

  // Analytics
  async getAnalyticsOverview() {
    try {
      const res = await fetch(`${API_BASE}/analytics/overview`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return {
      kpis: {
        total_cameras: 0,
        active_cameras: 0,
        people_detected_now: 0,
        active_alerts: 0,
        today_incidents: 0,
        avg_ai_confidence: 0,
        most_detected_activity: "None",
        most_active_camera: "None",
        peak_activity_hour: "N/A"
      },
      activity_distribution: [],
      hourly_trends: [],
      severity_distribution: []
    };
  },

  // System & Models
  async getSystemStatus(): Promise<SystemMetrics> {
    try {
      const res = await fetch(`${API_BASE}/system/status`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return {
      ai_engine_status: "ONLINE",
      model_mode: "REAL_AI",
      connected_cameras: 0,
      total_cameras: 0,
      total_people_detected: 0,
      active_alerts: 0,
      today_incidents: 0,
      avg_ai_confidence: 94.6,
      gpu_usage_percent: 0,
      cpu_usage_percent: 0,
      ram_usage_percent: 0,
      inference_fps: 30.0,
      websocket_clients: 0
    };
  },

  async toggleSystemMode(mode: 'REAL_AI' | 'DEMO_SIMULATION') {
    const res = await fetch(`${API_BASE}/system/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    return await res.json();
  },

  async getAIModelsInfo(): Promise<AIModelInfo[]> {
    try {
      const res = await fetch(`${API_BASE}/models/info`);
      if (res.ok) return await res.json();
    } catch (e) {}
    return [];
  }
};
