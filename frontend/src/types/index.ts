export type SeverityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type IncidentStatus = 'OPEN' | 'INVESTIGATING' | 'RESOLVED';
export type CameraStatus = 'ONLINE' | 'OFFLINE' | 'CONNECTING' | 'ERROR';
export type LayoutGridMode = '1x1' | '2x2' | '3x3' | 'focus';

export type ActivityClass = 
  | 'sitting'
  | 'standing'
  | 'cycling'
  | 'drinking'
  | 'eating'
  | 'fighting'
  | 'running'
  | 'sleeping'
  | 'using_laptop'
  | 'using_phone'
  | 'carrying_bag'
  | 'playing_sports'
  | string;

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  label: string;
  category: string;
  confidence: number;
  bbox: BoundingBox;
}

export interface PoseLandmark {
  id: number;
  name: string;
  x: number;
  y: number;
  pixel_x: number;
  pixel_y: number;
  z: number;
  visibility: number;
}

export interface PoseAngles {
  left_knee?: number | null;
  right_knee?: number | null;
  left_hip?: number | null;
  right_hip?: number | null;
  spine_incline?: number | null;
}

export interface TrackedPerson {
  track_id: number;
  bbox: BoundingBox;
  activity: string;
  confidence: number;
  interacting_objects?: DetectedObject[];
  landmarks?: PoseLandmark[];
  pose_angles?: PoseAngles;
  posture_details?: string;
  is_suspicious: boolean;
  threat_level: SeverityLevel | 'NORMAL';
  duration_sec: number;
}

export interface Camera {
  id: string;
  name: string;
  stream_url: string;
  location: string;
  camera_type: string;
  status: CameraStatus;
  fps: number;
  resolution: string;
  ai_confidence: number;
  people_detected: number;
  active_threats: number;
  sensitivity: number;
  is_recording: boolean;
  created_at?: string;
}

export interface ActivityTimelineItem {
  activity: string;
  confidence: number;
  time_offset_sec: number;
  is_suspicious: boolean;
  interacting_objects?: string[];
}

export interface Incident {
  id: string;
  camera_id: string;
  timestamp: string;
  location: string;
  activity: string;
  severity: SeverityLevel;
  confidence: number;
  status: IncidentStatus;
  person_track_id?: number;
  details?: string;
  ai_reasoning?: string;
  snapshot_url?: string;
  video_clip_url?: string;
  resolved_by?: string;
  resolution_notes?: string;
  created_at?: string;
  events?: ActivityTimelineItem[];
}

export interface Point {
  x: number;
  y: number;
}

export interface RestrictedZone {
  id: string;
  camera_id: string;
  name: string;
  polygon: Point[];
  severity: SeverityLevel;
  alert_type: string;
  is_active: boolean;
  color: string;
  schedule: string;
  created_at?: string;
}

export interface VideoAnalysisJob {
  video_id: string;
  filename: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress_percentage: number;
  frames_processed: number;
  total_frames: number;
  duration_sec: number;
  fps: number;
  people_detected_total: number;
  suspicious_events_count: number;
  dominant_activity: string;
  ai_confidence: number;
  timeline: ActivityTimelineItem[];
  incidents_generated: string[];
}

export interface SystemMetrics {
  ai_engine_status: string;
  model_mode: 'REAL_AI' | 'DEMO_SIMULATION';
  connected_cameras: number;
  total_cameras: number;
  total_people_detected: number;
  active_alerts: number;
  today_incidents: number;
  avg_ai_confidence: number;
  gpu_usage_percent: number;
  cpu_usage_percent: number;
  ram_usage_percent: number;
  inference_fps: number;
  websocket_clients: number;
}

export interface AIModelInfo {
  name: string;
  version: string;
  architecture: string;
  classes: string[];
  test_accuracy: number;
  status: string;
  model_path: string;
  framework: string;
}
