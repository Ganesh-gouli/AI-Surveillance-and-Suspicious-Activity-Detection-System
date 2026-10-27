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
  is_weapon?: boolean;
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

// =========================================================================
// Academic Report Results & Analysis Module Types (Sections 7.1 to 7.9)
// =========================================================================
export type ResultMode = 'demonstration' | 'experimental';

export interface ModelTrainingRecord {
  model_id: string;
  model_name: string;
  architecture: string;
  dataset_name: string;
  number_of_classes: number;
  class_names: string[];
  training_samples: number;
  validation_samples: number;
  test_samples: number;
  batch_size: number;
  epochs: number;
  optimizer: string;
  learning_rate: string;
  training_duration_sec: number;
  trainable_parameters: number;
  final_train_loss: number;
  final_val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
  test_accuracy: number;
  status: string;
}

export interface ClassMetricRecord {
  class_name: string;
  support: number;
  precision: number;
  recall: number;
  f1_score: number;
  specificity: number;
  correct_count: number;
  error_count: number;
}

export interface EvaluationMetricsSummary {
  accuracy: number;
  balanced_accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_score_macro: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_score_weighted: number;
  specificity: number;
  mcc: number;
  roc_auc: number;
  total_test_samples: number;
}

export interface EvaluationMetricsData {
  metadata: {
    result_source: ResultMode;
    model_name: string;
    dataset_name: string;
    dataset_split?: string;
    evaluation_timestamp: string;
    is_demonstration: boolean;
    evaluation_completed: boolean;
    disclaimer?: string;
  };
  summary: EvaluationMetricsSummary;
  classes: ClassMetricRecord[];
  surveillance_implication: string;
}

export interface ConfusionMatrixData {
  metadata: {
    result_source: ResultMode;
    model_name: string;
    dataset_name: string;
    class_names: string[];
    is_demonstration: boolean;
    disclaimer?: string;
  };
  classes: string[];
  matrix: number[][];
  row_normalized: number[][];
  total_samples: number;
  analysis: {
    correct_predictions: number;
    incorrect_predictions: number;
    primary_confusion: string;
    false_negative_rate: string;
    root_cause: string;
  };
}

export interface TrainingEpochRecord {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_accuracy: number;
  val_accuracy: number;
}

export interface TrainingCurvesData {
  metadata: {
    result_source: ResultMode;
    model_name: string;
    epochs: number;
    best_epoch: number;
    is_demonstration: boolean;
    disclaimer?: string;
  };
  history: TrainingEpochRecord[];
  observations: {
    convergence: string;
    best_epoch_metrics: string;
    overfitting_analysis: string;
  };
}

export interface EnvironmentalConditionRecord {
  condition: string;
  category: string;
  description: string;
  transformation_params?: string;
  accuracy: number;
  f1_score: number;
  delta_accuracy: number;
  robustness_score: number;
}

export interface HybridModelRecord {
  model_name: string;
  architecture_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  parameter_count: number;
  model_size_mb: number;
  inference_latency_ms: number;
  temporal_modeling: string;
}

export interface SystemInterfaceRecord {
  id: string;
  title: string;
  route: string;
  purpose: string;
  input: string;
  workflow: string;
  output: string;
  interaction: string;
}

export interface PerformanceBenchmarkModel {
  model_name: string;
  load_time_ms: number;
  mean_latency_ms: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  throughput_fps: number;
  preprocessing_ms?: number;
  inference_ms?: number;
  postprocessing_ms?: number;
  total_parameters: number;
  model_size_mb: number;
  memory_delta_mb?: number;
  cpu_utilization_percent?: number;
  realtime_viable: boolean;
}

export interface TraditionalModelRecord {
  model_name: string;
  feature_representation: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc?: number;
  training_duration_sec: number;
  inference_latency_ms: number;
  model_size_mb: number;
}
