import React, { useState, useEffect, useRef } from 'react';
import {
  Grid,
  Maximize2,
  Minimize2,
  Volume2,
  VolumeX,
  Camera as CamIcon,
  AlertTriangle,
  Play,
  Square,
  Shield,
  Layers,
  Zap,
  Sliders,
  Crosshair,
  Eye,
  Activity,
  CheckCircle,
  Radio,
  Flame,
  ShieldAlert,
  Moon,
  Sun,
  Video
} from 'lucide-react';
import { api, API_BASE, getMediaUrl } from '../services/api';
import { Camera, TrackedPerson, DetectedObject } from '../types';
import { CameraCanvasOverlay } from '../components/video/CameraCanvasOverlay';
import { audioService } from '../services/audioAlerts';

export const LiveMonitoring: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedLayout, setSelectedLayout] = useState<'1x1' | '2x2' | '3x3' | 'FOCUS'>('2x2');
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-02');
  const [cameraDetections, setCameraDetections] = useState<Record<string, TrackedPerson[]>>({});
  const [cameraObjects, setCameraObjects] = useState<Record<string, DetectedObject[]>>({});
  const [activeAlerts, setActiveAlerts] = useState<any[]>([]);

  // High-Accuracy AI Neural Controls
  const [accuracyMode, setAccuracyMode] = useState<'ultra' | 'high' | 'balanced' | 'fast'>('high');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(30);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [showObjects, setShowObjects] = useState<boolean>(true);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [latency, setLatency] = useState<number>(0);
  const [webcamActive, setWebcamActive] = useState<boolean>(false);

  // Refs for low-latency inference loop
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const accuracyModeRef = useRef(accuracyMode);
  accuracyModeRef.current = accuracyMode;
  const confidenceThresholdRef = useRef(confidenceThreshold);
  confidenceThresholdRef.current = confidenceThreshold;
  const selectedCameraIdRef = useRef(selectedCameraId);
  selectedCameraIdRef.current = selectedCameraId;
  const webcamStreamRef = useRef<MediaStream | null>(null);
  const lastAlertTimeRef = useRef<number>(0);

  // 1. Fetch camera configuration on mount
  useEffect(() => {
    const fetchCams = async () => {
      try {
        const cams = await api.getCameras();
        if (cams && cams.length > 0) {
          setCameras(cams);
          // Default to first active CCTV feed (e.g. CAM-02)
          const firstCctv = cams.find(c => c.id !== 'CAM-01') || cams[0];
          setSelectedCameraId(firstCctv.id);
        }
      } catch (e) {
        console.error("Error fetching cameras:", e);
      }
    };
    fetchCams();
  }, []);

  // 2. Start / Stop Webcam for CAM-01 if selected
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      webcamStreamRef.current = stream;
      if (videoRefs.current['CAM-01']) {
        videoRefs.current['CAM-01']!.srcObject = stream;
        videoRefs.current['CAM-01']!.play();
      }
      setWebcamActive(true);
    } catch (e) {
      console.warn("Could not start local webcam in monitoring wall:", e);
    }
  };

  const stopWebcam = () => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach(t => t.stop());
      webcamStreamRef.current = null;
    }
    if (videoRefs.current['CAM-01']) {
      videoRefs.current['CAM-01']!.srcObject = null;
    }
    setWebcamActive(false);
  };

  // 3. Real-Time High-Accuracy Frame Extraction and AI Inference Loop
  useEffect(() => {
    let isMounted = true;
    let timerId: any = null;

    const runInferenceCycle = async () => {
      if (!isMounted) return;

      const activeCamId = selectedCameraIdRef.current;
      const video = videoRefs.current[activeCamId];

      if (!video || video.readyState < 2 || video.paused || video.ended) {
        timerId = setTimeout(runInferenceCycle, 35);
        return;
      }

      const t0 = performance.now();
      try {
        const mode = accuracyModeRef.current;
        let captureW = 640;
        let captureH = 360;
        let quality = 0.78;
        if (mode === 'ultra') {
          captureW = 854;
          captureH = 480;
          quality = 0.82;
        } else if (mode === 'balanced') {
          captureW = 576;
          captureH = 324;
          quality = 0.70;
        } else if (mode === 'fast') {
          captureW = 480;
          captureH = 270;
          quality = 0.55;
        }

        let canvas = captureCanvasRef.current;
        if (!canvas) {
          canvas = document.createElement('canvas');
          captureCanvasRef.current = canvas;
        }
        if (canvas.width !== captureW || canvas.height !== captureH) {
          canvas.width = captureW;
          canvas.height = captureH;
        }

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, captureW, captureH);
          const base64Data = canvas.toDataURL('image/jpeg', quality);

          const response = await fetch(`${API_BASE}/detection/frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_base64: base64Data,
              camera_id: activeCamId,
              accuracy_mode: mode,
              confidence_threshold: confidenceThresholdRef.current / 100.0
            })
          });

          if (response.ok && isMounted) {
            const data = await response.json();

            // Coordinate normalization to standard 1280x720 canvas overlay
            const scaleX = 1280 / captureW;
            const scaleY = 720 / captureH;

            const scaledPeople: TrackedPerson[] = (data.people || []).map((p: any) => ({
              ...p,
              bbox: {
                x: p.bbox.x * scaleX,
                y: p.bbox.y * scaleY,
                width: p.bbox.width * scaleX,
                height: p.bbox.height * scaleY
              },
              landmarks: (p.landmarks || []).map((lm: any) => ({
                ...lm,
                pixel_x: lm.pixel_x ? lm.pixel_x * scaleX : (lm.x * 1280),
                pixel_y: lm.pixel_y ? lm.pixel_y * scaleY : (lm.y * 720)
              })),
              pose_angles: p.pose_angles,
              posture_details: p.posture_details,
              interacting_objects: (p.interacting_objects || []).map((obj: any) => ({
                ...obj,
                bbox: {
                  x: obj.bbox.x * scaleX,
                  y: obj.bbox.y * scaleY,
                  width: obj.bbox.width * scaleX,
                  height: obj.bbox.height * scaleY
                }
              }))
            }));

            const scaledObjects: DetectedObject[] = (data.objects || []).map((obj: any) => ({
              ...obj,
              bbox: {
                x: obj.bbox.x * scaleX,
                y: obj.bbox.y * scaleY,
                width: obj.bbox.width * scaleX,
                height: obj.bbox.height * scaleY
              }
            }));

            setCameraDetections(prev => ({
              ...prev,
              [activeCamId]: scaledPeople
            }));

            setCameraObjects(prev => ({
              ...prev,
              [activeCamId]: scaledObjects
            }));

            // Armed hazard or critical alert trigger
            const criticalThreat = scaledPeople.find(p => p.is_suspicious || p.activity === 'ARMED_HAZARD');
            if (criticalThreat && soundEnabled) {
              const now = Date.now();
              if (now - lastAlertTimeRef.current > 4000) {
                lastAlertTimeRef.current = now;
                audioService.playCriticalAlarm();
              }
            }

            const elapsed = performance.now() - t0;
            setFps(Math.round(1000 / Math.max(1, elapsed)));
            setLatency(Math.round(elapsed));
          }
        }
      } catch (err) {
        // Handle fetch errors gracefully
      } finally {
        if (isMounted) {
          timerId = setTimeout(runInferenceCycle, 25);
        }
      }
    };

    timerId = setTimeout(runInferenceCycle, 100);

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [selectedCameraId, soundEnabled]);

  const visibleCameras = selectedLayout === '1x1'
    ? cameras.filter(c => c.id === selectedCameraId)
    : selectedLayout === 'FOCUS'
    ? cameras.filter(c => c.id === selectedCameraId)
    : cameras.slice(0, selectedLayout === '2x2' ? 4 : 9);

  const focusedPeople = cameraDetections[selectedCameraId] || [];
  const focusedObjects = cameraObjects[selectedCameraId] || [];
  const hasThreat = focusedPeople.some(p => p.is_suspicious || p.activity === 'ARMED_HAZARD');

  return (
    <div className="p-6 space-y-6">
      {/* Top Banner: SOC Header & Status */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl shadow-2xl">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/40 text-cyan-400 shadow-lg shadow-cyan-500/10">
            <CamIcon className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold text-white font-mono tracking-wide">
                SOC LIVE SURVEILLANCE & AI MONITORING MATRIX
              </h1>
              <span className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-[10px] font-mono text-emerald-400 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                <span>ONLINE</span>
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              High-Precision Ensemble • <span className="text-purple-400 font-semibold">YOLOv11 Unified</span> + <span className="text-amber-400 font-semibold">Weapon Neural Net</span> + <span className="text-cyan-400 font-semibold">MediaPipe Pose 3D v1</span>
            </p>
          </div>
        </div>

        {/* Live Performance & Layout Controls */}
        <div className="flex items-center flex-wrap gap-3">
          <div className="flex items-center space-x-3 bg-slate-950/90 px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-mono">
            <div className="flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">FPS:</span>
              <span className="font-bold text-cyan-300">{fps}</span>
            </div>
            <div className="w-px h-3.5 bg-slate-800"></div>
            <div className="flex items-center space-x-1.5">
              <Radio className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-slate-400">LATENCY:</span>
              <span className="font-bold text-purple-300">{latency}ms</span>
            </div>
          </div>

          {/* Layout Switcher */}
          <div className="flex items-center space-x-1 bg-slate-950/90 p-1 rounded-lg border border-slate-800">
            {(['1x1', '2x2', '3x3', 'FOCUS'] as const).map((layout) => (
              <button
                key={layout}
                onClick={() => setSelectedLayout(layout)}
                className={`px-3 py-1 rounded text-xs font-mono font-bold transition-all ${
                  selectedLayout === layout
                    ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {layout}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* AI Detection Accuracy & Sensitivity Tuning Bar */}
      <div className="bg-[#0a0f19] border border-slate-800 p-3.5 rounded-xl flex flex-wrap items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center flex-wrap gap-3">
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-xs font-mono text-slate-300">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-bold">DETECTION ACCURACY:</span>
          </div>

          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 space-x-1">
            {[
              { id: 'ultra', label: 'ULTRA HD (512px)', sub: 'Max Weapon Precision' },
              { id: 'high', label: 'HIGH ACCURACY (416px)', sub: 'Optimal Surveillance' },
              { id: 'balanced', label: 'BALANCED (384px)', sub: '30+ FPS Smooth' },
              { id: 'fast', label: 'FAST (320px)', sub: 'Ultra-low Latency' },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setAccuracyMode(m.id as any)}
                className={`px-3 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                  accuracyMode === m.id
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_12px_rgba(0,240,255,0.4)]'
                    : 'text-slate-400 hover:text-white'
                }`}
                title={m.sub}
              >
                <span>{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Confidence Threshold Slider & Visualization Toggles */}
        <div className="flex items-center flex-wrap gap-3">
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <Sliders className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[11px] font-mono text-slate-400">CONFIDENCE THRESHOLD:</span>
            <input
              type="range"
              min="15"
              max="75"
              step="5"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseInt(e.target.value, 10))}
              className="w-24 accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              title="Minimum AI probability threshold to accept person/weapon detections"
            />
            <span className="text-xs font-mono font-bold text-amber-300 w-10 text-right">
              {confidenceThreshold}%
            </span>
          </div>

          {/* Skeletons Toggle */}
          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 border transition-all ${
              showSkeleton
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-[0_0_10px_rgba(0,240,255,0.25)]'
                : 'bg-slate-950 text-slate-400 border-slate-800'
            }`}
            title="Toggle 33-point MediaPipe Pose Wireframe Overlay"
          >
            <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
            <span>{showSkeleton ? 'SKELETON: ON' : 'SKELETON: OFF'}</span>
          </button>

          {/* Objects Toggle */}
          <button
            onClick={() => setShowObjects(!showObjects)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 border transition-all ${
              showObjects
                ? 'bg-purple-500/20 text-purple-300 border-purple-500/50 shadow-[0_0_10px_rgba(168,85,247,0.25)]'
                : 'bg-slate-950 text-slate-400 border-slate-800'
            }`}
            title="Toggle Context & Weapon Object Bounding Boxes"
          >
            <Layers className="w-3.5 h-3.5 text-purple-400" />
            <span>{showObjects ? 'OBJECTS: ON' : 'OBJECTS: OFF'}</span>
          </button>

          {/* Sound Toggle */}
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className={`p-2 rounded-lg border transition-all ${
              soundEnabled
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400'
                : 'bg-slate-950 border-slate-800 text-slate-500'
            }`}
            title="Toggle Threat Audible Audio Chime"
          >
            {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Critical Threat Alert Bar (Appears when active camera flags armed suspect) */}
      {hasThreat && (
        <div className="p-3.5 rounded-xl bg-red-950/50 border border-red-500/70 text-red-200 text-xs font-mono flex items-center justify-between shadow-[0_0_20px_rgba(239,68,68,0.3)] animate-pulse">
          <div className="flex items-center space-x-2.5">
            <ShieldAlert className="w-5 h-5 text-red-400" />
            <div>
              <span className="font-bold text-red-300">SECURITY ALERT:</span>
              <span className="ml-2">High-threat armed activity or weapon equipment detected in active feed!</span>
            </div>
          </div>
          <span className="px-2.5 py-1 rounded bg-red-600 text-white font-bold text-[11px]">
            ESCALATION CRITICAL
          </span>
        </div>
      )}

      {/* Surveillance Grid & SOC Viewports */}
      <div className={`grid gap-4 ${
        selectedLayout === '1x1' || selectedLayout === 'FOCUS'
          ? 'grid-cols-1'
          : selectedLayout === '2x2'
          ? 'grid-cols-1 md:grid-cols-2'
          : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
      }`}>
        {visibleCameras.map((cam) => {
          const isSelected = selectedCameraId === cam.id;
          const people = cameraDetections[cam.id] || [];
          const objects = cameraObjects[cam.id] || [];
          const camHasThreat = people.some(p => p.is_suspicious || p.activity === 'ARMED_HAZARD');
          const isWebcam = cam.stream_url === 'webcam://0';

          return (
            <div
              key={cam.id}
              onClick={() => setSelectedCameraId(cam.id)}
              className={`soc-panel rounded-2xl overflow-hidden relative group transition-all aspect-video bg-slate-950 border cursor-pointer ${
                isSelected
                  ? camHasThreat
                    ? 'border-red-500 shadow-[0_0_25px_rgba(239,68,68,0.4)]'
                    : 'border-cyan-500/80 shadow-[0_0_22px_rgba(0,240,255,0.2)]'
                  : camHasThreat
                  ? 'border-red-500/60'
                  : 'border-slate-800 hover:border-slate-700'
              }`}
            >
              {/* Surveillance Feed Viewport */}
              <div className="w-full h-full flex flex-col items-center justify-center relative bg-black">
                {isWebcam ? (
                  webcamActive ? (
                    <video
                      ref={el => { videoRefs.current[cam.id] = el; }}
                      className="w-full h-full object-cover"
                      playsInline
                      muted
                      autoPlay
                    />
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-slate-950 to-slate-900 p-6 text-center">
                      <div className="p-3 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 mb-3">
                        <Video className="w-8 h-8" />
                      </div>
                      <span className="text-sm font-bold font-mono text-white mb-1">LOCAL WEBCAM FEED</span>
                      <p className="text-xs font-mono text-slate-400 mb-4 max-w-xs">
                        Connect browser camera stream to test live multi-person tracking and weapon detection.
                      </p>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startWebcam();
                        }}
                        className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-bold text-xs flex items-center space-x-1.5 shadow-lg shadow-cyan-500/20"
                      >
                        <Play className="w-3.5 h-3.5" />
                        <span>ACTIVATE WEBCAM</span>
                      </button>
                    </div>
                  )
                ) : (
                  <video
                    ref={el => { videoRefs.current[cam.id] = el; }}
                    src={getMediaUrl(cam.stream_url)}
                    className="w-full h-full object-cover"
                    loop
                    muted
                    autoPlay
                    playsInline
                    crossOrigin="anonymous"
                  />
                )}

                {/* AI Canvas Overlay: Bounding Boxes, MediaPipe Wireframe Skeleton & Labels */}
                <CameraCanvasOverlay
                  people={people}
                  objects={showObjects ? objects : []}
                  videoWidth={1280}
                  videoHeight={720}
                  showBoundingBoxes={true}
                  showTrackingIds={true}
                  showActivityLabels={true}
                  showObjects={showObjects}
                  showSkeleton={showSkeleton}
                  highlightThreats={true}
                />
              </div>

              {/* Header Badges */}
              <div className="absolute top-3 left-3 z-30 flex items-center space-x-2">
                <span className="px-2 py-1 rounded bg-black/80 text-cyan-400 text-xs font-mono font-bold border border-slate-800">
                  {cam.id}
                </span>
                <span className="px-2 py-1 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 text-[11px] font-mono font-bold flex items-center space-x-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>LIVE</span>
                </span>
                {isSelected && (
                  <span className="px-2 py-1 rounded bg-cyan-500 text-slate-950 text-[10px] font-mono font-bold">
                    ACTIVE INFERENCE
                  </span>
                )}
              </div>

              {/* Threat Alert Badge if Flagged */}
              {camHasThreat && (
                <div className="absolute top-3 right-3 z-30 px-2.5 py-1 rounded bg-red-600 text-white text-xs font-mono font-bold flex items-center space-x-1 shadow-lg shadow-red-600/40 animate-bounce">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>ARMED HAZARD</span>
                </div>
              )}

              {/* Bottom Info Bar */}
              <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/95 via-black/70 to-transparent p-3 flex justify-between items-center text-xs font-mono text-slate-300">
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-white">{cam.name}</span>
                  <span className="text-slate-500">•</span>
                  <span className="text-slate-400">{cam.location}</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-cyan-400 font-bold">
                    {people.length} {people.length === 1 ? 'Person' : 'People'}
                  </span>
                  {objects.length > 0 && (
                    <span className="text-purple-400">
                      {objects.length} Objects
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Focus Mode & Forensic Telemetry Drawer for Selected Channel */}
      {selectedCameraId && (
        <div className="bg-[#0a0f19] border border-slate-800 p-4 rounded-xl shadow-xl space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-cyan-400" />
              <h2 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                CHANNEL TELEMETRY: {selectedCameraId}
              </h2>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Biomechanical Pose Angle & Spatial Object Analysis
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Person & Posture Breakdown */}
            <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 space-y-2">
              <span className="text-[11px] font-mono text-slate-400 font-bold block">
                DETECTED INDIVIDUALS ({focusedPeople.length})
              </span>
              {focusedPeople.length === 0 ? (
                <p className="text-xs font-mono text-slate-600">No active individuals in camera FOV</p>
              ) : (
                <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
                  {focusedPeople.map(p => (
                    <div
                      key={p.track_id}
                      className={`p-2 rounded border text-xs font-mono flex items-center justify-between ${
                        p.is_suspicious || p.activity === 'ARMED_HAZARD'
                          ? 'bg-red-950/30 border-red-500/40 text-red-300'
                          : 'bg-slate-900 border-slate-800 text-slate-200'
                      }`}
                    >
                      <div className="flex items-center space-x-1.5">
                        <span className="font-bold text-cyan-400">ID #{p.track_id}</span>
                        <span>•</span>
                        <span className="font-semibold">{p.activity}</span>
                      </div>
                      <span className="text-[11px] text-slate-400">{p.confidence.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Context Objects & Weapon Indicators */}
            <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 space-y-2">
              <span className="text-[11px] font-mono text-slate-400 font-bold block">
                CONTEXT & WEAPON DETECTIONS ({focusedObjects.length})
              </span>
              {focusedObjects.length === 0 ? (
                <p className="text-xs font-mono text-slate-600">No objects detected in active sector</p>
              ) : (
                <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto pr-1">
                  {focusedObjects.map((obj, i) => (
                    <span
                      key={i}
                      className={`px-2 py-1 rounded text-[11px] font-mono border ${
                        obj.category === 'threat' || obj.category === 'weapon'
                          ? 'bg-red-500/20 text-red-300 border-red-500/40 font-bold'
                          : 'bg-purple-500/10 text-purple-300 border-purple-500/30'
                      }`}
                    >
                      {obj.label} ({obj.confidence.toFixed(0)}%)
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Biomechanical Telemetry */}
            <div className="bg-slate-950/80 p-3 rounded-lg border border-slate-800/80 space-y-2">
              <span className="text-[11px] font-mono text-slate-400 font-bold block">
                BIOMECHANICAL POSE METRICS
              </span>
              {focusedPeople.length > 0 && focusedPeople[0].pose_angles ? (
                <div className="space-y-1 text-xs font-mono text-slate-300">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Spine Incline:</span>
                    <span className="font-bold text-cyan-400">
                      {focusedPeople[0].pose_angles.spine_incline ?? 'N/A'}°
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Left Knee Angle:</span>
                    <span className="font-bold text-purple-400">
                      {focusedPeople[0].pose_angles.left_knee ?? 'N/A'}°
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Right Knee Angle:</span>
                    <span className="font-bold text-purple-400">
                      {focusedPeople[0].pose_angles.right_knee ?? 'N/A'}°
                    </span>
                  </div>
                  {focusedPeople[0].posture_details && (
                    <p className="text-[10px] text-slate-400 italic pt-1 border-t border-slate-800">
                      {focusedPeople[0].posture_details}
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-xs font-mono text-slate-600">MediaPipe biomechanical angles standby</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
