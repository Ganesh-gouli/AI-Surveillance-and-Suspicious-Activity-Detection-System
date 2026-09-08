import React, { useState, useRef, useEffect } from 'react';
import {
  Webcam as WebcamIcon,
  Play,
  Square,
  Pause,
  Camera as SnapshotIcon,
  Activity,
  Cpu,
  CheckCircle,
  AlertTriangle,
  Layers,
  Box,
  Link as LinkIcon,
  Eye,
  Crosshair,
  Compass,
  Flame,
  ShieldAlert,
  Volume2,
  VolumeX,
  Moon,
  Sun,
  Sliders,
  Sparkles,
  Radio,
  Zap,
  Disc
} from 'lucide-react';
import { CameraCanvasOverlay } from '../components/video/CameraCanvasOverlay';
import { TrackedPerson, DetectedObject } from '../types';
import { audioService } from '../services/audioAlerts';
import { API_BASE } from '../services/api';

export const LiveWebcam: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [snapshotToast, setSnapshotToast] = useState(false);

  // Tactical Night Vision & Low-Light State
  const [nightVisionMode, setNightVisionMode] = useState<'off' | 'tactical_green' | 'low_light' | 'infrared' | 'thermal'>('off');
  const [sensorGain, setSensorGain] = useState<number>(1.5);
  const [irIlluminator, setIrIlluminator] = useState<boolean>(false);
  const [autoNightVision, setAutoNightVision] = useState<boolean>(true);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
  const [nightVisionTelemetry, setNightVisionTelemetry] = useState<{
    active: boolean;
    mode: string;
    estimated_lux: number;
    is_low_light: boolean;
    mean_luma: number;
    sensor_gain: number;
    sensor_gain_db: number;
    ir_illuminator: boolean;
    clahe_applied: boolean;
  }>({
    active: false,
    mode: 'off',
    estimated_lux: 85.0,
    is_low_light: false,
    mean_luma: 75.0,
    sensor_gain: 1.0,
    sensor_gain_db: 0.0,
    ir_illuminator: false,
    clahe_applied: false
  });

  // Refs for low-latency capture loop access without re-binding listeners
  const nightVisionModeRef = useRef(nightVisionMode);
  nightVisionModeRef.current = nightVisionMode;
  const sensorGainRef = useRef(sensorGain);
  sensorGainRef.current = sensorGain;
  const irIlluminatorRef = useRef(irIlluminator);
  irIlluminatorRef.current = irIlluminator;
  const autoNightVisionRef = useRef(autoNightVision);
  autoNightVisionRef.current = autoNightVision;

  // Web Audio Synthesizer: Military NVG Tube Capacitor Charge Whine
  const playNvgSound = (enable: boolean) => {
    if (!soundEnabled) return;
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gainNode = ctx.createGain();
      const filter = ctx.createBiquadFilter();

      filter.type = 'highpass';
      filter.frequency.setValueAtTime(1800, ctx.currentTime);

      if (enable) {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(3200, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(6800, ctx.currentTime + 0.26);
        gainNode.gain.setValueAtTime(0.09, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.32);
      } else {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(500, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(150, ctx.currentTime + 0.08);
        gainNode.gain.setValueAtTime(0.06, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
      }

      osc.connect(filter);
      filter.connect(gainNode);
      gainNode.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + (enable ? 0.34 : 0.11));
    } catch (e) {}
  };

  // PyTorch Fire Detection Live State (Section 11)
  const [fireMonitorActive, setFireMonitorActive] = useState<boolean>(true);
  const [fireData, setFireData] = useState<{
    is_fire: boolean;
    fire_confidence: number;
    no_fire_confidence: number;
    continuous_fire_alert: boolean;
    consecutive_fire_frames: number;
    boxes?: Array<{
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      width: number;
      height: number;
      label: string;
      confidence: number;
      is_fire: boolean;
    }>;
  }>({
    is_fire: false,
    fire_confidence: 0,
    no_fire_confidence: 100,
    continuous_fire_alert: false,
    consecutive_fire_frames: 0,
    boxes: []
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const isProcessingRef = useRef(false);

  // Real-time dynamic people & objects tracking on webcam
  const [people, setPeople] = useState<TrackedPerson[]>([]);
  const [objects, setObjects] = useState<DetectedObject[]>([]);

  // Start real webcam feed
  const startCamera = async () => {
    try {
      setMediaError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsStreaming(true);
      setIsPaused(false);
    } catch (err: any) {
      setMediaError("Could not access physical webcam. Please verify camera permissions in your browser.");
    }
  };

  // Stop webcam feed
  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
    setPeople([]);
    setObjects([]);
  };

  // Real-Time Adaptive Low-Latency Frame Capture & AI Inference Loop
  useEffect(() => {
    if (!isStreaming || isPaused) return;

    let isMounted = true;
    let timerId: any = null;

    const captureAndSend = async () => {
      if (!isMounted || isPaused) return;

      const video = videoRef.current;
      if (!video || video.readyState < 2 || isProcessingRef.current) {
        if (isMounted && !isPaused) {
          timerId = setTimeout(captureAndSend, 15);
        }
        return;
      }

      isProcessingRef.current = true;
      const t0 = performance.now();

      try {
        let canvas = captureCanvasRef.current;
        if (!canvas) {
          canvas = document.createElement('canvas');
          canvas.width = 480; // Optimized transfer resolution for instant low-latency inference
          canvas.height = 270;
          captureCanvasRef.current = canvas;
        }

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, 480, 270);
          // High-efficiency JPEG at 0.55 quality produces ~12-14KB payload for ultra-fast <2ms network transfer
          const base64Data = canvas.toDataURL('image/jpeg', 0.55);

          // Send real frame to FastAPI backend (YOLOv11 + MediaPipe Pose Biomechanics + Night Vision)
          const response = await fetch(`${API_BASE}/detection/frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_base64: base64Data,
              camera_id: 'LIVE-WEBCAM',
              night_vision_mode: nightVisionModeRef.current,
              night_vision_gain: sensorGainRef.current,
              ir_illuminator: irIlluminatorRef.current
            })
          });

          if (response.ok && isMounted) {
            const data = await response.json();
            
            // Scale bounding box & MediaPipe landmark coordinates back to full 1280x720 video display
            const scaleX = 1280 / 480;
            const scaleY = 720 / 270;

            const scaledPeople = (data.people || []).map((p: any) => ({
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

            const scaledObjects = (data.objects || []).map((obj: any) => ({
              ...obj,
              bbox: {
                x: obj.bbox.x * scaleX,
                y: obj.bbox.y * scaleY,
                width: obj.bbox.width * scaleX,
                height: obj.bbox.height * scaleY
              }
            }));

            setPeople(scaledPeople);
            setObjects(scaledObjects);

            // Night Vision Telemetry & Auto-Lux Handover
            if (data.night_vision) {
              setNightVisionTelemetry(data.night_vision);
              if (autoNightVisionRef.current && data.night_vision.is_low_light && nightVisionModeRef.current === 'off') {
                setNightVisionMode('tactical_green');
                playNvgSound(true);
              }
            }

            // Real-Time PyTorch Fire Detection Check (Section 11)
            if (fireMonitorActive) {
              try {
                const fireRes = await fetch(`${API_BASE}/fire/detect-frame`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    image_base64: base64Data,
                    camera_id: 'LIVE-WEBCAM'
                  })
                });
                if (fireRes.ok && isMounted) {
                  const fData = await fireRes.json();
                  setFireData(fData);
                  if (fData.continuous_fire_alert) {
                    audioService.playCriticalAlarm();
                  }
                }
              } catch (fErr) {}
            }

            const elapsed = performance.now() - t0;
            setFps(Math.round(1000 / Math.max(1, elapsed)));
            setLatency(Math.round(elapsed));
          }
        }
      } catch (err) {
        // Handle fetch errors gracefully
      } finally {
        isProcessingRef.current = false;
        if (isMounted && !isPaused) {
          // Immediately schedule next frame with minimal 10ms rest for 60fps UI fluidity
          timerId = setTimeout(captureAndSend, 10);
        }
      }
    };

    // Kick off adaptive loop
    timerId = setTimeout(captureAndSend, 10);

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [isStreaming, isPaused]);

  // Dynamic Video Filter & Color grading style for active Night Vision mode
  const getVideoFilterStyle = (): React.CSSProperties => {
    if (nightVisionMode === 'off') return {};

    if (nightVisionMode === 'tactical_green') {
      return {
        filter: `brightness(${1.15 * sensorGain}) contrast(1.4) saturate(0.2) hue-rotate(85deg) drop-shadow(0 0 14px rgba(34,197,94,0.4))`,
      };
    }
    if (nightVisionMode === 'infrared') {
      return {
        filter: `grayscale(1) brightness(${1.25 * sensorGain}) contrast(1.45)`,
      };
    }
    if (nightVisionMode === 'low_light') {
      return {
        filter: `brightness(${1.35 * sensorGain}) contrast(1.25) saturate(1.15)`,
      };
    }
    if (nightVisionMode === 'thermal') {
      return {
        filter: `invert(0.85) hue-rotate(180deg) saturate(2.4) contrast(1.4)`,
      };
    }
    return {};
  };

  // High-Resolution Snapshot Capture with active Night Vision enhancement baked in
  const handleSnapshot = () => {
    const video = videoRef.current;
    if (video && video.readyState >= 2) {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          if (nightVisionMode === 'tactical_green') {
            ctx.filter = `brightness(${1.2 * sensorGain}) contrast(1.4) saturate(0.2) hue-rotate(85deg)`;
          } else if (nightVisionMode === 'infrared') {
            ctx.filter = `grayscale(1) brightness(${1.25 * sensorGain}) contrast(1.5)`;
          } else if (nightVisionMode === 'low_light') {
            ctx.filter = `brightness(${1.4 * sensorGain}) contrast(1.25) saturate(1.15)`;
          } else if (nightVisionMode === 'thermal') {
            ctx.filter = `invert(0.85) hue-rotate(180deg) saturate(2.5) contrast(1.4)`;
          }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Tactical Watermark
          ctx.filter = 'none';
          ctx.fillStyle = nightVisionMode === 'tactical_green' ? '#22c55e' : '#00f0ff';
          ctx.font = 'bold 16px monospace';
          ctx.fillText(`SENTINELVISION AI • MODE: ${nightVisionMode.toUpperCase()} • ${new Date().toISOString()}`, 24, canvas.height - 24);

          const link = document.createElement('a');
          link.download = `sentinel_webcam_${nightVisionMode}_${Date.now()}.jpg`;
          link.href = canvas.toDataURL('image/jpeg', 0.92);
          link.click();
        }
      } catch (e) {}
    }

    setSnapshotToast(true);
    setTimeout(() => setSnapshotToast(false), 2000);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <WebcamIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              REAL-TIME WEBCAM AI VISION STUDIO
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              <span className="text-purple-400 font-bold">YOLOv11 Multi-Object Vision</span> • Person-Object Interaction • <span className="text-cyan-400 font-bold">best_human_activity_v2.keras</span>
            </p>
          </div>
        </div>

        {/* Live Controls */}
        <div className="flex items-center space-x-2">
          {!isStreaming ? (
            <button
              onClick={startCamera}
              className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg shadow-cyan-900/30 transition-all"
            >
              <Play className="w-4 h-4" />
              <span>START CAMERA</span>
            </button>
          ) : (
            <>
              <button
                onClick={() => setFireMonitorActive(!fireMonitorActive)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 border transition-all ${
                  fireMonitorActive
                    ? 'bg-red-500/20 text-red-300 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.3)]'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
                title="Toggle Real-Time PyTorch Fire Hazard Monitoring"
              >
                <Flame className={`w-3.5 h-3.5 ${fireMonitorActive ? 'text-red-400 animate-pulse' : 'text-slate-400'}`} />
                <span>{fireMonitorActive ? 'FIRE AI: ON' : 'FIRE AI: OFF'}</span>
              </button>

              <button
                onClick={() => setShowSkeleton(!showSkeleton)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 border transition-all ${
                  showSkeleton
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-[0_0_10px_rgba(0,240,255,0.25)]'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-400 border-slate-700'
                }`}
                title="Toggle Google MediaPipe 3D Skeleton Wireframe Overlay"
              >
                <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
                <span>{showSkeleton ? 'SKELETON: ON' : 'SKELETON: OFF'}</span>
              </button>

              <button
                onClick={() => setIsPaused(!isPaused)}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono flex items-center space-x-1 border border-slate-700"
              >
                {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                <span>{isPaused ? 'RESUME' : 'PAUSE'}</span>
              </button>

              <button
                onClick={handleSnapshot}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono flex items-center space-x-1 border border-slate-700"
              >
                <SnapshotIcon className="w-3.5 h-3.5 text-cyan-400" />
                <span>SNAPSHOT</span>
              </button>

              <button
                onClick={stopCamera}
                className="px-3 py-1.5 rounded-lg bg-red-600/80 hover:bg-red-600 text-white text-xs font-mono font-bold flex items-center space-x-1"
              >
                <Square className="w-3.5 h-3.5" />
                <span>STOP</span>
              </button>
            </>
          )}
        </div>
      </div>

      {mediaError && (
        <div className="p-3 rounded-lg bg-red-950/30 border border-red-500/40 text-xs font-mono text-red-300 flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span>{mediaError}</span>
        </div>
      )}

      {/* Tactical Night Vision System Bar */}
      <div className="bg-[#0a0f19] border border-slate-800 p-3.5 rounded-xl flex flex-wrap items-center justify-between gap-4 shadow-xl">
        {/* Left: Mode Switcher */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-xs font-mono text-slate-300">
            <Moon className={`w-3.5 h-3.5 ${nightVisionMode !== 'off' ? 'text-green-400 animate-pulse' : 'text-slate-400'}`} />
            <span className="font-bold">NIGHT VISION:</span>
          </div>

          {/* Buttons */}
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 space-x-1">
            <button
              onClick={() => {
                setNightVisionMode('off');
                playNvgSound(false);
              }}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                nightVisionMode === 'off'
                  ? 'bg-slate-800 text-white font-bold shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Sun className="w-3 h-3 text-amber-400" />
              <span>DAY</span>
            </button>

            <button
              onClick={() => {
                setNightVisionMode('tactical_green');
                playNvgSound(true);
              }}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                nightVisionMode === 'tactical_green'
                  ? 'bg-green-600 text-slate-950 font-bold shadow-[0_0_12px_rgba(34,197,94,0.5)]'
                  : 'text-green-400 hover:bg-green-950/40'
              }`}
              title="Military PVS-14 Gen-3 Phosphor Green Tactical Vision"
            >
              <Eye className="w-3 h-3" />
              <span>TACTICAL NVG</span>
            </button>

            <button
              onClick={() => {
                setNightVisionMode('low_light');
                playNvgSound(true);
              }}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                nightVisionMode === 'low_light'
                  ? 'bg-amber-500 text-slate-950 font-bold shadow-[0_0_12px_rgba(245,158,11,0.5)]'
                  : 'text-amber-400 hover:bg-amber-950/40'
              }`}
              title="Super-Starlight Adaptive LAB CLAHE Dynamic Range Recovery"
            >
              <Sparkles className="w-3 h-3" />
              <span>LOW-LIGHT HDR</span>
            </button>

            <button
              onClick={() => {
                setNightVisionMode('infrared');
                playNvgSound(true);
              }}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                nightVisionMode === 'infrared'
                  ? 'bg-slate-200 text-slate-950 font-bold shadow-[0_0_12px_rgba(255,255,255,0.4)]'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
              title="CCTV 850nm High-Contrast Monochrome Infrared"
            >
              <Radio className="w-3 h-3" />
              <span>CCTV IR</span>
            </button>

            <button
              onClick={() => {
                setNightVisionMode('thermal');
                playNvgSound(true);
              }}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-all flex items-center space-x-1 ${
                nightVisionMode === 'thermal'
                  ? 'bg-gradient-to-r from-red-600 to-purple-600 text-white font-bold shadow-[0_0_12px_rgba(239,68,68,0.5)]'
                  : 'text-red-400 hover:bg-red-950/40'
              }`}
              title="FLIR False-Color Thermal Heat Signatures"
            >
              <Flame className="w-3 h-3" />
              <span>THERMAL</span>
            </button>
          </div>
        </div>

        {/* Middle/Right: Sensor Gain, IR Beam, Auto-Lux & Audio */}
        <div className="flex items-center space-x-3">
          {/* Sensor Gain Slider */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[11px] font-mono text-slate-400">SENSOR GAIN:</span>
            <input
              type="range"
              min="1.0"
              max="3.5"
              step="0.1"
              value={sensorGain}
              onChange={(e) => setSensorGain(parseFloat(e.target.value))}
              className="w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              title="Digital Sensor ISO/Luminance Amplification"
            />
            <span className="text-xs font-mono font-bold text-cyan-300 w-12 text-right">
              +{nightVisionTelemetry.sensor_gain_db.toFixed(1)}dB
            </span>
          </div>

          {/* Virtual IR Illuminator Toggle */}
          <button
            onClick={() => {
              setIrIlluminator(!irIlluminator);
              playNvgSound(!irIlluminator);
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1.5 border transition-all ${
              irIlluminator
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/60 shadow-[0_0_14px_rgba(0,240,255,0.3)]'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
            }`}
            title="Virtual 850nm Center-Weighted Infrared Floodlight"
          >
            <Zap className={`w-3.5 h-3.5 ${irIlluminator ? 'text-cyan-400 animate-pulse' : 'text-slate-400'}`} />
            <span>{irIlluminator ? 'IR BEAM: ON' : 'IR BEAM: OFF'}</span>
          </button>

          {/* Auto-Lux Handover Toggle */}
          <button
            onClick={() => setAutoNightVision(!autoNightVision)}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-mono flex items-center space-x-1 border transition-all ${
              autoNightVision
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                : 'bg-slate-950 text-slate-500 border-slate-800'
            }`}
            title="Auto-activate Night Vision when room ambient lighting drops below 35 Lux"
          >
            <Disc className={`w-3 h-3 ${autoNightVision ? 'text-emerald-400 animate-spin' : 'text-slate-500'}`} />
            <span>AUTO-LUX</span>
          </button>

          {/* Sound Toggle */}
          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 hover:text-white"
            title="Toggle NVG tube audio sound effect"
          >
            {soundEnabled ? <Volume2 className="w-3.5 h-3.5 text-cyan-400" /> : <VolumeX className="w-3.5 h-3.5 text-slate-500" />}
          </button>
        </div>
      </div>

      {/* Main Webcam Feed & AI Telemetry HUD */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Video Viewport (3 cols) */}
        <div className="lg:col-span-3 space-y-4">
          <div className="relative aspect-video w-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl surveillance-feed">
            {/* Live Video Element with active Night Vision filter */}
            <video
              ref={videoRef}
              playsInline
              muted
              style={getVideoFilterStyle()}
              className={`w-full h-full object-cover transition-all duration-300 ${!isStreaming ? 'hidden' : ''}`}
            />

            {/* Virtual Center IR Illuminator Spotlight Beam */}
            {isStreaming && irIlluminator && (
              <div
                className="absolute inset-0 pointer-events-none z-10 transition-opacity duration-300"
                style={{
                  background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.08) 35%, transparent 70%)',
                  mixBlendMode: 'screen'
                }}
              />
            )}

            {/* Tactical Green Phosphor Vignette & Scanlines Texture */}
            {isStreaming && nightVisionMode === 'tactical_green' && (
              <div className="absolute inset-0 pointer-events-none z-10 overflow-hidden">
                {/* PVS-14 Optical Tube Vignette */}
                <div
                  className="absolute inset-0"
                  style={{
                    background: 'radial-gradient(ellipse at center, transparent 58%, rgba(0,25,5,0.4) 82%, rgba(0,10,2,0.92) 100%)'
                  }}
                />
                {/* Phosphor Scanlines */}
                <div
                  className="absolute inset-0 opacity-15"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(0deg, rgba(34,197,94,0.3) 0px, rgba(34,197,94,0.3) 1px, transparent 1px, transparent 3px)'
                  }}
                />
                {/* Tactical Center Reticle */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-28 h-28 border border-green-500/30 rounded-full flex items-center justify-center">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-ping" />
                    <div className="absolute w-8 h-0.5 bg-green-500/50 -top-4 left-1/2 -translate-x-1/2" />
                    <div className="absolute w-8 h-0.5 bg-green-500/50 -bottom-4 left-1/2 -translate-x-1/2" />
                    <div className="absolute h-8 w-0.5 bg-green-500/50 -left-4 top-1/2 -translate-y-1/2" />
                    <div className="absolute h-8 w-0.5 bg-green-500/50 -right-4 top-1/2 -translate-y-1/2" />
                  </div>
                </div>
              </div>
            )}

            {/* CCTV Monochrome Infrared Vignette */}
            {isStreaming && nightVisionMode === 'infrared' && (
              <div className="absolute inset-0 pointer-events-none z-10">
                <div
                  className="absolute inset-0"
                  style={{
                    background: 'radial-gradient(circle at center, transparent 70%, rgba(0,0,0,0.65) 100%)'
                  }}
                />
              </div>
            )}

            {/* Placeholder when stopped */}
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-gradient-to-br from-slate-950 to-[#070e1b]">
                <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-4 shadow-[0_0_20px_rgba(0,240,255,0.15)]">
                  <WebcamIcon className="w-8 h-8" />
                </div>
                <h3 className="text-base font-bold text-white font-mono">
                  YOLOv11 Vision Ready for Live Detection
                </h3>
                <p className="text-xs text-slate-400 font-mono mt-1 max-w-sm">
                  Click 'Start Camera' to begin real-time activity classification and object interaction detection via YOLOv11.
                </p>
                <button
                  onClick={startCamera}
                  className="mt-4 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-mono font-bold flex items-center space-x-2 shadow-lg shadow-cyan-500/25 transition-all"
                >
                  <Play className="w-4 h-4" />
                  <span>START LIVE AI DETECTION</span>
                </button>
              </div>
            )}

            {/* Real AI Bounding Box Canvas Overlay */}
            {isStreaming && (
              <CameraCanvasOverlay
                people={people}
                objects={objects}
                videoWidth={1280}
                videoHeight={720}
                showBoundingBoxes={true}
                showTrackingIds={true}
                showActivityLabels={true}
                showObjects={true}
                showSkeleton={showSkeleton}
              />
            )}

            {/* Real-Time Fire & Smoke Bounding Box Overlay (Section 6 & 11) */}
            {isStreaming && fireMonitorActive && fireData.boxes && fireData.boxes.map((b: any, idx: number) => {
              const scaleX = 1280 / 480;
              const scaleY = 720 / 270;
              const left = `${(b.x1 * scaleX / 1280) * 100}%`;
              const top = `${(b.y1 * scaleY / 720) * 100}%`;
              const width = `${(b.width * scaleX / 1280) * 100}%`;
              const height = `${(b.height * scaleY / 720) * 100}%`;
              const isFireBox = b.is_fire;

              return (
                <div
                  key={`fire-box-${idx}`}
                  style={{ left, top, width, height }}
                  className={`absolute pointer-events-none border-2 transition-all duration-75 z-20 ${
                    isFireBox
                      ? 'border-red-500 bg-red-500/20 shadow-[0_0_20px_rgba(239,68,68,0.8)] animate-pulse'
                      : 'border-yellow-400 bg-yellow-400/15'
                  }`}
                >
                  <div className={`absolute -top-6 left-0 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-white shadow-md flex items-center space-x-1 whitespace-nowrap ${
                    isFireBox ? 'bg-red-600 border border-yellow-400' : 'bg-slate-800 border border-yellow-500'
                  }`}>
                    <Flame className="w-3 h-3 text-yellow-300" />
                    <span>{b.label.toUpperCase()}: {b.confidence}%</span>
                  </div>
                </div>
              );
            })}

            {/* Top HUD Status */}
            {isStreaming && (
              <div className="absolute top-3 left-3 z-30 flex flex-wrap items-center gap-2">
                <span className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-red-600/90 text-white text-xs font-mono font-bold animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-white" />
                  <span>LIVE YOLOv11 + MEDIAPIPE</span>
                </span>
                {nightVisionMode !== 'off' && (
                  <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold flex items-center space-x-1.5 border shadow-lg ${
                    nightVisionMode === 'tactical_green'
                      ? 'bg-green-950/90 border-green-500 text-green-300 shadow-green-950/50'
                      : nightVisionMode === 'infrared'
                      ? 'bg-slate-900/95 border-slate-400 text-slate-200'
                      : nightVisionMode === 'thermal'
                      ? 'bg-purple-950/90 border-purple-500 text-purple-300'
                      : 'bg-amber-950/90 border-amber-500 text-amber-300'
                  }`}>
                    <Moon className="w-3 h-3 text-green-400 animate-pulse" />
                    <span>
                      {nightVisionMode === 'tactical_green'
                        ? 'NVG GEN-3'
                        : nightVisionMode === 'infrared'
                        ? 'IR 850nm'
                        : nightVisionMode === 'thermal'
                        ? 'FLIR THERMAL'
                        : 'LOW-LIGHT HDR'}
                    </span>
                    <span className="text-[10px] opacity-75">
                      +{nightVisionTelemetry.sensor_gain_db.toFixed(1)}dB • {nightVisionTelemetry.estimated_lux} lx
                    </span>
                  </span>
                )}
                {fireMonitorActive && (
                  <span className={`px-2 py-1 rounded border text-xs font-mono font-bold flex items-center space-x-1 ${
                    fireData.is_fire ? 'bg-red-950/80 border-red-500 text-red-300 animate-pulse' : 'bg-black/70 border-slate-700 text-emerald-400'
                  }`}>
                    <Flame className="w-3 h-3" />
                    <span>{fireData.is_fire ? `FIRE ${fireData.fire_confidence}%` : 'NO FIRE'}</span>
                  </span>
                )}
                <span className="px-2 py-1 rounded bg-black/70 border border-slate-700 text-xs font-mono text-cyan-400">
                  {fps} FPS • Latency: {latency}ms
                </span>
              </div>
            )}

            {/* Continuous Fire Alert Banner (Section 11 requirement) */}
            {fireMonitorActive && fireData.continuous_fire_alert && (
              <div className="absolute inset-x-0 top-16 z-50 flex flex-col items-center justify-center p-4 animate-bounce">
                <div className="bg-red-600/95 text-white font-mono font-black text-xl px-8 py-3 rounded-2xl border-2 border-yellow-400 shadow-[0_0_50px_rgba(239,68,68,1)] flex items-center space-x-3">
                  <Flame className="w-8 h-8 text-yellow-300 animate-pulse" />
                  <span>🔥🔥 FIRE ALERT 🔥🔥</span>
                  <Flame className="w-8 h-8 text-yellow-300 animate-pulse" />
                </div>
                <div className="mt-1.5 px-4 py-1 bg-black/90 rounded-full border border-red-500/50 text-red-300 text-xs font-mono shadow-lg">
                  Continuous Flame Hazard ({fireData.consecutive_fire_frames} frames) • Conf: {fireData.fire_confidence}% • Siren Active
                </div>
              </div>
            )}

            {/* Snapshot Toast */}
            {snapshotToast && (
              <div className="absolute top-12 right-4 z-40 bg-emerald-500 text-white text-xs font-mono font-bold px-3 py-1.5 rounded-lg shadow-lg flex items-center space-x-1.5 animate-bounce">
                <CheckCircle className="w-4 h-4" />
                <span>SNAPSHOT CAPTURED</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Real-Time Edge AI Telemetry HUD */}
        <div className="space-y-4">
          <div className="soc-panel p-4 rounded-xl space-y-4">
            <div className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>Real-Time AI Vision Telemetry</span>
            </div>

            {/* Ambient Lux & Night Vision Telemetry HUD */}
            <div className={`p-3.5 rounded-xl border space-y-2.5 transition-all ${
              nightVisionMode !== 'off'
                ? nightVisionMode === 'tactical_green'
                  ? 'bg-green-950/40 border-green-500/50 shadow-[0_0_15px_rgba(34,197,94,0.2)]'
                  : nightVisionMode === 'infrared'
                  ? 'bg-slate-900/90 border-slate-400/50'
                  : nightVisionMode === 'thermal'
                  ? 'bg-purple-950/40 border-purple-500/50'
                  : 'bg-amber-950/40 border-amber-500/50'
                : 'bg-slate-900/80 border-slate-800'
            }`}>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold flex items-center space-x-1.5 text-white">
                  <Moon className={`w-4 h-4 ${nightVisionMode !== 'off' ? 'text-green-400 animate-pulse' : 'text-slate-400'}`} />
                  <span>LUX & NIGHT VISION</span>
                </span>
                <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                  nightVisionMode !== 'off'
                    ? 'bg-green-500/20 text-green-300 border border-green-500/40'
                    : 'bg-slate-800 text-slate-400'
                }`}>
                  {nightVisionMode !== 'off' ? nightVisionMode.replace('_', ' ') : 'STANDBY'}
                </span>
              </div>

              {/* Real-time Ambient Lux Meter */}
              <div className="space-y-1 pt-1">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-400">Ambient Illuminance:</span>
                  <span className={`font-bold text-sm ${
                    nightVisionTelemetry.estimated_lux < 25
                      ? 'text-red-400'
                      : nightVisionTelemetry.estimated_lux < 45
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}>
                    {nightVisionTelemetry.estimated_lux.toFixed(1)} Lux
                  </span>
                </div>

                {/* Lux Visual Meter Bar */}
                <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800/80">
                  <div
                    className={`h-full transition-all duration-300 ${
                      nightVisionTelemetry.estimated_lux < 25
                        ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]'
                        : nightVisionTelemetry.estimated_lux < 45
                        ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]'
                        : 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]'
                    }`}
                    style={{ width: `${Math.min(100, (nightVisionTelemetry.estimated_lux / 250) * 100)}%` }}
                  />
                </div>

                <div className="flex justify-between items-center text-[10px] font-mono text-slate-500">
                  <span>0 lx (Dark)</span>
                  <span className={`${nightVisionTelemetry.is_low_light ? 'text-amber-400 font-bold' : 'text-slate-400'}`}>
                    {nightVisionTelemetry.estimated_lux < 15
                      ? '● PITCH DARK'
                      : nightVisionTelemetry.estimated_lux < 40
                      ? '● LOW LIGHT'
                      : '● NORMAL ILLUMINATED'}
                  </span>
                  <span>250+ lx</span>
                </div>
              </div>

              {/* Low Light Alert with 1-Click Activation if Night Vision is OFF */}
              {nightVisionTelemetry.is_low_light && nightVisionMode === 'off' && (
                <div className="p-2 rounded bg-amber-950/60 border border-amber-500/40 text-[11px] font-mono text-amber-200 flex flex-col gap-1.5 animate-pulse">
                  <div className="flex items-center space-x-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    <span>Low-Light Detected (&lt;40 Lux). Enhance sensor now:</span>
                  </div>
                  <button
                    onClick={() => {
                      setNightVisionMode('tactical_green');
                      playNvgSound(true);
                    }}
                    className="w-full py-1 rounded bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-slate-950 font-bold text-[10px] flex items-center justify-center space-x-1 shadow-md shadow-green-950/40"
                  >
                    <Moon className="w-3 h-3 text-slate-950" />
                    <span>ACTIVATE TACTICAL NVG</span>
                  </button>
                </div>
              )}

              {/* Diagnostics Grid */}
              <div className="grid grid-cols-2 gap-1.5 pt-1 text-[10px] font-mono text-slate-400">
                <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[9px]">SENSOR GAIN</span>
                  <span className="text-cyan-400 font-bold">+{nightVisionTelemetry.sensor_gain_db.toFixed(1)} dB</span>
                </div>
                <div className="bg-slate-950/80 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block text-[9px]">850nm IR BEAM</span>
                  <span className={`font-bold ${irIlluminator ? 'text-cyan-400' : 'text-slate-500'}`}>
                    {irIlluminator ? 'ACTIVE' : 'OFF'}
                  </span>
                </div>
              </div>
            </div>

            {/* Real-Time PyTorch Fire Telemetry HUD (Section 11) */}
            {fireMonitorActive && (
              <div className={`p-3 rounded-xl border space-y-2 transition-all ${
                fireData.continuous_fire_alert
                  ? 'bg-red-950/70 border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.5)]'
                  : fireData.is_fire
                  ? 'bg-orange-950/40 border-orange-500/60'
                  : 'bg-slate-900/80 border-slate-800'
              }`}>
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="font-bold flex items-center space-x-1.5 text-white">
                    <Flame className={`w-4 h-4 ${fireData.is_fire ? 'text-red-400 animate-pulse' : 'text-orange-400'}`} />
                    <span>PYTORCH FIRE AI</span>
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                    fireData.continuous_fire_alert
                      ? 'bg-red-500 text-white animate-pulse'
                      : fireData.is_fire
                      ? 'bg-orange-500/20 text-orange-400'
                      : 'bg-emerald-500/20 text-emerald-400'
                  }`}>
                    {fireData.continuous_fire_alert ? 'HAZARD ALARM' : (fireData.is_fire ? 'DETECTED' : 'NORMAL')}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs font-mono text-slate-300 pt-0.5">
                  <span>Fire Probability:</span>
                  <span className={`font-bold text-sm ${fireData.is_fire ? 'text-red-400' : 'text-emerald-400'}`}>
                    {fireData.fire_confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-200 ${
                      fireData.fire_confidence > 70 ? 'bg-red-500' : fireData.fire_confidence > 40 ? 'bg-orange-500' : 'bg-emerald-500'
                    }`}
                    style={{ width: `${fireData.fire_confidence}%` }}
                  />
                </div>
                <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 pt-0.5">
                  <span>Consecutive: {fireData.consecutive_fire_frames} frames</span>
                  <span>Rule: ≥3 frames alarm</span>
                </div>
              </div>
            )}

            {/* Primary Activity Badge */}
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg space-y-1">
              <span className="text-[10px] font-mono text-slate-500 block">PREDICTED ACTIVITY</span>
              <div className="text-lg font-mono font-bold text-cyan-400">
                {people.length > 0 ? people[0].activity : "NO PERSON DETECTED"}
              </div>
              <div className="flex justify-between items-center text-xs font-mono pt-1 text-slate-400">
                <span>Model Confidence:</span>
                <span className="text-emerald-400 font-bold">
                  {people.length > 0 ? `${people[0].confidence}%` : "0.0%"}
                </span>
              </div>

              {/* Interacting Objects Chip */}
              {people.length > 0 && people[0].interacting_objects && people[0].interacting_objects.length > 0 && (
                <div className="pt-2 border-t border-slate-800 flex items-center space-x-1.5 text-xs font-mono">
                  <LinkIcon className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                  <span className="text-[10px] text-slate-400">Interacting:</span>
                  <div className="flex flex-wrap gap-1">
                    {people[0].interacting_objects.map((obj, i) => (
                      <span key={i} className="px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300 border border-purple-500/30 text-[10px] font-bold">
                        {obj.label}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Real-Time Google MediaPipe Pose Biomechanical Telemetry */}
            <div className="bg-slate-900/90 border border-cyan-500/30 p-3 rounded-lg space-y-2.5 shadow-[0_0_15px_rgba(0,240,255,0.07)]">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono font-bold text-cyan-400 flex items-center space-x-1.5">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  <span>MEDIAPIPE 3D SKELETAL POSE</span>
                </span>
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30">
                  {people.length > 0 && people[0].landmarks && people[0].landmarks.length > 0
                    ? `${people[0].landmarks.length} NODES`
                    : "STANDBY"}
                </span>
              </div>

              {/* Posture Diagnosis Banner */}
              <div className="p-2 rounded bg-slate-950/80 border border-slate-800">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-400 text-[11px]">Posture Status:</span>
                  <span className={`font-bold px-2 py-0.5 rounded text-[11px] ${
                    people.length > 0 && people[0].activity.toLowerCase() === 'standing'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : people.length > 0 && people[0].activity.toLowerCase() === 'sitting'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : people.length > 0 && people[0].activity.toLowerCase() === 'sleeping'
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                      : 'bg-slate-800 text-slate-300'
                  }`}>
                    {people.length > 0 ? people[0].activity.toUpperCase() : "NO POSE"}
                  </span>
                </div>
                {people.length > 0 && people[0].posture_details && (
                  <p className="text-[10px] font-mono text-slate-400 mt-1 truncate" title={people[0].posture_details}>
                    {people[0].posture_details}
                  </p>
                )}
              </div>

              {/* Biomechanical Angles Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                {/* Knee Angle */}
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                  <div className="flex justify-between items-center text-[10px] text-slate-400">
                    <span>KNEE ANGLE</span>
                    <span className="text-slate-500">(150°+ Stand)</span>
                  </div>
                  <div className="text-sm font-bold text-cyan-400 mt-0.5">
                    {people.length > 0 && people[0].pose_angles?.left_knee !== undefined && people[0].pose_angles?.left_knee !== null
                      ? `${people[0].pose_angles.left_knee.toFixed(1)}°`
                      : people.length > 0 && people[0].pose_angles?.right_knee !== undefined && people[0].pose_angles?.right_knee !== null
                      ? `${people[0].pose_angles.right_knee.toFixed(1)}°`
                      : "--"}
                  </div>
                </div>

                {/* Hip Angle */}
                <div className="bg-slate-950/60 p-2 rounded border border-slate-800/80">
                  <div className="flex justify-between items-center text-[10px] text-slate-400">
                    <span>HIP ANGLE</span>
                    <span className="text-slate-500">(Torso-Thigh)</span>
                  </div>
                  <div className="text-sm font-bold text-blue-400 mt-0.5">
                    {people.length > 0 && people[0].pose_angles?.left_hip !== undefined && people[0].pose_angles?.left_hip !== null
                      ? `${people[0].pose_angles.left_hip.toFixed(1)}°`
                      : people.length > 0 && people[0].pose_angles?.right_hip !== undefined && people[0].pose_angles?.right_hip !== null
                      ? `${people[0].pose_angles.right_hip.toFixed(1)}°`
                      : "--"}
                  </div>
                </div>
              </div>

              {/* Spine Incline */}
              {people.length > 0 && people[0].pose_angles?.spine_incline !== undefined && people[0].pose_angles?.spine_incline !== null && (
                <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 px-1">
                  <span>Spine Incline from Vertical:</span>
                  <span className="text-purple-400 font-bold">
                    {people[0].pose_angles.spine_incline.toFixed(1)}°
                  </span>
                </div>
              )}
            </div>

            {/* Telemetry Metrics */}
            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-500 block text-[10px]">DETECTED PEOPLE</span>
                <span className="text-blue-400 font-bold text-sm">{people.length}</span>
              </div>
              <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-500 block text-[10px]">SCENE OBJECTS</span>
                <span className="text-purple-400 font-bold text-sm">{objects.length}</span>
              </div>
            </div>

            {/* Target Activity Classes Bar */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <span className="text-[10px] font-mono text-slate-500 block">SUPPORTED ACTIVITIES</span>
              <div className="flex flex-wrap gap-1.5">
                {["sitting", "standing", "using_laptop", "using_phone", "drinking", "eating", "cycling", "running", "fighting", "sleeping", "carrying_bag"].map((cls) => {
                  const isActive = people.length > 0 && people[0].activity.toLowerCase() === cls;
                  return (
                    <span
                      key={cls}
                      className={`text-[10px] px-2 py-0.5 rounded font-mono transition-colors ${
                        isActive
                          ? 'bg-cyan-500 text-slate-950 font-bold shadow-[0_0_8px_rgba(0,240,255,0.6)]'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {cls}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
