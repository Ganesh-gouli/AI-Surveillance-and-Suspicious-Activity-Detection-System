import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  Maximize,
  Camera as SnapshotIcon,
  Volume2,
  VolumeX,
  RotateCcw,
  CheckCircle,
  Eye,
  Sliders,
  AlertTriangle,
  CircleDot
} from 'lucide-react';
import { CameraCanvasOverlay } from './CameraCanvasOverlay';
import { TrackedPerson, RestrictedZone, ActivityTimelineItem } from '../../types';

interface VideoPlayerProps {
  streamTitle?: string;
  cameraId?: string;
  location?: string;
  fps?: number;
  aiConfidence?: number;
  people?: TrackedPerson[];
  zones?: RestrictedZone[];
  timelineEvents?: ActivityTimelineItem[];
  isLive?: boolean;
  videoSrc?: string;
  onSnapshot?: (dataUrl: string) => void;
  onTimelineSeek?: (seconds: number) => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  streamTitle = "Camera Feed",
  cameraId = "CAM-001",
  location = "Security Sector A",
  fps = 30,
  aiConfidence = 96.4,
  people = [],
  zones = [],
  timelineEvents = [],
  isLive = true,
  videoSrc,
  onSnapshot,
  onTimelineSeek
}) => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(154); // 02:34
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isRecording, setIsRecording] = useState(false);
  const [snapshotFeedback, setSnapshotFeedback] = useState(false);

  // Overlay HUD Layer Toggles
  const [showBoxes, setShowBoxes] = useState(true);
  const [showIds, setShowIds] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showZones, setShowZones] = useState(true);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err => console.log(err));
    } else {
      document.exitFullscreen();
    }
  };

  // Instant snapshot capture
  const handleSnapshot = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 1280, 720);
      ctx.fillStyle = '#00f0ff';
      ctx.font = 'bold 24px JetBrains Mono';
      ctx.fillText(`SENTINELVISION AI SNAPSHOT • ${cameraId} • ${new Date().toISOString()}`, 40, 60);
      
      const dataUrl = canvas.toDataURL('image/jpeg');
      if (onSnapshot) onSnapshot(dataUrl);

      setSnapshotFeedback(true);
      setTimeout(() => setSnapshotFeedback(false), 2000);
    }
  };

  const hasCriticalThreat = people.some(p => p.is_suspicious || p.threat_level === 'CRITICAL');

  return (
    <div
      ref={containerRef}
      className={`relative w-full bg-[#05080e] rounded-xl overflow-hidden border transition-all ${
        hasCriticalThreat ? 'border-red-500 shadow-[0_0_25px_rgba(239,68,68,0.3)] animate-threat-flash' : 'border-slate-800/80 shadow-2xl'
      }`}
    >
      {/* Top Video HUD Bar */}
      <div className="absolute top-0 inset-x-0 h-11 bg-gradient-to-b from-black/90 via-black/50 to-transparent z-30 px-4 flex items-center justify-between pointer-events-auto">
        <div className="flex items-center space-x-2.5">
          {isLive ? (
            <span className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-red-600/90 text-white text-[10px] font-mono font-bold tracking-wider animate-pulse">
              <CircleDot className="w-3 h-3" />
              <span>LIVE</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-cyan-600/90 text-white text-[10px] font-mono font-bold tracking-wider">
              <span>PLAYBACK</span>
            </span>
          )}
          <span className="text-xs font-mono font-bold text-cyan-300">{cameraId}</span>
          <span className="text-xs text-slate-300 font-medium hidden sm:inline">• {streamTitle}</span>
          <span className="text-[11px] text-slate-400 hidden md:inline">({location})</span>
        </div>

        {/* Real-time Video Stream Telemetry HUD */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="text-slate-400 hidden sm:block">
            FPS: <span className="text-cyan-400 font-bold">{fps}</span>
          </div>
          <div className="text-slate-400 hidden sm:block">
            PEOPLE: <span className="text-blue-400 font-bold">{people.length}</span>
          </div>
          <div className="text-slate-400 hidden sm:block">
            AI CONF: <span className="text-emerald-400 font-bold">{aiConfidence}%</span>
          </div>
          {isRecording && (
            <span className="flex items-center space-x-1 text-red-400 text-xs font-bold animate-pulse">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span>REC</span>
            </span>
          )}
        </div>
      </div>

      {/* Main Viewport Container */}
      <div className="relative aspect-video w-full bg-slate-950 flex items-center justify-center surveillance-feed select-none overflow-hidden">
        {/* Background Simulated Camera Feed Graphics */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-[#0a1120] to-[#060910] opacity-90" />

        {/* Ambient Grid overlay representing security camera sensor */}
        <div
          className="absolute inset-0 opacity-15"
          style={{
            backgroundImage: `radial-gradient(circle, #38bdf8 1px, transparent 1px)`,
            backgroundSize: '32px 32px'
          }}
        />

        {/* Camera Center Crosshair */}
        <div className="absolute inset-0 m-auto w-12 h-12 pointer-events-none opacity-30">
          <div className="w-full h-px bg-cyan-400 absolute top-1/2 -translate-y-1/2" />
          <div className="h-full w-px bg-cyan-400 absolute left-1/2 -translate-x-1/2" />
        </div>

        {/* AI Canvas Detection Overlay */}
        <CameraCanvasOverlay
          people={people}
          zones={zones}
          showBoundingBoxes={showBoxes}
          showTrackingIds={showIds}
          showActivityLabels={showLabels}
          showZones={showZones}
          highlightThreats={true}
        />

        {/* Snapshot Success Toast Overlay */}
        {snapshotFeedback && (
          <div className="absolute top-14 right-4 z-40 bg-emerald-500/90 text-white text-xs font-mono font-bold px-3 py-1.5 rounded-lg shadow-lg flex items-center space-x-2 animate-bounce">
            <CheckCircle className="w-4 h-4" />
            <span>SNAPSHOT CAPTURED & SAVED</span>
          </div>
        )}
      </div>

      {/* Interactive Activity Timeline Scrub Bar (If playback/forensic video) */}
      {timelineEvents.length > 0 && (
        <div className="bg-[#090e17] px-4 py-2 border-t border-slate-800">
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1.5">
            <span className="font-bold text-cyan-400 flex items-center space-x-1">
              <Sliders className="w-3 h-3" />
              <span>AI ACTIVITY TIMELINE & INCIDENT MARKERS</span>
            </span>
            <span>Click marker to jump video to suspicious event</span>
          </div>

          <div className="relative h-6 bg-slate-900 rounded-md flex items-center px-1 overflow-x-auto">
            {/* Timeline Progress Base */}
            <div className="absolute left-0 top-0 bottom-0 bg-cyan-500/10 w-full rounded" />
            
            {/* Clickable Event Badges */}
            <div className="flex items-center space-x-2 z-10 w-full">
              {timelineEvents.map((ev, i) => {
                const isSusp = ev.is_suspicious;
                return (
                  <button
                    key={i}
                    onClick={() => onTimelineSeek && onTimelineSeek(ev.time_offset_sec)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold whitespace-nowrap transition-all border ${
                      isSusp
                        ? 'bg-red-600 text-white border-red-400 shadow-[0_0_8px_rgba(239,68,68,0.5)] animate-pulse'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                    }`}
                  >
                    {Math.floor(ev.time_offset_sec / 60).toString().padStart(2, '0')}:
                    {Math.floor(ev.time_offset_sec % 60).toString().padStart(2, '0')} ─ {ev.activity}
                    {isSusp && ' ⚠'}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Bottom Video Controls & HUD Overlay Toggles */}
      <div className="h-12 bg-[#080d16] border-t border-slate-800/90 px-4 flex items-center justify-between z-30">
        {/* Left: Play/Pause, Snapshot, Record */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors"
            title={isPlaying ? "Pause Stream" : "Play Stream"}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>

          <button
            onClick={handleSnapshot}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-cyan-400 transition-colors"
            title="Instant Snapshot"
          >
            <SnapshotIcon className="w-4 h-4" />
          </button>

          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`px-2 py-1 rounded text-xs font-mono flex items-center space-x-1 border transition-colors ${
              isRecording
                ? 'bg-red-500/20 border-red-500/50 text-red-400'
                : 'bg-slate-800/80 border-slate-700 text-slate-300 hover:text-white'
            }`}
            title="Local Clip Recording"
          >
            <span className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-ping' : 'bg-slate-500'}`} />
            <span>{isRecording ? 'RECORDING' : 'RECORD'}</span>
          </button>
        </div>

        {/* Center: HUD Layer Switches */}
        <div className="hidden sm:flex items-center space-x-2 bg-slate-900/80 px-2 py-1 rounded-md border border-slate-800 text-[11px] font-mono">
          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className={`px-2 py-0.5 rounded transition-colors ${
              showBoxes ? 'bg-cyan-500/20 text-cyan-400 font-bold' : 'text-slate-500 hover:text-slate-400'
            }`}
          >
            BOXES
          </button>
          <button
            onClick={() => setShowIds(!showIds)}
            className={`px-2 py-0.5 rounded transition-colors ${
              showIds ? 'bg-cyan-500/20 text-cyan-400 font-bold' : 'text-slate-500 hover:text-slate-400'
            }`}
          >
            TRACK IDs
          </button>
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`px-2 py-0.5 rounded transition-colors ${
              showLabels ? 'bg-cyan-500/20 text-cyan-400 font-bold' : 'text-slate-500 hover:text-slate-400'
            }`}
          >
            ACTIVITIES
          </button>
          <button
            onClick={() => setShowZones(!showZones)}
            className={`px-2 py-0.5 rounded transition-colors ${
              showZones ? 'bg-cyan-500/20 text-cyan-400 font-bold' : 'text-slate-500 hover:text-slate-400'
            }`}
          >
            ZONES
          </button>
        </div>

        {/* Right: Fullscreen & Mute */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-cyan-400 transition-colors"
            title={isMuted ? "Unmute Audio" : "Mute Audio"}
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>

          <button
            onClick={toggleFullscreen}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Fullscreen Mode"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
