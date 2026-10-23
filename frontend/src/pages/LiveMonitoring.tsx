import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Maximize2,
  Minimize2,
  Camera as CamIcon,
  AlertTriangle,
  Play,
  Square,
  Shield,
  Layers,
  Sparkles,
  RotateCcw,
  Plus,
  Tv
} from 'lucide-react';
import { api } from '../services/api';
import { Camera, TrackedPerson } from '../types';
import { CameraCanvasOverlay } from '../components/video/CameraCanvasOverlay';

export const LiveMonitoring: React.FC = () => {
  const navigate = useNavigate();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedLayout, setSelectedLayout] = useState<'1x1' | '2x2' | '3x3' | 'FOCUS'>('2x2');
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [cameraDetections, setCameraDetections] = useState<Record<string, TrackedPerson[]>>({});
  const [isFullscreen, setIsFullscreen] = useState(false);
  const wallContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const fetchCams = async () => {
      try {
        const cams = await api.getCameras();
        setCameras(cams);
        if (cams.length > 0 && !selectedCameraId) {
          setSelectedCameraId(cams[0].id);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchCams();
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      wallContainerRef.current?.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  const visibleCameras =
    selectedLayout === '1x1'
      ? cameras.filter((c) => c.id === selectedCameraId)
      : selectedLayout === 'FOCUS'
      ? cameras.filter((c) => c.id === selectedCameraId)
      : cameras.slice(0, selectedLayout === '2x2' ? 4 : 9);

  return (
    <div ref={wallContainerRef} className="p-4 sm:p-6 space-y-6 cyber-grid-bg min-h-full">
      {/* Top Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 sm:p-5 rounded-2xl shadow-xl">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <CamIcon className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold text-white font-mono tracking-tight">
              LIVE SURVEILLANCE & MULTI-CAMERA MONITORING WALL
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Synchronous Ingestion • <span className="text-purple-400 font-bold">YOLOv11</span> +{' '}
              <span className="text-cyan-400 font-bold">Keras Activity Engine</span>
            </p>
          </div>
        </div>

        {/* Layout & Fullscreen Switcher */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            {(['1x1', '2x2', '3x3', 'FOCUS'] as const).map((layout) => (
              <button
                key={layout}
                onClick={() => setSelectedLayout(layout)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                  selectedLayout === layout
                    ? 'btn-tactical-primary text-slate-950 shadow-md shadow-cyan-500/20'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {layout}
              </button>
            ))}
          </div>

          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-xl btn-tactical-secondary text-slate-300 hover:text-white"
            title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen Surveillance Mode'}
            aria-label="Toggle Fullscreen"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4 text-cyan-400" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Camera Selection Pills */}
      {cameras.length > 0 && (
        <div className="flex items-center space-x-2 overflow-x-auto custom-scrollbar pb-1">
          <span className="text-xs font-mono text-slate-400 shrink-0 font-bold">FOCUS STREAM:</span>
          {cameras.map((c) => (
            <button
              key={c.id}
              onClick={() => setSelectedCameraId(c.id)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 border ${
                selectedCameraId === c.id
                  ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                  : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {c.id} • {c.name}
            </button>
          ))}
        </div>
      )}

      {/* Grid Container */}
      {cameras.length === 0 ? (
        <div className="soc-panel p-12 rounded-2xl border border-slate-800 text-center space-y-3 shadow-xl">
          <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-slate-500 inline-block">
            <Tv className="w-10 h-10 text-cyan-400" />
          </div>
          <h3 className="text-sm font-mono font-bold text-white">NO LIVE CAMERAS DETECTED</h3>
          <p className="text-xs font-mono text-slate-400 max-w-sm mx-auto">
            The surveillance wall requires active RTSP or IP camera feeds. Provision camera streams in Fleet Manager to view live feeds here.
          </p>
          <button
            onClick={() => navigate('/cameras')}
            className="px-4 py-2 rounded-xl btn-tactical-primary text-xs font-mono font-bold inline-flex items-center space-x-1.5 mt-2"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>PROVISION CAMERAS</span>
          </button>
        </div>
      ) : (
        <div
          className={`grid gap-4 ${
            selectedLayout === '1x1' || selectedLayout === 'FOCUS'
              ? 'grid-cols-1'
              : selectedLayout === '2x2'
              ? 'grid-cols-1 md:grid-cols-2'
              : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
          }`}
        >
          {visibleCameras.map((cam) => {
            const people = cameraDetections[cam.id] || [];

            return (
              <div
                key={cam.id}
                onClick={() => setSelectedCameraId(cam.id)}
                className={`soc-panel rounded-2xl overflow-hidden relative group transition-all aspect-video bg-slate-950 border ${
                  selectedCameraId === cam.id
                    ? 'border-cyan-500/80 shadow-[0_0_24px_rgba(0,240,255,0.2)]'
                    : 'border-slate-800 hover:border-slate-700'
                }`}
              >
                {/* Surveillance Feed Viewport */}
                <div className="w-full h-full flex flex-col items-center justify-center relative surveillance-feed bg-slate-950">
                  <div className="text-center space-y-2 select-none z-0">
                    <CamIcon className="w-12 h-12 text-slate-800 mx-auto group-hover:text-cyan-500/40 transition-colors" />
                    <span className="text-xs font-mono font-bold text-slate-400 block">{cam.name}</span>
                    <span className="text-[10px] font-mono text-slate-600 block truncate max-w-xs">{cam.stream_url}</span>
                  </div>

                  {/* Real Canvas Overlay */}
                  <CameraCanvasOverlay
                    people={people}
                    videoWidth={1280}
                    videoHeight={720}
                    showBoundingBoxes={true}
                    showTrackingIds={true}
                    showActivityLabels={true}
                  />

                  {/* Tactical Corner Brackets */}
                  <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-cyan-400/40 pointer-events-none" />
                  <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-cyan-400/40 pointer-events-none" />
                  <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-cyan-400/40 pointer-events-none" />
                  <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-cyan-400/40 pointer-events-none" />
                </div>

                {/* Header Badges */}
                <div className="absolute top-3 left-3 z-30 flex items-center space-x-2">
                  <span className="px-2 py-0.5 rounded bg-black/80 text-cyan-400 text-xs font-mono font-bold border border-slate-800">
                    {cam.id}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold border border-emerald-500/30 flex items-center space-x-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
                    <span>{cam.status}</span>
                  </span>
                </div>

                {/* Bottom Info Bar */}
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/70 to-transparent p-3 flex justify-between items-center text-xs font-mono text-slate-300 z-30">
                  <span className="font-semibold text-white">{cam.location}</span>
                  <span className="text-cyan-400 font-bold">{people.length} Person{people.length !== 1 ? 's' : ''} Tracked</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
