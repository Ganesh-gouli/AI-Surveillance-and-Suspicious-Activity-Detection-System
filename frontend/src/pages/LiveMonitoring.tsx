import React, { useState, useEffect } from 'react';
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
  Layers
} from 'lucide-react';
import { api } from '../services/api';
import { Camera, TrackedPerson } from '../types';
import { CameraCanvasOverlay } from '../components/video/CameraCanvasOverlay';

export const LiveMonitoring: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedLayout, setSelectedLayout] = useState<'1x1' | '2x2' | '3x3' | 'FOCUS'>('2x2');
  const [selectedCameraId, setSelectedCameraId] = useState<string>('CAM-01');
  const [cameraDetections, setCameraDetections] = useState<Record<string, TrackedPerson[]>>({});
  const [activeAlerts, setActiveAlerts] = useState<any[]>([]);

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

  const visibleCameras = selectedLayout === '1x1'
    ? cameras.filter(c => c.id === selectedCameraId)
    : selectedLayout === 'FOCUS'
    ? cameras.filter(c => c.id === selectedCameraId)
    : cameras.slice(0, selectedLayout === '2x2' ? 4 : 9);

  return (
    <div className="p-6 space-y-6">
      {/* Top Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <CamIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              LIVE SURVEILLANCE & BEHAVIOUR MONITORING WALL
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Real Neural Stream • <span className="text-purple-400 font-bold">YOLOv11 Multi-Object</span> + <span className="text-cyan-400 font-bold">best_human_activity_v2.keras</span>
            </p>
          </div>
        </div>

        {/* Layout Switcher */}
        <div className="flex items-center space-x-2 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
          {(['1x1', '2x2', '3x3', 'FOCUS'] as const).map((layout) => (
            <button
              key={layout}
              onClick={() => setSelectedLayout(layout)}
              className={`px-3 py-1.5 rounded-md text-xs font-mono font-bold transition-all ${
                selectedLayout === layout
                  ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {layout}
            </button>
          ))}
        </div>
      </div>

      {/* Grid Container */}
      <div className={`grid gap-4 ${
        selectedLayout === '1x1' || selectedLayout === 'FOCUS'
          ? 'grid-cols-1'
          : selectedLayout === '2x2'
          ? 'grid-cols-1 md:grid-cols-2'
          : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'
      }`}>
        {visibleCameras.map((cam) => {
          const people = cameraDetections[cam.id] || [];

          return (
            <div
              key={cam.id}
              onClick={() => setSelectedCameraId(cam.id)}
              className={`soc-panel rounded-2xl overflow-hidden relative group transition-all aspect-video bg-slate-950 border ${
                selectedCameraId === cam.id
                  ? 'border-cyan-500/80 shadow-[0_0_20px_rgba(0,240,255,0.15)]'
                  : 'border-slate-800'
              }`}
            >
              {/* Surveillance Feed Viewport */}
              <div className="w-full h-full flex flex-col items-center justify-center relative">
                <CamIcon className="w-12 h-12 text-slate-800 mb-2" />
                <span className="text-xs font-mono text-slate-400">{cam.name}</span>
                <span className="text-[11px] font-mono text-slate-600">{cam.stream_url}</span>

                {/* Real Canvas Overlay */}
                <CameraCanvasOverlay
                  people={people}
                  videoWidth={1280}
                  videoHeight={720}
                  showBoundingBoxes={true}
                  showTrackingIds={true}
                  showActivityLabels={true}
                />
              </div>

              {/* Header Badges */}
              <div className="absolute top-3 left-3 z-30 flex items-center space-x-2">
                <span className="px-2 py-1 rounded bg-black/80 text-cyan-400 text-xs font-mono font-bold border border-slate-800">
                  {cam.id}
                </span>
                <span className="px-2 py-1 rounded bg-emerald-500/20 text-emerald-400 text-[11px] font-mono font-bold">
                  {cam.status}
                </span>
              </div>

              {/* Bottom Info Bar */}
              <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-3 flex justify-between items-center text-xs font-mono text-slate-300">
                <span>{cam.location}</span>
                <span className="text-cyan-400">{people.length} People Detected</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
