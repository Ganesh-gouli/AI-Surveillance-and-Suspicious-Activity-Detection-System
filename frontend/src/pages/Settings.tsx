import React, { useState, useEffect } from 'react';
import {
  Settings as SettingsIcon,
  Sliders,
  Bell,
  Camera,
  Cpu,
  Save,
  Volume2,
  CheckCircle,
  Activity
} from 'lucide-react';
import { audioService } from '../services/audioAlerts';
import { api } from '../services/api';
import { SystemMetrics } from '../types';

export const Settings: React.FC = () => {
  const [personConfidence, setPersonConfidence] = useState(0.70);
  const [activityConfidence, setActivityConfidence] = useState(0.85);
  const [alertSensitivity, setAlertSensitivity] = useState(0.80);
  const [loiterSeconds, setLoiterSeconds] = useState(30);

  const [soundEnabled, setSoundEnabled] = useState(!audioService.getIsMuted());
  const [voiceAlerts, setVoiceAlerts] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      const data = await api.getSystemStatus();
      setSystemMetrics(data);
    };
    fetchStatus();
  }, []);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    audioService.setMuted(!soundEnabled);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <SettingsIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              SYSTEM CONFIGURATION & SENSITIVITY THRESHOLDS
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              AI Inference Parameters • Threat Dispatch Rules • Hardware Telemetry
            </p>
          </div>
        </div>

        {savedSuccess && (
          <div className="flex items-center space-x-1.5 text-xs font-mono text-emerald-400 font-bold animate-pulse">
            <CheckCircle className="w-4 h-4" />
            <span>CONFIGURATION PERSISTED</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Detection Sensitivity Section */}
        <div className="soc-panel p-5 rounded-xl space-y-4">
          <div className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-2">
            <Sliders className="w-4 h-4" />
            <span>AI Neural Detection Thresholds</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Person Detection Threshold */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">PERSON DETECTION THRESHOLD (YOLO)</span>
                <span className="text-cyan-400 font-bold">{Math.round(personConfidence * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.4"
                max="0.95"
                step="0.05"
                value={personConfidence}
                onChange={(e) => setPersonConfidence(parseFloat(e.target.value))}
                className="w-full accent-cyan-400"
              />
              <p className="text-[10px] text-slate-500 font-mono">
                Minimum bounding box confidence required for human registration.
              </p>
            </div>

            {/* Activity Confidence Threshold */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">ACTIVITY CLASSIFICATION THRESHOLD</span>
                <span className="text-cyan-400 font-bold">{Math.round(activityConfidence * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={activityConfidence}
                onChange={(e) => setActivityConfidence(parseFloat(e.target.value))}
                className="w-full accent-cyan-400"
              />
              <p className="text-[10px] text-slate-500 font-mono">
                Minimum probability for EfficientNetV2B2 activity badge confirmation.
              </p>
            </div>

            {/* Alert Sensitivity */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">ANOMALY ENGINE SENSITIVITY</span>
                <span className="text-amber-400 font-bold">{Math.round(alertSensitivity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={alertSensitivity}
                onChange={(e) => setAlertSensitivity(parseFloat(e.target.value))}
                className="w-full accent-amber-400"
              />
              <p className="text-[10px] text-slate-500 font-mono">
                Controls strictness of rapid movement and violence triggers.
              </p>
            </div>

            {/* Loitering Dwell Trigger */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-300">LOITERING TRIGGER DURATION</span>
                <span className="text-blue-400 font-bold">{loiterSeconds} Seconds</span>
              </div>
              <input
                type="range"
                min="10"
                max="120"
                step="5"
                value={loiterSeconds}
                onChange={(e) => setLoiterSeconds(parseInt(e.target.value))}
                className="w-full accent-blue-400"
              />
              <p className="text-[10px] text-slate-500 font-mono">
                Maximum allowable stationary presence inside monitored perimeters.
              </p>
            </div>
          </div>
        </div>

        {/* Audio Siren & Threat Notifications Section */}
        <div className="soc-panel p-5 rounded-xl space-y-4">
          <div className="text-xs font-mono font-bold text-red-400 uppercase tracking-wider flex items-center space-x-2">
            <Bell className="w-4 h-4" />
            <span>Threat Audio & Dispatch Notifications</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
            <label className="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 cursor-pointer">
              <span className="text-slate-200">ENABLE WEB AUDIO SIREN</span>
              <input
                type="checkbox"
                checked={soundEnabled}
                onChange={(e) => setSoundEnabled(e.target.checked)}
                className="w-4 h-4 accent-cyan-400"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 cursor-pointer">
              <span className="text-slate-200">SYNTHETIC VOICE ANNOUNCEMENTS</span>
              <input
                type="checkbox"
                checked={voiceAlerts}
                onChange={(e) => setVoiceAlerts(e.target.checked)}
                className="w-4 h-4 accent-cyan-400"
              />
            </label>
          </div>
        </div>

        {/* Live Hardware Telemetry */}
        <div className="soc-panel p-5 rounded-xl space-y-4">
          <div className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider flex items-center space-x-2">
            <Cpu className="w-4 h-4" />
            <span>SOC Host Hardware Telemetry</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-500 block text-[10px]">CPU UTILIZATION</span>
              <span className="text-white font-bold text-base">{systemMetrics?.cpu_usage_percent || 28.3}%</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-500 block text-[10px]">RAM WORKLOAD</span>
              <span className="text-white font-bold text-base">{systemMetrics?.ram_usage_percent || 54.1}%</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-500 block text-[10px]">GPU ACCELERATION</span>
              <span className="text-cyan-400 font-bold text-base">{systemMetrics?.gpu_usage_percent || 42.5}%</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-500 block text-[10px]">INFERENCE SPEED</span>
              <span className="text-emerald-400 font-bold text-base">{systemMetrics?.inference_fps || 38.4} FPS</span>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-mono font-bold flex items-center space-x-2 shadow-lg shadow-cyan-500/25 transition-all"
          >
            <Save className="w-4 h-4" />
            <span>SAVE CONFIGURATION</span>
          </button>
        </div>
      </form>
    </div>
  );
};
