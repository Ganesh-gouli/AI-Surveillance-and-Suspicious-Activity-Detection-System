import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Camera as CamIcon,
  Users,
  AlertTriangle,
  Flame,
  Brain,
  Webcam,
  Upload,
  Layers,
  ArrowRight,
  ShieldCheck,
  Activity
} from 'lucide-react';
import { api } from '../services/api';
import { Camera, Incident, SystemMetrics } from '../types';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [camsData, incsData, sysData] = await Promise.all([
          api.getCameras(),
          api.getIncidents(),
          api.getSystemStatus(),
        ]);
        setCameras(camsData);
        setIncidents(incsData);
        setMetrics(sysData);
      } catch (err) {
        console.error("Dashboard data load error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, []);

  const totalCams = metrics?.total_cameras ?? cameras.length;
  const activeCams = metrics?.connected_cameras ?? cameras.filter(c => c.status === 'ONLINE').length;
  const totalIncidents = metrics?.today_incidents ?? incidents.length;
  const activeAlerts = metrics?.active_alerts ?? incidents.filter(i => i.status === 'OPEN').length;

  return (
    <div className="p-6 space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-gradient-to-r from-[#0b1329] via-[#091020] to-[#06090e] border border-cyan-950/60 p-5 rounded-2xl shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-widest">
              AI INFERENCE ENGINE ACTIVE • REAL-TIME VISION
            </span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold text-white font-mono tracking-tight">
            SENTINELVISION AI COMMAND CENTER
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            <span className="text-purple-400 font-bold">YOLOv11 Multi-Object Vision</span> • Direct Inference on <span className="text-cyan-400 font-bold">best_human_activity_v2.keras</span> • Zero Fake Data
          </p>
        </div>

        {/* Quick Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => navigate('/webcam')}
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-mono font-bold flex items-center space-x-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
          >
            <Webcam className="w-4 h-4" />
            <span>START WEBCAM AI</span>
          </button>
          <button
            onClick={() => navigate('/video-analysis')}
            className="px-4 py-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-white text-xs font-mono font-bold flex items-center space-x-2 border border-slate-700 transition-all cursor-pointer"
          >
            <Upload className="w-4 h-4 text-cyan-400" />
            <span>ANALYZE VIDEO</span>
          </button>
        </div>
      </div>

      {/* Real KPI Summary Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="soc-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>TOTAL CAMERAS</span>
            <CamIcon className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">{totalCams}</div>
          <div className="text-[11px] font-mono text-emerald-400">{activeCams} Active / Online</div>
        </div>

        <div className="soc-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>ACTIVE ALERTS</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">{activeAlerts}</div>
          <div className="text-[11px] font-mono text-slate-400">Unresolved Threats</div>
        </div>

        <div className="soc-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>INCIDENTS LOGGED</span>
            <Flame className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">{totalIncidents}</div>
          <div className="text-[11px] font-mono text-slate-400">Total Recorded</div>
        </div>

        <div className="soc-panel p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>MODEL ACCURACY</span>
            <Brain className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">87.38%</div>
          <div className="text-[11px] font-mono text-slate-400">EfficientNetV2-B2</div>
        </div>
      </div>

      {/* Main Content Area: Cameras & Recent Real Incidents */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Surveillance Channels (2 cols) */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
              <CamIcon className="w-4 h-4 text-cyan-400" />
              <span>Surveillance Camera Feeds</span>
            </h2>
            <button
              onClick={() => navigate('/monitoring')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
            >
              <span>OPEN MONITORING WALL</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {cameras.map((cam) => (
              <div
                key={cam.id}
                onClick={() => navigate('/monitoring')}
                className="soc-panel rounded-xl overflow-hidden cursor-pointer hover:border-cyan-500/50 transition-all group"
              >
                <div className="aspect-video bg-slate-950 relative flex items-center justify-center p-4">
                  <div className="text-center space-y-2">
                    <CamIcon className="w-8 h-8 text-cyan-500/40 mx-auto group-hover:text-cyan-400 transition-colors" />
                    <span className="text-xs font-mono text-slate-400 block">{cam.name}</span>
                  </div>
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-cyan-400 border border-slate-800">
                    {cam.id}
                  </div>
                  <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-emerald-500/20 text-[10px] font-mono text-emerald-400 font-bold">
                    {cam.status}
                  </div>
                </div>
                <div className="p-3 bg-slate-900/60 border-t border-slate-800 flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300">{cam.location}</span>
                  <span className="text-slate-500">{cam.resolution}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Real Incidents Log (1 col) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Real Incident History</span>
            </h2>
            <button
              onClick={() => navigate('/incidents')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
            >
              <span>VIEW ALL</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="soc-panel p-4 rounded-xl space-y-3 min-h-[300px]">
            {incidents.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center py-12 space-y-2">
                <ShieldCheck className="w-10 h-10 text-slate-600" />
                <span className="text-xs font-mono text-slate-400 font-bold">NO INCIDENTS RECORDED</span>
                <p className="text-[11px] font-mono text-slate-500 max-w-xs">
                  Incidents will appear here automatically when suspicious activity or restricted zone intrusions are detected on live feeds.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {incidents.map((inc) => (
                  <div
                    key={inc.id}
                    onClick={() => navigate('/incidents')}
                    className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-slate-700 cursor-pointer text-xs font-mono space-y-1"
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-white">{inc.activity}</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        inc.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' :
                        inc.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                        'bg-amber-500/20 text-amber-400'
                      }`}>
                        {inc.severity}
                      </span>
                    </div>
                    <div className="text-slate-400 text-[11px] flex justify-between">
                      <span>{inc.camera_id} • {inc.location}</span>
                      <span>{inc.confidence}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
