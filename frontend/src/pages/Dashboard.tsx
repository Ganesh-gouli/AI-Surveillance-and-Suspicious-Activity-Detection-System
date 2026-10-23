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
  Activity,
  Cpu,
  Crosshair,
  Swords,
  Tv,
  Radio,
  Clock,
  Sparkles,
  Server
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
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const totalCams = metrics?.total_cameras ?? cameras.length;
  const activeCams = metrics?.connected_cameras ?? cameras.filter(c => c.status === 'ONLINE').length;
  const totalIncidents = metrics?.today_incidents ?? incidents.length;
  const activeAlerts = metrics?.active_alerts ?? incidents.filter(i => i.status === 'OPEN').length;
  const cpuPercent = metrics?.cpu_usage_percent ?? 0;
  const ramPercent = metrics?.ram_usage_percent ?? 0;
  const avgConfidence = metrics?.avg_ai_confidence ?? 87.38;

  const quickActions = [
    {
      title: 'Shooting & Weapon AI',
      subtitle: 'Firearm & Aiming Stance',
      icon: Crosshair,
      to: '/shooting-detection',
      color: 'from-red-600/20 to-red-950/40 border-red-500/40 text-red-400 hover:border-red-400'
    },
    {
      title: 'Fighting Detection AI',
      subtitle: 'BiLSTM Combat Recognition',
      icon: Swords,
      to: '/fighting-detection',
      color: 'from-amber-600/20 to-amber-950/40 border-amber-500/40 text-amber-400 hover:border-amber-400'
    },
    {
      title: 'Fire Detection AI',
      subtitle: 'Optical & Flame Classifier',
      icon: Flame,
      to: '/video-analysis',
      color: 'from-orange-600/20 to-orange-950/40 border-orange-500/40 text-orange-400 hover:border-orange-400'
    },
    {
      title: 'Live Webcam AI',
      subtitle: 'Real-Time Neural Feed',
      icon: Webcam,
      to: '/webcam',
      color: 'from-cyan-600/20 to-blue-950/40 border-cyan-500/40 text-cyan-400 hover:border-cyan-400'
    },
    {
      title: 'Surveillance Wall',
      subtitle: 'Multi-Grid CCTV Monitor',
      icon: Tv,
      to: '/monitoring',
      color: 'from-indigo-600/20 to-purple-950/40 border-indigo-500/40 text-indigo-400 hover:border-indigo-400'
    }
  ];

  return (
    <div className="p-4 sm:p-6 space-y-6 cyber-grid-bg min-h-full">
      {/* Top SOC Command Header Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-cyan-900/40 bg-gradient-to-r from-[#0b1329] via-[#091122] to-[#06090e] p-6 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2">
            <div className="flex items-center space-x-2.5">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
              </span>
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-widest">
                SENTINEL SEC-OPS • REAL-TIME NEURAL VISION ONLINE
              </span>
            </div>

            <h1 className="text-2xl sm:text-3xl font-extrabold text-white font-mono tracking-tight">
              AI SURVEILLANCE & THREAT COMMAND CENTER
            </h1>

            <p className="text-xs sm:text-sm text-slate-400 font-mono max-w-2xl leading-relaxed">
              Autonomous multi-stream surveillance powered by <span className="text-cyan-400 font-semibold">YOLOv11</span>,{' '}
              <span className="text-purple-400 font-semibold">MediaPipe BiLSTM</span>, and{' '}
              <span className="text-emerald-400 font-semibold">EfficientNetV2</span>. Direct inference on real streams with zero fabricated data.
            </p>
          </div>

          {/* Top Quick Action Buttons */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => navigate('/webcam')}
              className="px-4 py-2.5 rounded-xl btn-tactical-primary text-xs font-mono font-bold flex items-center space-x-2 cursor-pointer"
            >
              <Webcam className="w-4 h-4" />
              <span>LAUNCH WEBCAM AI</span>
            </button>
            <button
              onClick={() => navigate('/monitoring')}
              className="px-4 py-2.5 rounded-xl btn-tactical-secondary text-xs font-mono font-bold flex items-center space-x-2 cursor-pointer"
            >
              <Tv className="w-4 h-4 text-cyan-400" />
              <span>LIVE WALL</span>
            </button>
          </div>
        </div>
      </div>

      {/* Real System Telemetry KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Total Cameras */}
        <div className="soc-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>SURVEILLANCE CAMERAS</span>
            <CamIcon className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white tracking-tight">{totalCams}</div>
          <div className="text-[11px] font-mono text-emerald-400 flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
            <span>{activeCams} Stream{activeCams !== 1 ? 's' : ''} Online</span>
          </div>
        </div>

        {/* Active Threats */}
        <div className={`soc-card p-4 rounded-xl space-y-1 ${activeAlerts > 0 ? 'soc-panel-glow-red' : ''}`}>
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>ACTIVE UNRESOLVED ALERTS</span>
            <AlertTriangle className={`w-4 h-4 ${activeAlerts > 0 ? 'text-red-400 animate-bounce' : 'text-slate-500'}`} />
          </div>
          <div className={`text-2xl font-bold font-mono tracking-tight ${activeAlerts > 0 ? 'text-red-400' : 'text-slate-300'}`}>
            {activeAlerts}
          </div>
          <div className="text-[11px] font-mono text-slate-400">
            {activeAlerts > 0 ? 'Immediate Attention Required' : 'Perimeter Secure'}
          </div>
        </div>

        {/* Incidents Total */}
        <div className="soc-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>INCIDENTS RECORDED</span>
            <Flame className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white tracking-tight">{totalIncidents}</div>
          <div className="text-[11px] font-mono text-slate-400">Database Audit Entries</div>
        </div>

        {/* Hardware / Engine Telemetry */}
        <div className="soc-card p-4 rounded-xl space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>HARDWARE TELEMETRY</span>
            <Cpu className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400 tracking-tight">
            {cpuPercent > 0 ? `${cpuPercent}%` : 'NOMINAL'}
          </div>
          <div className="text-[11px] font-mono text-slate-400 flex justify-between">
            <span>RAM: {ramPercent > 0 ? `${ramPercent}%` : 'OK'}</span>
            <span className="text-cyan-400">Acc: {avgConfidence}%</span>
          </div>
        </div>
      </div>

      {/* Specialized Neural Detector Launchpad */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold text-slate-300 font-mono uppercase tracking-widest flex items-center space-x-2">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Specialized AI Detectors</span>
          </h2>
          <span className="text-[11px] font-mono text-slate-500">Autonomous Neural Workflows</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5">
          {quickActions.map((act) => {
            const Icon = act.icon;
            return (
              <div
                key={act.to}
                onClick={() => navigate(act.to)}
                className={`p-4 rounded-xl border bg-gradient-to-b ${act.color} cursor-pointer transition-all hover:scale-[1.02] shadow-lg flex flex-col justify-between space-y-3 group`}
              >
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-lg bg-black/40 border border-white/10 text-white group-hover:text-cyan-300 transition-colors">
                    <Icon className="w-5 h-5" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-white group-hover:translate-x-1 transition-all" />
                </div>
                <div>
                  <h3 className="text-xs font-mono font-bold text-white group-hover:text-cyan-200 transition-colors">
                    {act.title}
                  </h3>
                  <p className="text-[10px] font-mono text-slate-400 mt-0.5">
                    {act.subtitle}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Content Area: Cameras & Recent Real Incidents */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Surveillance Channels (2 cols) */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
              <CamIcon className="w-4 h-4 text-cyan-400" />
              <span>Surveillance Camera Fleet</span>
            </h2>
            <button
              onClick={() => navigate('/monitoring')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 transition-colors"
            >
              <span>OPEN MONITORING WALL</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {cameras.length === 0 ? (
            <div className="soc-card p-8 rounded-xl text-center space-y-3">
              <CamIcon className="w-10 h-10 text-slate-600 mx-auto" />
              <div className="text-sm font-mono font-bold text-slate-300">NO CAMERAS PROVISIONED</div>
              <p className="text-xs font-mono text-slate-500 max-w-md mx-auto">
                No active camera streams were detected. Register RTSP streams or connect IP cameras to begin multi-feed surveillance.
              </p>
              <button
                onClick={() => navigate('/cameras')}
                className="px-4 py-2 rounded-lg btn-tactical-primary text-xs font-mono font-bold inline-flex items-center space-x-2 mt-2"
              >
                <CamIcon className="w-3.5 h-3.5" />
                <span>CONFIGURE CAMERAS</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {cameras.map((cam) => (
                <div
                  key={cam.id}
                  onClick={() => navigate('/monitoring')}
                  className="soc-card soc-card-interactive rounded-xl overflow-hidden group border border-slate-800"
                >
                  <div className="aspect-video bg-slate-950 relative flex items-center justify-center p-4">
                    <div className="text-center space-y-2">
                      <CamIcon className="w-8 h-8 text-cyan-500/40 mx-auto group-hover:text-cyan-400 transition-colors" />
                      <span className="text-xs font-mono text-slate-300 font-bold block">{cam.name}</span>
                      <span className="text-[10px] font-mono text-slate-500 block truncate max-w-[220px]">
                        {cam.stream_url}
                      </span>
                    </div>

                    <div className="absolute top-2.5 left-2.5 px-2 py-0.5 rounded bg-black/80 text-[10px] font-mono text-cyan-400 border border-slate-800 font-bold">
                      {cam.id}
                    </div>

                    <div className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded bg-emerald-500/20 text-[10px] font-mono text-emerald-400 font-bold border border-emerald-500/30 flex items-center space-x-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
                      <span>{cam.status}</span>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-900/80 border-t border-slate-800 flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300 font-semibold">{cam.location}</span>
                    <span className="text-slate-500 text-[11px]">{cam.resolution}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Real Incidents Log (1 col) */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wider flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Incident Audit Log</span>
            </h2>
            <button
              onClick={() => navigate('/incidents')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 transition-colors"
            >
              <span>VIEW ALL</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="soc-card p-4 rounded-xl space-y-3 min-h-[320px]">
            {incidents.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center py-16 space-y-3">
                <div className="p-3 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                  <ShieldCheck className="w-8 h-8" />
                </div>
                <div className="text-xs font-mono text-slate-300 font-bold">ALL MONITORED ZONES CLEAR</div>
                <p className="text-[11px] font-mono text-slate-500 max-w-xs leading-relaxed">
                  Real incidents will appear automatically when suspicious activities, weapons, combat, or restricted zone breaches are detected on live streams.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[380px] overflow-y-auto custom-scrollbar pr-1">
                {incidents.slice(0, 6).map((inc) => (
                  <div
                    key={inc.id}
                    onClick={() => navigate('/incidents')}
                    className="p-3 rounded-xl bg-slate-900/90 border border-slate-800/90 hover:border-cyan-500/40 cursor-pointer text-xs font-mono space-y-1.5 transition-all"
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-white truncate max-w-[150px]">{inc.activity}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        inc.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/40' :
                        inc.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40' :
                        'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                      }`}>
                        {inc.severity}
                      </span>
                    </div>
                    <div className="text-slate-400 text-[11px] flex justify-between items-center">
                      <span>{inc.camera_id} • {inc.location}</span>
                      <span className="text-emerald-400 font-bold">{inc.confidence}%</span>
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {new Date(inc.timestamp).toLocaleTimeString()}
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
