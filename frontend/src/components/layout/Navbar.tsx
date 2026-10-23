import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Volume2,
  VolumeX,
  Cpu,
  Brain,
  Clock,
  User,
  Activity,
  Menu,
  X,
  Server,
  Layers
} from 'lucide-react';
import { audioService } from '../../services/audioAlerts';
import { SystemMetrics } from '../../types';

interface NavbarProps {
  systemMetrics?: SystemMetrics | null;
  onToggleDemoMode?: () => void;
  isDemoMode?: boolean;
  onToggleMobileMenu?: () => void;
  isMobileMenuOpen?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  systemMetrics,
  onToggleDemoMode,
  isDemoMode,
  onToggleMobileMenu,
  isMobileMenuOpen = false
}) => {
  const [currentTime, setCurrentTime] = useState<string>('');
  const [isAudioMuted, setIsAudioMuted] = useState(false);

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString('en-US', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      );
    };
    updateClock();
    const timer = setInterval(updateClock, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleToggleAudio = () => {
    const muted = audioService.toggleMute();
    setIsAudioMuted(muted);
  };

  const isOnline = systemMetrics?.ai_engine_status === 'ONLINE' || !systemMetrics;
  const cpuLoad = systemMetrics?.cpu_usage_percent ?? 0;
  const ramLoad = systemMetrics?.ram_usage_percent ?? 0;
  const connectedCams = systemMetrics?.connected_cameras ?? 0;
  const totalCams = systemMetrics?.total_cameras ?? 0;

  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#06090e]/95 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between z-40 select-none sticky top-0">
      {/* Brand & Identity + Mobile Menu Button */}
      <div className="flex items-center space-x-3">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white md:hidden transition-colors"
            aria-label="Toggle Navigation"
          >
            {isMobileMenuOpen ? <X className="w-5 h-5 text-cyan-400" /> : <Menu className="w-5 h-5" />}
          </button>
        )}

        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-500 to-indigo-600 p-[1px] flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.25)]">
          <div className="w-full h-full bg-[#06090e] rounded-[11px] flex items-center justify-center">
            <ShieldAlert className="w-5 h-5 text-cyan-400 animate-pulse" />
          </div>
        </div>

        <div>
          <div className="flex items-center space-x-2">
            <span className="font-mono font-bold tracking-wider text-sm text-white">
              SENTINEL<span className="text-cyan-400">VISION</span> AI
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono font-bold border border-cyan-500/30">
              v2.4.0
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono hidden sm:block">
            Autonomous Neural Surveillance & Threat Defense
          </p>
        </div>
      </div>

      {/* Center Live Telemetry (Dynamic Real Data) */}
      <div className="hidden xl:flex items-center space-x-4 bg-slate-900/70 border border-slate-800/90 px-4 py-1.5 rounded-full text-xs font-mono shadow-inner">
        {/* Engine Status */}
        <div className="flex items-center space-x-1.5 text-emerald-400">
          <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
          <span className="font-bold">{isOnline ? 'ENGINE: ONLINE' : 'ENGINE: OFFLINE'}</span>
        </div>
        <span className="text-slate-700">|</span>

        {/* Real Hardware CPU & RAM Metrics */}
        {cpuLoad > 0 && (
          <>
            <div className="flex items-center space-x-1.5 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>CPU: <strong className="text-white">{cpuLoad}%</strong></span>
              <span className="text-slate-600">/</span>
              <span>RAM: <strong className="text-white">{ramLoad}%</strong></span>
            </div>
            <span className="text-slate-700">|</span>
          </>
        )}

        {/* Real Cameras telemetry */}
        <div className="flex items-center space-x-1.5 text-slate-300">
          <Layers className="w-3.5 h-3.5 text-purple-400" />
          <span>FEEDS: <strong className="text-cyan-300">{connectedCams}/{totalCams}</strong></span>
        </div>
        <span className="text-slate-700">|</span>

        {/* Time */}
        <div className="flex items-center space-x-1.5 text-slate-300">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span className="tabular-nums font-semibold">{currentTime}</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center space-x-2.5">
        {/* Audio Siren Toggle */}
        <button
          onClick={handleToggleAudio}
          className={`p-2 rounded-xl border transition-all ${
            isAudioMuted
              ? 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300'
              : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 shadow-[0_0_12px_rgba(0,240,255,0.2)]'
          }`}
          title={isAudioMuted ? 'Security Sirens Muted (Click to Unmute)' : 'Security Sirens Active (Click to Mute)'}
          aria-label="Toggle Siren Audio"
        >
          {isAudioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </button>

        {/* Operator Profile */}
        <div className="flex items-center space-x-2.5 bg-slate-900/90 border border-slate-800/90 py-1.5 px-3 rounded-xl shadow-sm">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-500/20 to-blue-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <User className="w-3.5 h-3.5" />
          </div>
          <div className="text-left font-mono">
            <div className="text-xs font-bold text-slate-200 leading-tight">SOC Operator</div>
            <div className="text-[10px] text-cyan-400 leading-tight flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 inline-block animate-ping" />
              <span>Admin Verified</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
