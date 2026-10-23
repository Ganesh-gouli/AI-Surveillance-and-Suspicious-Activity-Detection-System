import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Volume2,
  VolumeX,
  Radio,
  Cpu,
  Brain,
  Wifi,
  Clock,
  User,
  Activity
} from 'lucide-react';
import { audioService } from '../../services/audioAlerts';
import { SystemMetrics } from '../../types';

interface NavbarProps {
  systemMetrics?: SystemMetrics | null;
  onToggleDemoMode?: () => void;
  isDemoMode?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ systemMetrics, onToggleDemoMode, isDemoMode }) => {
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

  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#06090e]/95 backdrop-blur-md px-6 flex items-center justify-between z-40 select-none">
      {/* Brand & Identity */}
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 p-[1px] flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.3)]">
          <div className="w-full h-full bg-[#06090e] rounded-[11px] flex items-center justify-center">
            <ShieldAlert className="w-5 h-5 text-cyan-400" />
          </div>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-mono font-bold tracking-wider text-sm text-white">
              SENTINEL<span className="text-cyan-400">VISION</span> AI
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono border border-cyan-500/30">
              v2.4.0
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono">
            Human Activity & Behaviour Intelligence
          </p>
        </div>
      </div>

      {/* Center Live Telemetry */}
      <div className="hidden lg:flex items-center space-x-4 bg-slate-900/60 border border-slate-800/80 px-4 py-1.5 rounded-full text-xs font-mono">
        <div className="flex items-center space-x-1.5 text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-bold">AI ENGINE: ONLINE</span>
        </div>
        <span className="text-slate-700">|</span>

        <div className="flex items-center space-x-1.5 text-cyan-400">
          <Brain className="w-3.5 h-3.5" />
          <span>best_human_activity_v2.keras</span>
        </div>
        <span className="text-slate-700">|</span>

        <div className="flex items-center space-x-1.5 text-slate-300">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>{currentTime}</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center space-x-3">
        {/* Audio Siren Toggle */}
        <button
          onClick={handleToggleAudio}
          className={`p-2 rounded-lg border transition-all ${
            isAudioMuted
              ? 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
              : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
          }`}
          title={isAudioMuted ? 'Unmute Security Sirens' : 'Mute Security Sirens'}
        >
          {isAudioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </button>

        {/* Operator Profile */}
        <div className="flex items-center space-x-2.5 bg-slate-900/80 border border-slate-800 py-1.5 px-3 rounded-xl">
          <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <User className="w-4 h-4" />
          </div>
          <div className="text-left font-mono">
            <div className="text-xs font-bold text-slate-200 leading-tight">SOC Operator</div>
            <div className="text-[10px] text-cyan-400 leading-tight">Admin Level</div>
          </div>
        </div>
      </div>
    </header>
  );
};
