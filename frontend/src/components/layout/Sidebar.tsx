import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Tv,
  Camera,
  Video,
  Image,
  AlertTriangle,
  Layers,
  BarChart3,
  Cpu,
  Settings,
  Webcam,
  Flame,
  Swords,
  Crosshair
} from 'lucide-react';

interface SidebarProps {
  activeThreatCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeThreatCount = 4 }) => {
  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/shooting-detection', label: 'Shooting & Weapon AI', icon: Crosshair, badge: '🎯 NEW' },
    { to: '/fighting-detection', label: 'Fighting Detection AI', icon: Swords, badge: '🥊' },
    { to: '/video-analysis', label: 'Fire Detection AI', icon: Flame, badge: '🔥' },
    { to: '/webcam', label: 'Live Webcam', icon: Webcam, badge: 'AI' },
    { to: '/monitoring', label: 'Live Monitoring', icon: Tv, badge: 'LIVE' },
    { to: '/cameras', label: 'CCTV / Cameras', icon: Camera },
    { to: '/image-analysis', label: 'Image Analysis', icon: Image },
    { to: '/incidents', label: 'Incidents & Threats', icon: AlertTriangle, threatBadge: activeThreatCount },
    { to: '/zones', label: 'Restricted Zones', icon: Layers },
    { to: '/analytics', label: 'Heatmaps & Analytics', icon: BarChart3 },
    { to: '/models', label: 'AI Model Hub', icon: Cpu },
    { to: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-[#080d15] border-r border-slate-800/80 flex flex-col justify-between select-none flex-shrink-0">
      <div className="py-4 px-3 space-y-1">
        <div className="px-3 pb-2 text-[10px] font-mono font-bold tracking-widest text-slate-500 uppercase">
          Surveillance Console
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all group ${
                  isActive
                    ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-[0_0_12px_rgba(0,240,255,0.15)] font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`
              }
            >
              <div className="flex items-center space-x-3">
                <Icon className="w-4 h-4 transition-colors group-hover:text-cyan-400" />
                <span>{item.label}</span>
              </div>

              {item.badge && (
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-400 font-mono font-bold">
                  {item.badge}
                </span>
              )}

              {item.threatBadge !== undefined && item.threatBadge > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-red-500/20 border border-red-500/40 text-red-400 font-mono font-bold animate-pulse">
                  {item.threatBadge} ALERTS
                </span>
              )}
            </NavLink>
          );
        })}
      </div>

      {/* Footer System Status Mini */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 m-2 rounded-lg text-[11px] font-mono space-y-2">
        <div className="flex justify-between items-center text-slate-400">
          <span>PIPELINE</span>
          <span className="text-emerald-400 font-bold">YOLOv8 + Keras</span>
        </div>
        <div className="flex justify-between items-center text-slate-400">
          <span>ACCURACY</span>
          <span className="text-cyan-400 font-bold">87.38% (8 cls)</span>
        </div>
        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
          <div className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full w-[87%]" />
        </div>
      </div>
    </aside>
  );
};
