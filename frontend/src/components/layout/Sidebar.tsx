import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Tv,
  Camera,
  Image,
  AlertTriangle,
  Layers,
  BarChart3,
  Cpu,
  Settings,
  Webcam,
  Flame,
  Swords,
  Crosshair,
  Shield,
  Activity,
  X
} from 'lucide-react';

interface SidebarProps {
  activeThreatCount?: number;
  isOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeThreatCount = 0,
  isOpen = false,
  onClose
}) => {
  const navSections = [
    {
      heading: 'Command Core',
      items: [
        { to: '/', label: 'Dashboard', icon: LayoutDashboard },
      ]
    },
    {
      heading: 'Neural AI Engines',
      items: [
        { to: '/shooting-detection', label: 'Shooting & Weapon AI', icon: Crosshair, badge: 'WEAPON' },
        { to: '/fighting-detection', label: 'Fighting Detection AI', icon: Swords, badge: 'COMBAT' },
        { to: '/video-analysis', label: 'Fire Detection AI', icon: Flame, badge: 'THERMAL' },
        { to: '/webcam', label: 'Live Webcam AI', icon: Webcam, badge: 'REAL-TIME' },
      ]
    },
    {
      heading: 'Surveillance & Feeds',
      items: [
        { to: '/monitoring', label: 'Live Monitoring Wall', icon: Tv, badge: 'WALL' },
        { to: '/cameras', label: 'CCTV / IP Cameras', icon: Camera },
        { to: '/image-analysis', label: 'Image Forensic AI', icon: Image },
      ]
    },
    {
      heading: 'Threat Intel & Zones',
      items: [
        {
          to: '/incidents',
          label: 'Incidents & Audit Log',
          icon: AlertTriangle,
          threatBadge: activeThreatCount
        },
        { to: '/zones', label: 'Restricted Zones', icon: Layers },
        { to: '/analytics', label: 'Heatmaps & Analytics', icon: BarChart3 },
      ]
    },
    {
      heading: 'Infrastructure',
      items: [
        { to: '/models', label: 'AI Model Hub', icon: Cpu },
        { to: '/settings', label: 'System Settings', icon: Settings },
      ]
    }
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 md:hidden animate-in fade-in duration-200"
        />
      )}

      {/* Main Sidebar */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-64 bg-[#080d16] border-r border-slate-800/80 flex flex-col justify-between select-none flex-shrink-0 transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Navigation list */}
        <div className="py-4 px-3 space-y-4 overflow-y-auto flex-1 custom-scrollbar">
          {/* Mobile Header Close */}
          <div className="flex items-center justify-between px-2 pb-1 md:hidden">
            <span className="text-xs font-mono font-bold text-cyan-400">NAVIGATION MENU</span>
            <button
              onClick={onClose}
              className="p-1 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {navSections.map((sec, secIdx) => (
            <div key={secIdx} className="space-y-1">
              <div className="px-3 text-[10px] font-mono font-bold tracking-widest text-slate-500 uppercase">
                {sec.heading}
              </div>

              {sec.items.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    onClick={() => {
                      if (onClose) onClose();
                    }}
                    className={({ isActive }) =>
                      `flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all group ${
                        isActive
                          ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/35 shadow-[0_0_14px_rgba(0,240,255,0.14)] font-semibold'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                      }`
                    }
                  >
                    <div className="flex items-center space-x-2.5 min-w-0">
                      <Icon className="w-4 h-4 shrink-0 transition-colors group-hover:text-cyan-400" />
                      <span className="truncate">{item.label}</span>
                    </div>

                    {item.badge && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/25 text-cyan-400 font-mono font-bold tracking-tight shrink-0">
                        {item.badge}
                      </span>
                    )}

                    {item.threatBadge !== undefined && item.threatBadge > 0 && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 border border-red-500/40 text-red-400 font-mono font-bold shrink-0 animate-pulse">
                        {item.threatBadge}
                      </span>
                    )}
                  </NavLink>
                );
              })}
            </div>
          ))}
        </div>

        {/* Footer Real Pipeline Telemetry Mini */}
        <div className="p-3.5 border-t border-slate-800/80 bg-slate-950/60 m-2 rounded-xl text-[11px] font-mono space-y-2 border border-slate-800/40">
          <div className="flex justify-between items-center text-slate-400">
            <span className="text-[10px] tracking-wider text-slate-500 font-bold uppercase">Vision Pipeline</span>
            <span className="text-emerald-400 font-bold text-[10px] flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block animate-pulse" />
              <span>YOLOv11</span>
            </span>
          </div>
          <div className="text-[10px] text-slate-400 leading-snug">
            Keras 3.x • BiLSTM Combat • Fire CNN
          </div>
          <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden border border-slate-800">
            <div className="bg-gradient-to-r from-cyan-500 via-blue-500 to-emerald-400 h-full w-[100%]" />
          </div>
        </div>
      </aside>
    </>
  );
};
