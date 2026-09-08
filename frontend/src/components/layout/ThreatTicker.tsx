import React from 'react';
import { AlertTriangle, ChevronRight, ShieldAlert } from 'lucide-react';
import { Incident } from '../../types';

interface ThreatTickerProps {
  incidents: Incident[];
  onSelectIncident: (incident: Incident) => void;
}

export const ThreatTicker: React.FC<ThreatTickerProps> = ({ incidents, onSelectIncident }) => {
  const activeThreats = incidents.filter(i => i.status !== 'RESOLVED').slice(0, 4);

  if (activeThreats.length === 0) return null;

  return (
    <div className="bg-gradient-to-r from-red-950/40 via-slate-900/80 to-slate-900/60 border-y border-red-900/40 px-4 py-2 flex items-center justify-between overflow-hidden">
      <div className="flex items-center space-x-2 text-xs font-mono text-red-400 font-bold shrink-0">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
        </span>
        <ShieldAlert className="w-4 h-4 text-red-400" />
        <span>LIVE THREAT FEED:</span>
      </div>

      <div className="flex items-center space-x-6 overflow-x-auto no-scrollbar mx-4 py-0.5">
        {activeThreats.map((threat) => {
          const isCritical = threat.severity === 'CRITICAL';
          return (
            <div
              key={threat.id}
              onClick={() => onSelectIncident(threat)}
              className={`flex items-center space-x-2 text-xs cursor-pointer px-2.5 py-1 rounded transition-colors whitespace-nowrap border ${
                isCritical
                  ? 'bg-red-500/10 border-red-500/30 hover:bg-red-500/20 text-red-200'
                  : 'bg-amber-500/10 border-amber-500/30 hover:bg-amber-500/20 text-amber-200'
              }`}
            >
              <span className="font-mono font-bold text-[10px] text-slate-400">
                {new Date(threat.timestamp).toLocaleTimeString()}
              </span>
              <span className="font-semibold text-cyan-300 font-mono">[{threat.camera_id}]</span>
              <span className="font-bold">{threat.activity}</span>
              <span
                className={`text-[9px] px-1 py-0.2 rounded font-mono font-bold ${
                  isCritical ? 'bg-red-600 text-white' : 'bg-amber-600 text-black'
                }`}
              >
                {threat.severity}
              </span>
              <ChevronRight className="w-3 h-3 text-slate-400" />
            </div>
          );
        })}
      </div>
    </div>
  );
};
