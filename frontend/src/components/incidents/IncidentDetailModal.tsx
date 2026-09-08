import React, { useState } from 'react';
import {
  X,
  ShieldAlert,
  Clock,
  Camera,
  MapPin,
  CheckCircle,
  FileDown,
  UserCheck,
  Cpu,
  Activity,
  AlertOctagon,
  Share2
} from 'lucide-react';
import { Incident } from '../../types';

interface IncidentDetailModalProps {
  incident: Incident | null;
  onClose: () => void;
  onResolve: (id: string, notes: string) => void;
}

export const IncidentDetailModal: React.FC<IncidentDetailModalProps> = ({
  incident,
  onClose,
  onResolve
}) => {
  if (!incident) return null;

  const [notes, setNotes] = useState(incident.resolution_notes || '');
  const [isResolving, setIsResolving] = useState(false);

  const handleResolve = () => {
    onResolve(incident.id, notes || 'Verified and resolved by Security Officer.');
    setIsResolving(false);
  };

  const isCritical = incident.severity === 'CRITICAL';
  const isHigh = incident.severity === 'HIGH';

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0b1019] border border-slate-700 w-full max-w-4xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className={`px-6 py-4 border-b flex items-center justify-between ${
          isCritical ? 'bg-red-950/40 border-red-900/50' : 'bg-slate-900/80 border-slate-800'
        }`}>
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${isCritical ? 'bg-red-600 text-white' : 'bg-amber-500 text-black'}`}>
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-mono font-bold text-lg text-white">{incident.id}</span>
                <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                  isCritical ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                }`}>
                  {incident.severity}
                </span>
                <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
                  incident.status === 'RESOLVED'
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : 'bg-cyan-500/20 text-cyan-400'
                }`}>
                  STATUS: {incident.status}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                {new Date(incident.timestamp).toLocaleString()} • {incident.location}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Top Grid: Evidence Preview & Quick Info */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Visual Evidence Viewport */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden relative aspect-video flex items-center justify-center">
              <div className="absolute inset-0 bg-gradient-to-br from-slate-900 to-black opacity-80" />
              
              {/* Surveillance Simulated Snapshot Frame */}
              <div className="relative text-center p-4">
                <div className="w-16 h-16 rounded-full bg-red-500/20 border border-red-500/50 flex items-center justify-center mx-auto mb-2 text-red-400 animate-pulse">
                  <AlertOctagon className="w-8 h-8" />
                </div>
                <div className="text-sm font-mono font-bold text-red-400 uppercase">
                  {incident.activity} DETECTED
                </div>
                <div className="text-xs font-mono text-slate-400 mt-1">
                  Confidence Score: <span className="text-emerald-400 font-bold">{incident.confidence}%</span>
                </div>
                <div className="text-[11px] text-slate-500 font-mono mt-0.5">
                  Track ID: Person #{incident.person_track_id || '01'}
                </div>
              </div>

              {/* Top watermark */}
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[10px] font-mono text-cyan-400 border border-cyan-500/30">
                EVIDENCE FEED: {incident.camera_id}
              </div>
            </div>

            {/* Incident Summary Card */}
            <div className="space-y-3">
              <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl space-y-2.5">
                <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
                  Incident Parameters
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div>
                    <span className="text-slate-500 block">CAMERA ID</span>
                    <span className="text-cyan-400 font-bold">{incident.camera_id}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">LOCATION</span>
                    <span className="text-slate-200">{incident.location}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">ACTIVITY</span>
                    <span className="text-red-400 font-bold">{incident.activity}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">SEVERITY</span>
                    <span className="text-amber-400 font-bold">{incident.severity}</span>
                  </div>
                </div>
              </div>

              {/* AI Reasoning Rationale */}
              <div className="bg-cyan-950/20 border border-cyan-500/30 p-4 rounded-xl space-y-1.5">
                <div className="flex items-center space-x-1.5 text-xs font-mono font-bold text-cyan-400">
                  <Cpu className="w-4 h-4" />
                  <span>AI INFERENCE & TRIGGER REASONING</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed font-mono">
                  {incident.ai_reasoning || "Neural spatio-temporal engine confirmed target activity signature with high statistical confidence across 45 consecutive video frames."}
                </p>
              </div>
            </div>
          </div>

          {/* Incident Progression Timeline */}
          <div className="bg-slate-900/40 border border-slate-800 p-4 rounded-xl space-y-3">
            <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span>Event Progression & Timeline</span>
            </div>
            
            <div className="flex items-center space-x-2 overflow-x-auto py-1">
              {(incident.events || [
                { activity: "Sitting", confidence: 89.2, time_offset_sec: 0, is_suspicious: false },
                { activity: "Standing", confidence: 92.4, time_offset_sec: 27, is_suspicious: false },
                { activity: incident.activity, confidence: incident.confidence, time_offset_sec: 52, is_suspicious: true },
                { activity: "Running", confidence: 92.8, time_offset_sec: 68, is_suspicious: true }
              ]).map((ev, idx) => (
                <div
                  key={idx}
                  className={`flex-shrink-0 px-3 py-2 rounded-lg border text-xs font-mono ${
                    ev.is_suspicious
                      ? 'bg-red-500/10 border-red-500/50 text-red-300'
                      : 'bg-slate-800/80 border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="text-[10px] text-slate-500">
                    +{Math.floor(ev.time_offset_sec)}s
                  </div>
                  <div className="font-bold">{ev.activity}</div>
                  <div className="text-[10px] text-emerald-400">{ev.confidence.toFixed(1)}% conf</div>
                </div>
              ))}
            </div>
          </div>

          {/* Operator Resolution Notes */}
          <div className="bg-slate-900/40 border border-slate-800 p-4 rounded-xl space-y-3">
            <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center space-x-1.5">
              <UserCheck className="w-4 h-4 text-emerald-400" />
              <span>SOC Operator Resolution & Action Taken</span>
            </div>

            {incident.status === 'RESOLVED' ? (
              <div className="bg-emerald-950/20 border border-emerald-500/30 p-3 rounded-lg text-xs font-mono text-emerald-300">
                <span className="font-bold">RESOLVED BY:</span> {incident.resolved_by || 'Officer Ramirez (Badge #8821)'}
                <div className="mt-1 text-slate-300">{incident.resolution_notes || 'Action verified and closed.'}</div>
              </div>
            ) : (
              <div className="space-y-2">
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Enter response notes, guard dispatch details, or false alarm dismissal rationale..."
                  className="w-full h-20 bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
                />
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={handleResolve}
                    className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg shadow-emerald-900/30"
                  >
                    <CheckCircle className="w-4 h-4" />
                    <span>MARK INCIDENT AS RESOLVED</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 bg-slate-950 border-t border-slate-800 flex items-center justify-between text-xs font-mono">
          <button
            onClick={() => alert(`Exporting Docket SOC-EVID-${incident.id}.pdf`)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-colors"
          >
            <FileDown className="w-4 h-4 text-cyan-400" />
            <span>EXPORT EVIDENCE DOSSIER</span>
          </button>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
          >
            CLOSE
          </button>
        </div>
      </div>
    </div>
  );
};
