import React, { useState, useRef } from 'react';
import { Plus, Trash2, Check, ShieldAlert, Layers } from 'lucide-react';
import { useToast } from '../../context/ToastContext';
import { Point, RestrictedZone, SeverityLevel } from '../../types';

interface ZonePolygonEditorProps {
  cameraId: string;
  existingZones: RestrictedZone[];
  onSaveZone: (zone: Partial<RestrictedZone>) => void;
  onDeleteZone: (zoneId: string) => void;
}

export const ZonePolygonEditor: React.FC<ZonePolygonEditorProps> = ({
  cameraId,
  existingZones,
  onSaveZone,
  onDeleteZone
}) => {
  const { warning, success } = useToast();
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [zoneName, setZoneName] = useState('');
  const [severity, setSeverity] = useState<SeverityLevel>('CRITICAL');
  const [schedule, setSchedule] = useState('24/7');

  const containerRef = useRef<HTMLDivElement | null>(null);

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    setCurrentPoints((prev) => [...prev, { x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 }]);
  };

  const handleSave = () => {
    if (currentPoints.length < 3) {
      warning('Vertex Minimum', 'A restricted zone polygon requires at least 3 vertices.');
      return;
    }
    if (!zoneName.trim()) {
      warning('Zone Name Required', 'Please provide a descriptive name for the restricted perimeter.');
      return;
    }

    onSaveZone({
      camera_id: cameraId,
      name: zoneName.trim(),
      polygon: currentPoints,
      severity,
      alert_type: 'UNAUTHORIZED_ENTRY',
      is_active: true,
      color: severity === 'CRITICAL' ? '#ef4444' : '#f59e0b',
      schedule
    });

    success('Zone Established', `Restricted perimeter "${zoneName}" configured for ${cameraId}.`);
    setIsDrawing(false);
    setCurrentPoints([]);
    setZoneName('');
  };

  const handleCancel = () => {
    setIsDrawing(false);
    setCurrentPoints([]);
    setZoneName('');
  };

  return (
    <div className="space-y-4">
      {/* Interactive Drawing Viewport */}
      <div
        ref={containerRef}
        onClick={handleCanvasClick}
        className={`relative aspect-video w-full bg-slate-950 rounded-2xl border overflow-hidden select-none shadow-2xl transition-all ${
          isDrawing ? 'cursor-crosshair border-cyan-500 shadow-[0_0_24px_rgba(0,240,255,0.25)]' : 'border-slate-800'
        }`}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-[#0a1120] to-[#0c1424] opacity-90" />

        {/* Existing Polygons */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {existingZones.map((z) => {
            const pts = z.polygon.map((p) => `${p.x}%,${p.y}%`).join(' ');
            return (
              <g key={z.id}>
                <polygon
                  points={pts}
                  fill={z.severity === 'CRITICAL' ? 'rgba(239,68,68,0.25)' : 'rgba(245,158,11,0.25)'}
                  stroke={z.color || '#ef4444'}
                  strokeWidth="2.5"
                  strokeDasharray="6,4"
                />
              </g>
            );
          })}

          {/* Currently Drawing Polygon */}
          {currentPoints.length > 0 && (
            <polygon
              points={currentPoints.map((p) => `${p.x}%,${p.y}%`).join(' ')}
              fill="rgba(0,240,255,0.25)"
              stroke="#00f0ff"
              strokeWidth="2.5"
            />
          )}
        </svg>

        {/* Vertex Markers */}
        {currentPoints.map((p, idx) => (
          <div
            key={idx}
            style={{ left: `${p.x}%`, top: `${p.y}%` }}
            className="absolute -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-cyan-400 border-2 border-white shadow-[0_0_12px_rgba(0,240,255,0.9)] z-30 flex items-center justify-center text-[8px] font-mono font-bold text-black"
          >
            {idx + 1}
          </div>
        ))}

        {/* Viewport Overlay Hints */}
        <div className="absolute top-3 left-3 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-xl text-xs font-mono text-cyan-300 backdrop-blur-md">
          TARGET CAMERA: <span className="font-bold text-white">{cameraId || 'Unselected'}</span>
          {isDrawing && (
            <span className="ml-3 text-amber-400 font-bold animate-pulse">
              [Click inside feed to place vertices • {currentPoints.length} placed]
            </span>
          )}
        </div>
      </div>

      {/* Editor Controls */}
      <div className="soc-panel p-5 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
        {!isDrawing ? (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h4 className="text-sm font-bold text-white font-mono">Configured Restricted Perimeters</h4>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                Define prohibited geometric polygons with automated intrusion alerts and ray-casting.
              </p>
            </div>
            <button
              onClick={() => setIsDrawing(true)}
              className="px-4 py-2 rounded-xl btn-tactical-primary text-xs font-mono font-bold flex items-center space-x-1.5 transition-all"
            >
              <Plus className="w-4 h-4" />
              <span>DRAW NEW RESTRICTED ZONE</span>
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-2">
              <Layers className="w-4 h-4" />
              <span>New Zone Configuration</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">ZONE NAME *</label>
                <input
                  type="text"
                  placeholder="e.g. Server Vault Perimeter"
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                />
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">BREACH SEVERITY</label>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value as SeverityLevel)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
                >
                  <option value="CRITICAL">CRITICAL (Red Siren)</option>
                  <option value="HIGH">HIGH (Amber Alert)</option>
                  <option value="MEDIUM">MEDIUM (Warning)</option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">SCHEDULE</label>
                <select
                  value={schedule}
                  onChange={(e) => setSchedule(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
                >
                  <option value="24/7">24/7 Continuous Defense</option>
                  <option value="After-Hours (20:00 - 06:00)">After-Hours (20:00 - 06:00)</option>
                  <option value="Weekend Lockout">Weekend Lockout</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end space-x-2.5 pt-2">
              <button
                onClick={handleCancel}
                className="px-4 py-2 rounded-xl bg-slate-800 text-slate-300 text-xs font-mono hover:bg-slate-700"
              >
                CANCEL
              </button>
              <button
                onClick={handleSave}
                disabled={currentPoints.length < 3}
                className="px-4 py-2 rounded-xl btn-tactical-primary text-xs font-mono font-bold flex items-center space-x-1.5 disabled:opacity-40"
              >
                <Check className="w-4 h-4" />
                <span>SAVE RESTRICTED ZONE</span>
              </button>
            </div>
          </div>
        )}

        {/* Existing Zones List */}
        {existingZones.length > 0 && (
          <div className="divide-y divide-slate-800/80 pt-2">
            {existingZones.map((z) => (
              <div key={z.id} className="py-2.5 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center space-x-2.5">
                  <span
                    className="w-3.5 h-3.5 rounded-full shrink-0 shadow-sm"
                    style={{ backgroundColor: z.color || '#ef4444' }}
                  />
                  <span className="font-bold text-white">{z.name}</span>
                  <span className="text-slate-500">[{z.schedule}]</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      z.severity === 'CRITICAL'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                        : 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                    }`}
                  >
                    {z.severity}
                  </span>
                  <button
                    onClick={() => onDeleteZone(z.id)}
                    className="text-slate-500 hover:text-red-400 p-1 transition-colors"
                    title="Delete Zone"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
