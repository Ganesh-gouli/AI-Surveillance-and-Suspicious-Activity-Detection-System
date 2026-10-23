import React, { useState, useRef } from 'react';
import { Plus, Trash2, Check, ShieldAlert, Layers } from 'lucide-react';
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

    setCurrentPoints(prev => [...prev, { x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 }]);
  };

  const handleSave = () => {
    if (currentPoints.length < 3) {
      alert("A polygon requires at least 3 vertices.");
      return;
    }
    if (!zoneName.trim()) {
      alert("Please name the restricted zone.");
      return;
    }

    onSaveZone({
      camera_id: cameraId,
      name: zoneName,
      polygon: currentPoints,
      severity,
      alert_type: "UNAUTHORIZED_ENTRY",
      is_active: true,
      color: severity === 'CRITICAL' ? '#ef4444' : '#f59e0b',
      schedule
    });

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
        className={`relative aspect-video w-full bg-slate-950 rounded-xl border overflow-hidden select-none ${
          isDrawing ? 'cursor-crosshair border-cyan-500 shadow-[0_0_20px_rgba(0,240,255,0.2)]' : 'border-slate-800'
        }`}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 to-[#0c1424] opacity-90" />

        {/* Existing Polygons */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {existingZones.map((z) => {
            const pts = z.polygon.map(p => `${p.x}%,${p.y}%`).join(' ');
            return (
              <g key={z.id}>
                <polygon
                  points={pts}
                  fill={z.severity === 'CRITICAL' ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}
                  stroke={z.color || '#ef4444'}
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              </g>
            );
          })}

          {/* Currently Drawing Polygon */}
          {currentPoints.length > 0 && (
            <polygon
              points={currentPoints.map(p => `${p.x}%,${p.y}%`).join(' ')}
              fill="rgba(0,240,255,0.2)"
              stroke="#00f0ff"
              strokeWidth="2"
            />
          )}
        </svg>

        {/* Vertex Markers */}
        {currentPoints.map((p, idx) => (
          <div
            key={idx}
            style={{ left: `${p.x}%`, top: `${p.y}%` }}
            className="absolute -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-cyan-400 border-2 border-white shadow-[0_0_8px_rgba(0,240,255,0.8)] z-30"
          />
        ))}

        {/* Viewport Overlay Hints */}
        <div className="absolute top-3 left-3 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-cyan-300">
          CAMERA FEED: <span className="font-bold text-white">{cameraId}</span>
          {isDrawing && (
            <span className="ml-3 text-amber-400 font-bold animate-pulse">
              [Click on feed to place vertices ({currentPoints.length} placed)]
            </span>
          )}
        </div>
      </div>

      {/* Editor Controls */}
      <div className="bg-[#0a0f18] border border-slate-800 p-4 rounded-xl space-y-4">
        {!isDrawing ? (
          <div className="flex justify-between items-center">
            <div>
              <h4 className="text-sm font-semibold text-white">Configured Restricted Zones</h4>
              <p className="text-xs text-slate-400">Define prohibited perimeters with automated intrusion alerts.</p>
            </div>
            <button
              onClick={() => setIsDrawing(true)}
              className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg shadow-cyan-900/30 transition-all"
            >
              <Plus className="w-4 h-4" />
              <span>DRAW NEW RESTRICTED ZONE</span>
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-xs font-mono font-bold text-cyan-400 uppercase">
              Zone Configuration
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">ZONE NAME</label>
                <input
                  type="text"
                  placeholder="e.g. Server Vault Perimeter"
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                />
              </div>

              <div>
                <label className="text-[11px] font-mono text-slate-400 block mb-1">BREACH SEVERITY</label>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value as SeverityLevel)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
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
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                >
                  <option value="24/7">24/7 Continuous Defense</option>
                  <option value="After-Hours (20:00 - 06:00)">After-Hours (20:00 - 06:00)</option>
                  <option value="Weekend Lockout">Weekend Lockout</option>
                </select>
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={handleCancel}
                className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-mono hover:bg-slate-700"
              >
                CANCEL
              </button>
              <button
                onClick={handleSave}
                disabled={currentPoints.length < 3}
                className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg disabled:opacity-50"
              >
                <Check className="w-4 h-4" />
                <span>SAVE RESTRICTED ZONE</span>
              </button>
            </div>
          </div>
        )}

        {/* Existing Zones List */}
        {existingZones.length > 0 && (
          <div className="divide-y divide-slate-800 pt-2">
            {existingZones.map((z) => (
              <div key={z.id} className="py-2 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: z.color || '#ef4444' }}
                  />
                  <span className="font-bold text-slate-200">{z.name}</span>
                  <span className="text-slate-500">[{z.schedule}]</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    z.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
                  }`}>
                    {z.severity}
                  </span>
                  <button
                    onClick={() => onDeleteZone(z.id)}
                    className="text-slate-500 hover:text-red-400 p-1"
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
