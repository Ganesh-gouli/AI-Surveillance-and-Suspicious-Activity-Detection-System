import React, { useState, useEffect } from 'react';
import { Layers, ShieldCheck, Plus, AlertTriangle } from 'lucide-react';
import { ZonePolygonEditor } from '../components/zones/ZonePolygonEditor';
import { api } from '../services/api';
import { Camera, RestrictedZone } from '../types';

export const RestrictedZones: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamId, setSelectedCamId] = useState<string>('CAM-005');
  const [zones, setZones] = useState<RestrictedZone[]>([]);

  const loadData = async () => {
    const cams = await api.getCameras();
    const zoneList = await api.getZones();
    setCameras(cams);
    setZones(zoneList);
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSaveZone = async (zoneData: Partial<RestrictedZone>) => {
    await api.createZone(zoneData);
    loadData();
  };

  const handleDeleteZone = async (zoneId: string) => {
    await api.deleteZone(zoneId);
    loadData();
  };

  const cameraZones = zones.filter(z => z.camera_id === selectedCamId);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              RESTRICTED ZONES & PERIMETER DEFENSE
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Vector Polygon Drawing • Real-Time Ray Casting Intrusion Alerts
            </p>
          </div>
        </div>

        {/* Camera Selector */}
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="text-slate-400">SELECT FEED:</span>
          <select
            value={selectedCamId}
            onChange={(e) => setSelectedCamId(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-white focus:outline-none focus:border-cyan-400"
          >
            {cameras.map(c => (
              <option key={c.id} value={c.id}>{c.id} ─ {c.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Interactive Polygon Editor */}
      <ZonePolygonEditor
        cameraId={selectedCamId}
        existingZones={cameraZones}
        onSaveZone={handleSaveZone}
        onDeleteZone={handleDeleteZone}
      />
    </div>
  );
};
