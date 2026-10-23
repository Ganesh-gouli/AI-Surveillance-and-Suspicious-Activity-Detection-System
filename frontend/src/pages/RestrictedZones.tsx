import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layers, ShieldCheck, Plus, AlertTriangle, Camera as CamIcon } from 'lucide-react';
import { ZonePolygonEditor } from '../components/zones/ZonePolygonEditor';
import { useToast } from '../context/ToastContext';
import { api } from '../services/api';
import { Camera, RestrictedZone } from '../types';

export const RestrictedZones: React.FC = () => {
  const navigate = useNavigate();
  const { success, warning } = useToast();
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamId, setSelectedCamId] = useState<string>('');
  const [zones, setZones] = useState<RestrictedZone[]>([]);

  const loadData = async () => {
    try {
      const cams = await api.getCameras();
      const zoneList = await api.getZones();
      setCameras(cams);
      setZones(zoneList);
      if (cams.length > 0 && !selectedCamId) {
        setSelectedCamId(cams[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSaveZone = async (zoneData: Partial<RestrictedZone>) => {
    try {
      await api.createZone(zoneData);
      loadData();
    } catch (e) {
      warning('Zone Save Failed', 'Could not save restricted zone to database.');
    }
  };

  const handleDeleteZone = async (zoneId: string) => {
    try {
      await api.deleteZone(zoneId);
      success('Zone Deleted', `Restricted perimeter ${zoneId} removed.`);
      loadData();
    } catch (e) {
      warning('Delete Failed', 'Could not remove zone.');
    }
  };

  const cameraZones = zones.filter((z) => z.camera_id === selectedCamId);

  return (
    <div className="p-4 sm:p-6 space-y-6 cyber-grid-bg min-h-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 sm:p-5 rounded-2xl shadow-xl">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold text-white font-mono tracking-tight">
              RESTRICTED ZONES & PERIMETER DEFENSE
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Vector Polygon Drawing • Real-Time Ray Casting Intrusion Alerts
            </p>
          </div>
        </div>

        {/* Camera Selector */}
        {cameras.length > 0 && (
          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="text-slate-400">SELECT FEED:</span>
            <select
              value={selectedCamId}
              onChange={(e) => setSelectedCamId(e.target.value)}
              className="bg-slate-900 border border-slate-700/80 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
            >
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ─ {c.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Main Interactive Polygon Editor */}
      {cameras.length === 0 ? (
        <div className="soc-panel p-12 rounded-2xl border border-slate-800 text-center space-y-3 shadow-xl">
          <div className="p-4 rounded-full bg-slate-900 border border-slate-800 text-slate-500 inline-block">
            <CamIcon className="w-10 h-10 text-cyan-400" />
          </div>
          <h3 className="text-sm font-mono font-bold text-white">NO CAMERAS FOR ZONE CONFIGURATION</h3>
          <p className="text-xs font-mono text-slate-400 max-w-sm mx-auto">
            Restricted zones must be mapped to an existing surveillance camera stream. Add a camera in Fleet Manager first.
          </p>
          <button
            onClick={() => navigate('/cameras')}
            className="px-4 py-2 rounded-xl btn-tactical-primary text-xs font-mono font-bold inline-flex items-center space-x-1.5 mt-2"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>PROVISION CAMERA</span>
          </button>
        </div>
      ) : (
        <ZonePolygonEditor
          cameraId={selectedCamId}
          existingZones={cameraZones}
          onSaveZone={handleSaveZone}
          onDeleteZone={handleDeleteZone}
        />
      )}
    </div>
  );
};
