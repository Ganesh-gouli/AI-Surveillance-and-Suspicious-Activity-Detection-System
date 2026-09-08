import React, { useState, useEffect } from 'react';
import {
  Camera as CameraIcon,
  Plus,
  Radio,
  Search,
  CheckCircle,
  AlertTriangle,
  Play,
  Square,
  Trash2,
  Lock,
  Wifi,
  Sliders
} from 'lucide-react';
import { api } from '../services/api';
import { Camera, CameraStatus } from '../types';

export const CameraManagement: React.FC = () => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [isTesting, setIsTesting] = useState(false);

  // New Camera Form State
  const [newCamera, setNewCamera] = useState({
    name: '',
    stream_url: 'rtsp://edge-vms.internal/stream/',
    location: '',
    camera_type: 'Fixed IP',
    sensitivity: 0.80
  });

  const loadCameras = async () => {
    const data = await api.getCameras();
    setCameras(data);
  };

  useEffect(() => {
    loadCameras();
  }, []);

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await api.testConnection(newCamera.stream_url);
      setTestResult(res);
    } catch (e) {
      setTestResult({ success: false, message: "Handshake timed out." });
    }
    setIsTesting(false);
  };

  const handleSaveCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCamera.name || !newCamera.stream_url || !newCamera.location) {
      alert("Please fill in all required camera parameters.");
      return;
    }

    try {
      await api.createCamera(newCamera);
      setShowAddModal(false);
      setNewCamera({
        name: '',
        stream_url: 'rtsp://edge-vms.internal/stream/',
        location: '',
        camera_type: 'Fixed IP',
        sensitivity: 0.80
      });
      setTestResult(null);
      loadCameras();
    } catch (e) {
      alert("Error registering new CCTV stream.");
    }
  };

  const filteredCameras = cameras.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      {/* Top Controls Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <CameraIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              CCTV & IP CAMERA FLEET MANAGER
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              Enterprise RTSP/HTTP Stream Provisioning • Sensor Health & Sensitivity Tuning
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search cameras by ID, name, location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400 w-64"
            />
          </div>

          <button
            onClick={() => setShowAddModal(true)}
            className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold flex items-center space-x-1.5 shadow-lg shadow-cyan-900/30 transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>ADD IP CAMERA</span>
          </button>
        </div>
      </div>

      {/* Cameras Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCameras.map((cam) => {
          const isOnline = cam.status === 'ONLINE';
          const isOffline = cam.status === 'OFFLINE';
          const isConnecting = cam.status === 'CONNECTING';

          return (
            <div
              key={cam.id}
              className="soc-panel p-4 rounded-xl border border-slate-800 hover:border-slate-700 transition-all space-y-3 relative group"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono font-bold text-white">{cam.id}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    isOnline
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : isOffline
                      ? 'bg-slate-800 text-slate-400 border border-slate-700'
                      : isConnecting
                      ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      : 'bg-red-500/20 text-red-400 border border-red-500/30'
                  }`}>
                    ● {cam.status}
                  </span>
                </div>

                <span className="text-xs text-slate-400 font-mono">{cam.camera_type}</span>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-100">{cam.name}</h3>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{cam.location}</p>
              </div>

              {/* Stream URL (Masked Credentials) */}
              <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800/80 text-[11px] font-mono text-slate-400 flex items-center space-x-2 truncate">
                <Lock className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                <span className="truncate">{cam.stream_url}</span>
              </div>

              {/* Stream Telemetry */}
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 text-xs font-mono">
                <div>
                  <span className="text-slate-500 block text-[10px]">FPS</span>
                  <span className="text-cyan-400 font-bold">{cam.fps}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">RESOLUTION</span>
                  <span className="text-slate-300">{cam.resolution}</span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px]">AI CONF</span>
                  <span className="text-emerald-400 font-bold">{cam.ai_confidence}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Add Camera Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0b1019] border border-slate-700 w-full max-w-lg rounded-2xl shadow-2xl p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <CameraIcon className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-white font-mono">
                  Register New CCTV / IP Camera Stream
                </h3>
              </div>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSaveCamera} className="space-y-3.5">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">CAMERA NAME *</label>
                <input
                  type="text"
                  placeholder="e.g. Main Entrance Gate 1"
                  value={newCamera.name}
                  onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                  required
                />
              </div>

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">
                  STREAM URL (RTSP / HTTP) * <span className="text-slate-500">(Credentials Masked)</span>
                </label>
                <input
                  type="text"
                  placeholder="rtsp://admin:****@192.168.1.100:554/stream1"
                  value={newCamera.stream_url}
                  onChange={(e) => setNewCamera({ ...newCamera, stream_url: e.target.value })}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">LOCATION *</label>
                  <input
                    type="text"
                    placeholder="Building A - Floor 1"
                    value={newCamera.location}
                    onChange={(e) => setNewCamera({ ...newCamera, location: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                    required
                  />
                </div>

                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">CAMERA TYPE</label>
                  <select
                    value={newCamera.camera_type}
                    onChange={(e) => setNewCamera({ ...newCamera, camera_type: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                  >
                    <option value="Fixed IP">Fixed IP 4K</option>
                    <option value="PTZ Long-Range">PTZ Long-Range</option>
                    <option value="Panoramic 360">Panoramic 360</option>
                    <option value="Thermal Hybrid">Thermal Hybrid</option>
                  </select>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-300">DETECTION SENSITIVITY</span>
                  <span className="text-cyan-400 font-bold">{Math.round(newCamera.sensitivity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.99"
                  step="0.01"
                  value={newCamera.sensitivity}
                  onChange={(e) => setNewCamera({ ...newCamera, sensitivity: parseFloat(e.target.value) })}
                  className="w-full accent-cyan-400"
                />
              </div>

              {/* Handshake Tester Output */}
              {testResult && (
                <div className={`p-3 rounded-lg border text-xs font-mono ${
                  testResult.success ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300' : 'bg-red-950/30 border-red-500/40 text-red-300'
                }`}>
                  <div className="font-bold">{testResult.message}</div>
                  {testResult.latency_ms && (
                    <div className="text-[10px] text-slate-400 mt-1">
                      Latency: {testResult.latency_ms}ms • Stream Codec: {testResult.codec}
                    </div>
                  )}
                </div>
              )}

              <div className="flex items-center justify-between pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={isTesting}
                  className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono flex items-center space-x-1.5 border border-slate-700"
                >
                  <Wifi className="w-3.5 h-3.5 text-cyan-400" />
                  <span>{isTesting ? 'CONNECTING...' : 'TEST CONNECTION'}</span>
                </button>

                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={() => setShowAddModal(false)}
                    className="px-3.5 py-2 rounded-lg bg-slate-800 text-slate-300 text-xs font-mono"
                  >
                    CANCEL
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-mono font-bold shadow-lg shadow-cyan-900/30"
                  >
                    SAVE CAMERA
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
