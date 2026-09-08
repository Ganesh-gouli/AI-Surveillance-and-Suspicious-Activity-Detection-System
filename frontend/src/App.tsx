import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { ThreatTicker } from './components/layout/ThreatTicker';
import { IncidentDetailModal } from './components/incidents/IncidentDetailModal';

import { api } from './services/api';
import { audioService } from './services/audioAlerts';
import { Incident, SystemMetrics } from './types';

// Code-split page components for instant initial page loading & route caching
const Dashboard = lazy(() => import('./pages/Dashboard').then(m => ({ default: m.Dashboard })));
const LiveMonitoring = lazy(() => import('./pages/LiveMonitoring').then(m => ({ default: m.LiveMonitoring })));
const LiveWebcam = lazy(() => import('./pages/LiveWebcam').then(m => ({ default: m.LiveWebcam })));
const CameraManagement = lazy(() => import('./pages/CameraManagement').then(m => ({ default: m.CameraManagement })));
const VideoAnalysis = lazy(() => import('./pages/VideoAnalysis').then(m => ({ default: m.VideoAnalysis })));
const FightingDetection = lazy(() => import('./pages/FightingDetection').then(m => ({ default: m.FightingDetection })));
const ShootingDetection = lazy(() => import('./pages/ShootingDetection').then(m => ({ default: m.ShootingDetection })));
const ImageAnalysis = lazy(() => import('./pages/ImageAnalysis').then(m => ({ default: m.ImageAnalysis })));
const Incidents = lazy(() => import('./pages/Incidents').then(m => ({ default: m.Incidents })));
const RestrictedZones = lazy(() => import('./pages/RestrictedZones').then(m => ({ default: m.RestrictedZones })));
const HeatmapAnalytics = lazy(() => import('./pages/HeatmapAnalytics').then(m => ({ default: m.HeatmapAnalytics })));
const AIModels = lazy(() => import('./pages/AIModels').then(m => ({ default: m.AIModels })));
const Settings = lazy(() => import('./pages/Settings').then(m => ({ default: m.Settings })));

// Sleek high-tech cybernetic loading indicator during page transitions
const CyberLoadingSpinner: React.FC = () => (
  <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-3">
    <div className="relative w-10 h-10">
      <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 border-t-cyan-400 animate-spin" />
      <div className="absolute inset-1.5 rounded-full border-2 border-purple-500/20 border-b-purple-400 animate-spin [animation-direction:reverse]" />
    </div>
    <span className="text-[11px] font-mono text-cyan-400 tracking-widest uppercase animate-pulse">
      Loading AI Module...
    </span>
  </div>
);

export const App: React.FC = () => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);

  const refreshGlobalState = async () => {
    try {
      const metrics = await api.getSystemStatus();
      const incList = await api.getIncidents();
      setSystemMetrics(metrics);
      setIncidents(incList);
      setIsDemoMode(metrics.model_mode === 'DEMO_SIMULATION');
    } catch (e) {}
  };

  useEffect(() => {
    refreshGlobalState();
    const interval = setInterval(refreshGlobalState, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleDemoMode = async () => {
    const nextMode = isDemoMode ? 'REAL_AI' : 'DEMO_SIMULATION';
    await api.toggleSystemMode(nextMode);
    setIsDemoMode(!isDemoMode);
    audioService.playWarningBeep();
  };

  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-[#06090e] text-slate-100 font-sans">
        {/* Top SOC Navbar */}
        <Navbar
          systemMetrics={systemMetrics}
          onToggleDemoMode={handleToggleDemoMode}
          isDemoMode={isDemoMode}
        />

        {/* Global Live Threat Ticker Bar */}
        <ThreatTicker
          incidents={incidents}
          onSelectIncident={(inc) => setSelectedIncident(inc)}
        />

        {/* Main Content Layout with Sidebar */}
        <div className="flex flex-1 overflow-hidden">
          <Sidebar activeThreatCount={incidents.filter(i => i.status === 'OPEN').length} />

          {/* Page Routing Viewport with Lazy Loading */}
          <main className="flex-1 overflow-y-auto bg-[#070b12]">
            <Suspense fallback={<CyberLoadingSpinner />}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/monitoring" element={<LiveMonitoring />} />
                <Route path="/webcam" element={<LiveWebcam />} />
                <Route path="/cameras" element={<CameraManagement />} />
                <Route path="/fighting-detection" element={<FightingDetection />} />
                <Route path="/shooting-detection" element={<ShootingDetection />} />
                <Route path="/video-analysis" element={<VideoAnalysis />} />
                <Route path="/image-analysis" element={<ImageAnalysis />} />
                <Route path="/incidents" element={<Incidents />} />
                <Route path="/zones" element={<RestrictedZones />} />
                <Route path="/analytics" element={<HeatmapAnalytics />} />
                <Route path="/models" element={<AIModels />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Suspense>
          </main>
        </div>

        {/* Global Incident Modal */}
        <IncidentDetailModal
          incident={selectedIncident}
          onClose={() => setSelectedIncident(null)}
          onResolve={async (id, notes) => {
            await api.updateIncident(id, { status: 'RESOLVED', resolution_notes: notes });
            refreshGlobalState();
            setSelectedIncident(null);
          }}
        />
      </div>
    </Router>
  );
};

export default App;
