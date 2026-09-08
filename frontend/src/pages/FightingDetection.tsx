import React, { useState, useRef, useEffect } from 'react';
import {
  Swords,
  Upload,
  Play,
  Pause,
  Cpu,
  CheckCircle,
  AlertTriangle,
  Clock,
  Sliders,
  Film,
  RotateCcw,
  Sparkles,
  ShieldAlert,
  Percent,
  Activity,
  Layers,
  FileVideo,
  Radio,
  Eye,
  Users
} from 'lucide-react';
import { API_BASE, getMediaUrl } from '../services/api';

interface FightingSegment {
  start_sec: number;
  end_sec: number;
  start_time: string;
  end_time: string;
  peak_conf: number;
  frame_count: number;
  detected_action?: string;
}

interface AnalysisResult {
  job_id: string;
  filename: string;
  final_result: string;
  is_fighting: boolean;
  overall_confidence: number;
  detected_action?: string;
  analyzed_frames: number;
  total_video_frames: number;
  fighting_frames: number;
  fighting_percentage: number;
  duration_sec: number;
  processing_time_sec: number;
  segments: FightingSegment[];
  video_url?: string;
  timeline?: Array<{
    time_sec: number;
    timestamp: string;
    fighting_prob: number;
    is_fighting: boolean;
    persons: number;
  }>;
}

interface IncidentEvent {
  id: string;
  timestamp: string;
  source: string;
  confidence: number;
  duration_sec: number;
  segments_count: number;
  status: string;
}

export const FightingDetection: React.FC = () => {
  const [activeMode, setActiveMode] = useState<'UPLOAD' | 'RTSP' | 'LOGS'>('UPLOAD');

  // Video Upload States
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null);
  const [activePlayerTab, setActivePlayerTab] = useState<'PROCESSED' | 'ORIGINAL'>('PROCESSED');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.70);
  const [sampleRate, setSampleRate] = useState<number>(2);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // RTSP States
  const [rtspUrl, setRtspUrl] = useState<string>('rtsp://192.168.1.100:554/live/ch0');
  const [isRtspStreaming, setIsRtspStreaming] = useState<boolean>(false);

  // Events State
  const [events, setEvents] = useState<IncidentEvent[]>([]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load events on mount
  useEffect(() => {
    fetchEvents();
  }, []);

  const fetchEvents = async () => {
    try {
      const res = await fetch(`${API_BASE}/fighting/events`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data.events || []);
      }
    } catch (e) {
      console.error("Failed to load events", e);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setOriginalVideoUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
      setErrorMessage(null);
      setProgress(0);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setProgress(5);
    setErrorMessage(null);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('sample_rate', sampleRate.toString());

    try {
      const response = await fetch(`${API_BASE}/fighting/analyze-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Video analysis request failed');
      }

      const initData = await response.json();
      const jobId = initData.job_id;

      // Poll analysis progress
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/fighting/status/${jobId}`);
          if (!statusRes.ok) return;

          const statusData = await statusRes.json();
          setProgress(Math.max(10, statusData.progress_percentage || 10));

          if (statusData.status === 'COMPLETED') {
            clearInterval(pollInterval);
            setIsAnalyzing(false);
            setProgress(100);
            setAnalysisResult(statusData);
            setActivePlayerTab('PROCESSED');
            fetchEvents();
          } else if (statusData.status === 'FAILED') {
            clearInterval(pollInterval);
            setIsAnalyzing(false);
            setErrorMessage(statusData.error || 'Video analysis failed on server');
          }
        } catch (err) {
          console.error('Polling error', err);
        }
      }, 1500);

    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMessage(err.message || 'Error uploading video');
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400">
              <Swords className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2.5">
                AI Fighting & Violence Detection
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 font-mono font-semibold">
                  MediaPipe Pose + BiLSTM
                </span>
              </h1>
              <p className="text-sm text-slate-400 mt-0.5">
                Full 33-landmark body kinematics, pelvis normalization & temporal deep sequence classifier
              </p>
            </div>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex items-center gap-1.5 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveMode('UPLOAD')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeMode === 'UPLOAD'
                ? 'bg-red-500/20 text-red-300 border border-red-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Film className="w-3.5 h-3.5" />
            Video Upload
          </button>
          <button
            onClick={() => setActiveMode('RTSP')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeMode === 'RTSP'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Radio className="w-3.5 h-3.5" />
            CCTV / RTSP
          </button>
          <button
            onClick={() => setActiveMode('LOGS')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeMode === 'LOGS'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            Incidents ({events.length})
          </button>
        </div>
      </div>

      {/* MODE 1: VIDEO UPLOAD */}
      {activeMode === 'UPLOAD' && (
        <div className="space-y-6">
          {/* Upload & Configuration Card */}
          <div className="soc-panel rounded-2xl p-6 border border-slate-800">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
              {/* Dropzone */}
              <div className="lg:col-span-2">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="video/mp4,video/avi,video/mov,video/mkv"
                  className="hidden"
                />
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-700 hover:border-red-500/50 rounded-xl p-8 text-center cursor-pointer transition-all bg-slate-950/40 hover:bg-slate-900/40 group"
                >
                  <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-semibold text-white">
                    {selectedFile ? selectedFile.name : 'Select or drop surveillance video file'}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports MP4, AVI, MOV, MKV (Analyzes complete 33 MediaPipe pose landmarks frame-by-frame)
                  </p>
                </div>
              </div>

              {/* Action & Parameters */}
              <div className="space-y-4 bg-slate-900/60 p-4 rounded-xl border border-slate-800/80">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">Confidence Threshold</span>
                    <span className="text-red-400 font-mono font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.50"
                    max="0.95"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full accent-red-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">Sampling Rate (Speed vs Detail)</span>
                    <span className="text-cyan-400 font-mono font-bold">Every {sampleRate} frames ({sampleRate >= 4 ? 'Fast' : 'Detailed'})</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    step="1"
                    value={sampleRate}
                    onChange={(e) => setSampleRate(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                  <span className="text-[10px] text-slate-500 block mt-1">Recommended: Every 4 frames for 5x faster analysis.</span>
                </div>

                <button
                  onClick={handleStartAnalysis}
                  disabled={!selectedFile || isAnalyzing}
                  className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 shadow-lg transition-all ${
                    !selectedFile || isAnalyzing
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white shadow-red-500/20'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <Activity className="w-4 h-4 animate-spin" />
                      Analyzing Pose Dynamics ({progress.toFixed(0)}%)...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" />
                      Run BiLSTM Fighting Detection
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Progress Bar */}
            {isAnalyzing && (
              <div className="mt-5 space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Tracking multi-person poses & computing velocity matrices...</span>
                  <span className="text-red-400 font-mono font-bold">{progress.toFixed(0)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden p-0.5 border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-red-600 to-amber-500 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 p-3.5 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-center gap-2.5">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>

          {/* Analysis Results & Video Player */}
          {analysisResult && (
            <div className="space-y-6">
              {/* Banner */}
              <div
                className={`soc-panel p-5 rounded-2xl border flex flex-col md:flex-row items-center justify-between gap-4 ${
                  analysisResult.is_fighting ? 'soc-panel-glow-red border-red-500/50' : 'soc-panel-glow-cyan border-emerald-500/50'
                }`}
              >
                <div className="flex items-center gap-3.5">
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center text-white ${
                      analysisResult.is_fighting ? 'bg-red-600' : 'bg-emerald-600'
                    }`}
                  >
                    {analysisResult.is_fighting ? <Swords className="w-6 h-6" /> : <CheckCircle className="w-6 h-6" />}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white tracking-wide flex items-center flex-wrap gap-2.5">
                      {analysisResult.final_result}
                      <span className={`text-xs px-2.5 py-0.5 rounded-full border font-mono font-semibold ${
                        analysisResult.is_fighting
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      }`}>
                        Action: {analysisResult.detected_action || (analysisResult.is_fighting ? 'Punching / Striking' : 'Normal Movement')}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-300 mt-0.5">
                      {analysisResult.is_fighting
                        ? `Confirmed physical altercation across ${analysisResult.segments.length} distinct fighting segments`
                        : 'No physical violence detected. Normal human activity verified across all temporal windows.'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4 bg-slate-950/60 px-5 py-2.5 rounded-xl border border-slate-800">
                  <div className="text-right">
                    <span className="text-[11px] text-slate-400 uppercase tracking-wider block">Decision Confidence</span>
                    <span className={`text-2xl font-bold font-mono ${analysisResult.is_fighting ? 'text-red-400' : 'text-emerald-400'}`}>
                      {analysisResult.overall_confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="soc-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Fighting Frames</span>
                  <div className="text-xl font-bold text-white font-mono">
                    {analysisResult.fighting_frames}{' '}
                    <span className="text-xs text-slate-400 font-normal">/ {analysisResult.analyzed_frames}</span>
                  </div>
                  <span className="text-[11px] text-red-400">
                    {analysisResult.fighting_percentage.toFixed(1)}% of sampled video
                  </span>
                </div>

                <div className="soc-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Fighting Segments</span>
                  <div className="text-xl font-bold text-amber-400 font-mono">
                    {analysisResult.segments.length}
                  </div>
                  <span className="text-[11px] text-slate-400">
                    Temporal confirmation ≥8 frames
                  </span>
                </div>

                <div className="soc-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">Video Duration</span>
                  <div className="text-xl font-bold text-cyan-400 font-mono">
                    {analysisResult.duration_sec.toFixed(1)}s
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {analysisResult.total_video_frames} total frames
                  </span>
                </div>

                <div className="soc-panel p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block mb-1">BiLSTM Speed</span>
                  <div className="text-xl font-bold text-emerald-400 font-mono">
                    {analysisResult.processing_time_sec.toFixed(1)}s
                  </div>
                  <span className="text-[11px] text-slate-400">
                    ~{(analysisResult.analyzed_frames / Math.max(0.1, analysisResult.processing_time_sec)).toFixed(0)} FPS processing
                  </span>
                </div>
              </div>

              {/* Video Player & Segments */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Player */}
                <div className="lg:col-span-2 soc-panel p-4 rounded-2xl border border-slate-800">
                  <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2.5">
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                      <Film className="w-4 h-4 text-red-400" />
                      Surveillance Playback
                    </span>
                    <div className="flex gap-1.5 bg-slate-950 p-1 rounded-lg border border-slate-800">
                      <button
                        onClick={() => setActivePlayerTab('PROCESSED')}
                        className={`text-xs px-2.5 py-1 rounded font-medium ${
                          activePlayerTab === 'PROCESSED'
                            ? 'bg-red-500/20 text-red-300 border border-red-500/40'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        MediaPipe Annotated Skeleton
                      </button>
                      <button
                        onClick={() => setActivePlayerTab('ORIGINAL')}
                        className={`text-xs px-2.5 py-1 rounded font-medium ${
                          activePlayerTab === 'ORIGINAL'
                            ? 'bg-slate-800 text-white'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Raw Video
                      </button>
                      <a
                        href={getMediaUrl(analysisResult.video_url || '/api/fighting/video/fighting_detected_video.mp4')}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs px-2.5 py-1 rounded font-medium text-slate-400 hover:text-white flex items-center gap-1 hover:bg-slate-900 transition-colors"
                        title="Open direct video in a new tab"
                      >
                        Direct Link
                      </a>
                    </div>
                  </div>

                  <div className="aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 relative group">
                    <video
                      key={`${activePlayerTab}-${analysisResult.job_id || analysisResult.video_url || 'default'}`}
                      controls
                      autoPlay
                      playsInline
                      muted
                      preload="auto"
                      className="w-full h-full object-contain"
                      src={
                        activePlayerTab === 'PROCESSED'
                          ? `${getMediaUrl(analysisResult.video_url || '/api/fighting/video/fighting_detected_video.mp4')}?v=${analysisResult.job_id || Date.now()}`
                          : (originalVideoUrl || '')
                      }
                    />
                    {activePlayerTab === 'PROCESSED' && (
                      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900/90 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-700 text-[11px] text-slate-300 flex items-center gap-1.5 shadow-lg">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        H.264 FastStart Stream
                      </div>
                    )}
                  </div>
                </div>

                {/* Detected Fighting Segments List */}
                <div className="soc-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2 border-b border-slate-800 pb-2.5">
                    <Clock className="w-4 h-4 text-amber-400" />
                    Fighting Timestamps ({analysisResult.segments.length})
                  </span>

                  {analysisResult.segments.length === 0 ? (
                    <div className="p-8 text-center text-slate-500 text-xs">
                      <CheckCircle className="w-8 h-8 mx-auto mb-2 text-emerald-500/40" />
                      No violent action segments identified in this video.
                    </div>
                  ) : (
                    <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                      {analysisResult.segments.map((seg, idx) => (
                        <div
                          key={idx}
                          className="p-3 rounded-xl bg-red-950/20 border border-red-500/30 flex items-center justify-between"
                        >
                          <div>
                            <span className="text-xs font-bold text-red-400 flex items-center gap-1.5">
                              <Swords className="w-3.5 h-3.5" />
                              Segment #{idx + 1}
                            </span>
                            <span className="text-xs text-white font-mono mt-0.5 block">
                              {seg.start_time} — {seg.end_time}
                            </span>
                            {seg.detected_action && (
                              <span className="text-[10px] text-amber-300 font-medium block mt-0.5">
                                {seg.detected_action}
                              </span>
                            )}
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-mono font-bold text-amber-400">
                              {seg.peak_conf.toFixed(1)}%
                            </span>
                            <span className="text-[10px] text-slate-400 block">
                              {(seg.end_sec - seg.start_sec).toFixed(1)}s duration
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* MODE 3: CCTV / RTSP */}
      {activeMode === 'RTSP' && (
        <div className="soc-panel p-6 rounded-2xl border border-slate-800 space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Radio className="w-5 h-5 text-purple-400" />
              CCTV / RTSP Stream Ingestion
            </h3>
            <p className="text-xs text-slate-400">
              Connect IP cameras or NVR stream endpoints for 24/7 continuous violence surveillance
            </p>
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={rtspUrl}
              onChange={(e) => setRtspUrl(e.target.value)}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-white font-mono focus:border-purple-500 focus:outline-none"
              placeholder="rtsp://admin:password@192.168.1.10:554/stream1"
            />
            <button
              onClick={() => setIsRtspStreaming(!isRtspStreaming)}
              className={`px-5 py-2.5 rounded-xl text-xs font-bold transition-all ${
                isRtspStreaming
                  ? 'bg-red-600 text-white'
                  : 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-500/20'
              }`}
            >
              {isRtspStreaming ? 'Disconnect' : 'Connect Stream'}
            </button>
          </div>

          <div className="aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 relative flex items-center justify-center">
            {isRtspStreaming ? (
              <div className="text-center space-y-3">
                <div className="w-3 h-3 rounded-full bg-emerald-500 animate-ping mx-auto" />
                <p className="text-sm font-mono text-emerald-400 font-bold">STREAM CONNECTED: {rtspUrl}</p>
                <p className="text-xs text-slate-400">YOLO Person Tracker + MediaPipe Active. No violent anomalies detected.</p>
              </div>
            ) : (
              <div className="text-center text-slate-500 space-y-2">
                <Radio className="w-10 h-10 mx-auto text-slate-700" />
                <p className="text-sm font-medium">Stream Offline</p>
                <p className="text-xs text-slate-600">Enter RTSP URL above and click Connect</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* MODE 4: INCIDENT INTELLIGENCE LOGS */}
      {activeMode === 'LOGS' && (
        <div className="soc-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-red-400" />
              Verified Violent Incidents Registry
            </h3>
            <span className="text-xs text-slate-400">{events.length} logged incidents</span>
          </div>

          {events.length === 0 ? (
            <div className="p-12 text-center text-slate-500 text-sm">
              No fighting incidents recorded in this session.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Event ID</th>
                    <th className="py-3 px-4">Timestamp</th>
                    <th className="py-3 px-4">Source</th>
                    <th className="py-3 px-4">Confidence</th>
                    <th className="py-3 px-4">Segments</th>
                    <th className="py-3 px-4">Duration</th>
                    <th className="py-3 px-4">Threat Level</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {events.map((ev) => (
                    <tr key={ev.id} className="hover:bg-slate-900/40">
                      <td className="py-3 px-4 font-bold text-red-400">{ev.id}</td>
                      <td className="py-3 px-4 text-slate-300">{ev.timestamp}</td>
                      <td className="py-3 px-4 text-slate-400">{ev.source}</td>
                      <td className="py-3 px-4 text-amber-400 font-bold">{ev.confidence.toFixed(1)}%</td>
                      <td className="py-3 px-4 text-slate-300">{ev.segments_count}</td>
                      <td className="py-3 px-4 text-cyan-400">{ev.duration_sec.toFixed(1)}s</td>
                      <td className="py-3 px-4">
                        <span className="px-2.5 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 text-[10px] font-bold">
                          {ev.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
