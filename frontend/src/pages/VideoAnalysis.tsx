import React, { useState, useRef, useEffect } from 'react';
import {
  Flame,
  Upload,
  Play,
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
  FileVideo
} from 'lucide-react';
import { API_BASE, getMediaUrl } from '../services/api';

interface AnalysisResult {
  job_id: string;
  filename: string;
  final_result: string;
  fire_detected: boolean;
  overall_confidence: number;
  fire_frames: number;
  analyzed_frames: number;
  total_video_frames: number;
  total_fire_boxes_detected?: number;
  fire_percentage: number;
  duration_sec: number;
  processing_time_sec: number;
  processed_video_url?: string;
  timeline?: Array<{
    frame_idx: number;
    time_offset_sec: number;
    fire_confidence: number;
    is_fire: boolean;
  }>;
}

export const VideoAnalysis: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'PROCESSED' | 'ORIGINAL'>('PROCESSED');
  
  // Configurable Temporal Decision Thresholds (Section 2 & 3)
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.70);
  const [frameRatio, setFrameRatio] = useState<number>(0.20);
  const [sampleRate, setSampleRate] = useState<number>(5);
  const [showConfig, setShowConfig] = useState<boolean>(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoPlayerRef = useRef<HTMLVideoElement | null>(null);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file: File) => {
    setSelectedFile(file);
    if (originalVideoUrl) {
      URL.revokeObjectURL(originalVideoUrl);
    }
    setOriginalVideoUrl(URL.createObjectURL(file));
    setIsAnalyzing(false);
    setProgress(0);
    setAnalysisResult(null);
    setErrorMessage(null);
    setActiveTab('PROCESSED');
  };

  const startAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setProgress(0);
    setAnalysisResult(null);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('confidence_threshold', String(confidenceThreshold));
      formData.append('frame_ratio', String(frameRatio));
      formData.append('sample_rate', String(sampleRate));

      const res = await fetch(`${API_BASE}/fire/analyze-video`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      const jobId = data.job_id;

      // Poll status every 800ms
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/fire/status/${jobId}`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setProgress(statusData.progress_percentage || 0);

            if (statusData.status === 'COMPLETED') {
              clearInterval(pollInterval);
              setIsAnalyzing(false);
              setAnalysisResult(statusData);
              setActiveTab('PROCESSED');
            } else if (statusData.status === 'FAILED') {
              clearInterval(pollInterval);
              setIsAnalyzing(false);
              setErrorMessage(statusData.error || 'Video analysis failed.');
            }
          }
        } catch (err) {
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          setErrorMessage('Network connection lost during analysis.');
        }
      }, 800);
    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMessage(err.message || 'Failed to initialize video analysis.');
    }
  };

  // Load sample video directly from the Fire folder
  const loadSampleVideo = async (filename: string, isFire: boolean) => {
    try {
      // Fetch the sample from dataset endpoint or test file
      const url = getMediaUrl(isFire ? `/results/sample_${filename}` : `/dataset/no_fire/${filename}`);
      // In case local file needs fetching, fallback to creating file from Fire folder or show alert
      const response = await fetch(url);
      if (response.ok) {
        const blob = await response.blob();
        const file = new File([blob], filename, { type: 'video/mp4' });
        handleFileSelected(file);
      } else {
        // Direct hint to user
        alert(`You can upload any MP4 from your Fire folder:\nC:\\Users\\lenovo\\OneDrive\\Desktop\\My new major project\\Fire\\${filename}`);
      }
    } catch (e) {
      alert(`Please select your fire video from:\nC:\\Users\\lenovo\\OneDrive\\Desktop\\My new major project\\Fire`);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Top Header */}
      <div className="bg-[#0a0f19] border border-slate-800 p-5 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center space-x-3.5">
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.2)]">
            <Flame className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white font-mono tracking-tight flex items-center space-x-2">
              <span>FIRE DETECTION AI & FORENSIC VIDEO ANALYZER</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/40">
                PyTorch MobileNetV3
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Frame-by-Frame OpenCV Extraction • <span className="text-orange-400 font-bold">Temporal Aggregation</span> • <span className="text-cyan-400 font-bold">H.264 Annotated Output</span>
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-2.5">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className={`px-3 py-2 rounded-xl text-xs font-mono font-medium flex items-center space-x-1.5 border transition-all ${
              showConfig
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-[0_0_15px_rgba(0,240,255,0.2)]'
                : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border-slate-800'
            }`}
          >
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span>THRESHOLDS</span>
          </button>

          {selectedFile && !isAnalyzing && !analysisResult && (
            <button
              onClick={startAnalysis}
              className="px-5 py-2 rounded-xl bg-gradient-to-r from-red-600 via-orange-600 to-amber-600 hover:from-red-500 hover:to-orange-500 text-white text-xs font-mono font-bold flex items-center space-x-2 shadow-lg shadow-red-900/40 transition-all active:scale-95"
            >
              <Flame className="w-4 h-4" />
              <span>ANALYZE VIDEO</span>
            </button>
          )}

          {analysisResult && (
            <button
              onClick={() => {
                setSelectedFile(null);
                setAnalysisResult(null);
                setOriginalVideoUrl(null);
                setProgress(0);
              }}
              className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono flex items-center space-x-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
              <span>NEW ANALYSIS</span>
            </button>
          )}
        </div>
      </div>

      {/* Configurable Threshold Settings Tray */}
      {showConfig && (
        <div className="bg-slate-950/90 border border-cyan-500/30 p-5 rounded-2xl grid grid-cols-1 md:grid-cols-3 gap-6 shadow-[0_0_25px_rgba(0,240,255,0.06)] animate-in fade-in duration-200">
          {/* FIRE_CONFIDENCE_THRESHOLD */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                <Flame className="w-3.5 h-3.5 text-red-400" />
                <span>FIRE_CONFIDENCE_THRESHOLD</span>
              </span>
              <span className="text-red-400 font-bold">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              disabled={isAnalyzing}
              className="w-full accent-red-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[10px] font-mono text-slate-500">
              Minimum AI probability for a single frame to be counted as containing fire.
            </p>
          </div>

          {/* FIRE_FRAME_RATIO */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                <Percent className="w-3.5 h-3.5 text-orange-400" />
                <span>FIRE_FRAME_RATIO</span>
              </span>
              <span className="text-orange-400 font-bold">{(frameRatio * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.50"
              step="0.05"
              value={frameRatio}
              onChange={(e) => setFrameRatio(parseFloat(e.target.value))}
              disabled={isAnalyzing}
              className="w-full accent-orange-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[10px] font-mono text-slate-500">
              Percentage of analyzed frames that must have fire to classify the entire video as FIRE.
            </p>
          </div>

          {/* FRAME_SAMPLE_RATE */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                <span>FRAME_SAMPLE_RATE</span>
              </span>
              <span className="text-cyan-400 font-bold">Every {sampleRate}th Frame</span>
            </div>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={sampleRate}
              onChange={(e) => setSampleRate(parseInt(e.target.value))}
              disabled={isAnalyzing}
              className="w-full accent-cyan-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[10px] font-mono text-slate-500">
              Frame extraction sampling interval. Lower = deeper scan; higher = faster execution.
            </p>
          </div>
        </div>
      )}

      {/* Error Notice */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/50 text-xs font-mono text-red-300 flex items-center space-x-2.5">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Drag and Drop Area if no video chosen */}
      {!selectedFile && (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-red-500/60 bg-slate-950/40 hover:bg-slate-900/50 rounded-2xl p-14 text-center cursor-pointer transition-all space-y-4 group"
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".mp4,.avi,.mov,.mkv,.webm"
            onChange={(e) => e.target.files && handleFileSelected(e.target.files[0])}
          />
          <div className="w-20 h-20 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto text-red-400 group-hover:scale-105 transition-all shadow-[0_0_25px_rgba(239,68,68,0.15)]">
            <Upload className="w-9 h-9 text-red-400" />
          </div>
          <div className="space-y-1.5">
            <h3 className="text-base font-bold text-white font-mono">
              Upload Video for Genuine PyTorch Fire Detection Analysis
            </h3>
            <p className="text-xs text-slate-400 font-mono max-w-lg mx-auto">
              Drag and drop your video file here, or click to browse. Supports <span className="text-slate-200">MP4, AVI, MOV, MKV, WEBM</span>.
            </p>
          </div>

          {/* Quick Shortcuts for your Fire folder */}
          <div className="pt-3 border-t border-slate-800/80 max-w-md mx-auto">
            <p className="text-[11px] font-mono text-slate-500 mb-2">
              Ready videos in your project:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-[10px] font-mono text-orange-400">
                Fire\Explosion005_x264A.mp4
              </span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-[10px] font-mono text-orange-400">
                Fire\Explosion023_x264A.mp4
              </span>
              <span className="px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-[10px] font-mono text-orange-400">
                Fire\Explosion041_x264A.mp4
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Analysis In-Progress State (Section 10) */}
      {isAnalyzing && (
        <div className="bg-[#090e18] border border-cyan-500/40 p-6 rounded-2xl space-y-4 shadow-[0_0_30px_rgba(0,240,255,0.1)]">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-cyan-400 font-bold flex items-center space-x-2">
              <Cpu className="w-4 h-4 animate-spin text-cyan-400" />
              <span>ANALYZING FRAMES WITH PYTORCH FIRE DETECTOR...</span>
            </span>
            <span className="text-white font-bold bg-cyan-950 px-2.5 py-1 rounded border border-cyan-500/40">
              {progress}%
            </span>
          </div>

          <div className="w-full bg-slate-800/80 h-3 rounded-full overflow-hidden p-0.5">
            <div
              className="bg-gradient-to-r from-cyan-500 via-orange-500 to-red-500 h-full rounded-full transition-all duration-300 shadow-[0_0_15px_rgba(239,68,68,0.5)]"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
            <span>Frame Sampling: Every {sampleRate}th Frame</span>
            <span>Threshold: {(confidenceThreshold * 100).toFixed(0)}% | Ratio: {(frameRatio * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Main Viewport & Forensic Results (Section 4 & 5) */}
      {selectedFile && !isAnalyzing && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left / Center 2 Cols: Video Viewport */}
          <div className="lg:col-span-2 space-y-4">
            {/* View Mode Switcher */}
            {analysisResult && (
              <div className="flex items-center justify-between bg-slate-900/60 p-2 rounded-xl border border-slate-800 text-xs font-mono">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setActiveTab('PROCESSED')}
                    className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                      activeTab === 'PROCESSED'
                        ? 'bg-red-600 text-white font-bold shadow-lg shadow-red-900/40'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Play className="w-3.5 h-3.5" />
                    <span>PLAY PROCESSED VIDEO (WITH AI HUD)</span>
                  </button>

                  <button
                    onClick={() => setActiveTab('ORIGINAL')}
                    className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all ${
                      activeTab === 'ORIGINAL'
                        ? 'bg-slate-700 text-white font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <FileVideo className="w-3.5 h-3.5" />
                    <span>ORIGINAL UPLOAD</span>
                  </button>
                </div>

                <span className="text-slate-500 text-[10px]">
                  {activeTab === 'PROCESSED' ? 'Output: results/detected_fire_video.mp4' : selectedFile.name}
                </span>
              </div>
            )}

            {/* Video Player */}
            <div className="relative aspect-video w-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl">
              {activeTab === 'PROCESSED' && analysisResult?.processed_video_url ? (
                <video
                  ref={videoPlayerRef}
                  key={analysisResult.processed_video_url}
                  src={getMediaUrl(analysisResult.processed_video_url)}
                  controls
                  autoPlay
                  className="w-full h-full object-contain"
                />
              ) : (
                originalVideoUrl && (
                  <video
                    ref={videoPlayerRef}
                    key={originalVideoUrl}
                    src={originalVideoUrl}
                    controls
                    className="w-full h-full object-contain"
                  />
                )
              )}
            </div>

            {/* Interactive Timeline Strip */}
            {analysisResult?.timeline && analysisResult.timeline.length > 0 && (
              <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl space-y-2">
                <div className="text-xs font-mono font-bold text-slate-300 flex items-center justify-between">
                  <span className="flex items-center space-x-1.5">
                    <Activity className="w-4 h-4 text-orange-400" />
                    <span>FRAME-LEVEL DETECTION TIMELINE (CLICK TO JUMP)</span>
                  </span>
                  <span className="text-[10px] text-slate-500">
                    {analysisResult.timeline.length} sample points
                  </span>
                </div>

                <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pt-1">
                  {analysisResult.timeline.map((point, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        if (videoPlayerRef.current) {
                          videoPlayerRef.current.currentTime = point.time_offset_sec;
                          videoPlayerRef.current.play();
                        }
                      }}
                      className={`px-2 py-1 rounded text-[10px] font-mono border transition-all ${
                        point.is_fire
                          ? 'bg-red-500/20 text-red-300 border-red-500/50 hover:bg-red-500/30'
                          : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800'
                      }`}
                    >
                      {point.time_offset_sec.toFixed(1)}s • {point.is_fire ? `🔥 ${point.fire_confidence}%` : `${point.fire_confidence}%`}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Forensic AI Results Dossier (Section 4) */}
          <div className="space-y-4">
            <div className="bg-[#090e18] border border-slate-800 p-5 rounded-2xl space-y-5 shadow-xl">
              {/* Dossier Title */}
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
                  <ShieldAlert className="w-4 h-4 text-red-400" />
                  <span>AI Forensic Verdict</span>
                </div>
                <span className={`text-[10px] font-mono px-2.5 py-0.5 rounded font-bold ${
                  analysisResult
                    ? analysisResult.fire_detected
                      ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                      : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                    : 'bg-slate-800 text-slate-400'
                }`}>
                  {analysisResult ? 'ANALYSIS COMPLETE' : 'READY TO ANALYZE'}
                </span>
              </div>

              {analysisResult ? (
                <div className="space-y-4">
                  {/* Final Result Big Banner (Section 4 requirement) */}
                  <div className={`p-4 rounded-xl border flex flex-col items-center justify-center text-center space-y-1 shadow-lg ${
                    analysisResult.fire_detected
                      ? 'bg-red-950/40 border-red-500/60 shadow-red-950/40'
                      : 'bg-emerald-950/40 border-emerald-500/60 shadow-emerald-950/40'
                  }`}>
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">
                      Final Classification
                    </span>
                    <div className={`text-xl font-mono font-extrabold ${
                      analysisResult.fire_detected ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {analysisResult.final_result}
                    </div>
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                    {/* Video Name */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90 col-span-2">
                      <span className="text-slate-500 block text-[10px]">VIDEO NAME</span>
                      <span className="text-slate-200 font-bold truncate block" title={analysisResult.filename}>
                        {analysisResult.filename}
                      </span>
                    </div>

                    {/* Overall Confidence */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">OVERALL CONFIDENCE</span>
                      <span className={`text-base font-bold ${
                        analysisResult.fire_detected ? 'text-red-400' : 'text-emerald-400'
                      }`}>
                        {analysisResult.overall_confidence.toFixed(1)}%
                      </span>
                    </div>

                    {/* Fire Percentage */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">FIRE PERCENTAGE</span>
                      <span className="text-base font-bold text-orange-400">
                        {analysisResult.fire_percentage.toFixed(1)}%
                      </span>
                    </div>

                    {/* Fire Frames */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">FIRE FRAMES</span>
                      <span className="text-base font-bold text-red-300">
                        {analysisResult.fire_frames}
                      </span>
                    </div>

                    {/* Analyzed Frames */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">ANALYZED FRAMES</span>
                      <span className="text-base font-bold text-cyan-300">
                        {analysisResult.analyzed_frames} <span className="text-slate-500 text-xs">/ {analysisResult.total_video_frames}</span>
                      </span>
                    </div>

                    {/* Localized Fire Regions */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">LOCALIZED FIRE BOXES</span>
                      <span className="text-base font-bold text-yellow-400">
                        {analysisResult.total_fire_boxes_detected ?? analysisResult.fire_frames} <span className="text-[10px] text-slate-500 font-normal">regions</span>
                      </span>
                    </div>

                    {/* Processing Time */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">PROCESSING TIME</span>
                      <span className="text-base font-bold text-slate-200">
                        {analysisResult.processing_time_sec}s
                      </span>
                    </div>

                    {/* Video Duration */}
                    <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/90">
                      <span className="text-slate-500 block text-[10px]">DURATION</span>
                      <span className="text-base font-bold text-slate-200">
                        {analysisResult.duration_sec}s
                      </span>
                    </div>
                  </div>

                  {/* Temporal Decision Rule Explanation Banner */}
                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
                    <div className="text-slate-300 font-bold flex items-center space-x-1">
                      <Layers className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Temporal Logic Verification:</span>
                    </div>
                    <p className="text-[10px] text-slate-400 leading-relaxed">
                      {analysisResult.fire_frames} frames ({(analysisResult.fire_percentage).toFixed(1)}%) met the ≥{(confidenceThreshold * 100).toFixed(0)}% confidence threshold.
                      {analysisResult.fire_detected
                        ? ` Exceeds the ${(frameRatio * 100).toFixed(0)}% ratio requirement.`
                        : ` Below the ${(frameRatio * 100).toFixed(0)}% ratio requirement.`}
                    </p>
                  </div>

                  {/* Output video location */}
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-[10px] font-mono text-slate-400 flex items-center justify-between">
                    <span>Saved Output Video:</span>
                    <span className="text-cyan-400 font-bold">results/detected_fire_video.mp4</span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-10 space-y-3">
                  <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
                    <FileVideo className="w-6 h-6" />
                  </div>
                  <p className="text-xs font-mono text-slate-400">
                    Video loaded: <span className="text-white font-bold">{selectedFile.name}</span>
                  </p>
                  <button
                    onClick={startAnalysis}
                    className="w-full py-2.5 rounded-xl bg-gradient-to-r from-red-600 via-orange-600 to-amber-600 hover:from-red-500 hover:to-orange-500 text-white text-xs font-mono font-bold flex items-center justify-center space-x-2 shadow-lg shadow-red-900/30 transition-all"
                  >
                    <Flame className="w-4 h-4" />
                    <span>START AI ANALYSIS NOW</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
export default VideoAnalysis;
