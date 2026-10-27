import React, { useState, useRef } from 'react';
import {
  Flame,
  Upload,
  Cpu,
  CheckCircle,
  AlertTriangle,
  Clock,
  Sliders,
  RotateCcw,
  Sparkles,
  Percent,
  FileVideo,
  Image as ImageIcon,
  Check,
  X,
  Eye,
  Info,
  ShieldCheck,
  Maximize2
} from 'lucide-react';
import { API_BASE, getMediaUrl } from '../services/api';

interface FireBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
  label: string;
  confidence: number;
  is_fire: boolean;
}

interface ImageFireResult {
  status: 'FIRE DETECTED' | 'NO FIRE DETECTED' | 'Uncertain — Please Verify' | string;
  status_code: 'FIRE' | 'NO_FIRE' | 'UNCERTAIN';
  is_fire: boolean;
  confidence: number;
  confidence_threshold: number;
  boxes: FireBox[];
  boxes_count: number;
  description: string;
  image_width: number;
  image_height: number;
  mean_luminance: number;
  firenet_fire_confidence: number;
  firenet_no_fire_confidence: number;
  yolo_detected: boolean;
  processing_time_ms: number;
  annotated_image: string;
  filename: string;
}

interface VideoAnalysisResult {
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
  // Operational Mode: IMAGE is primary and default
  const [activeMode, setActiveMode] = useState<'IMAGE' | 'VIDEO'>('IMAGE');

  // ===================== IMAGE FIRE DETECTION STATE =====================
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [isImageAnalyzing, setIsImageAnalyzing] = useState<boolean>(false);
  const [imageResult, setImageResult] = useState<ImageFireResult | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [imageConfidenceThreshold, setImageConfidenceThreshold] = useState<number>(0.60);
  const [showImageConfig, setShowImageConfig] = useState<boolean>(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const imageFileInputRef = useRef<HTMLInputElement | null>(null);

  // ===================== VIDEO FIRE FORENSICS STATE =====================
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [originalVideoUrl, setOriginalVideoUrl] = useState<string | null>(null);
  const [activeVideoTab, setActiveVideoTab] = useState<'PROCESSED' | 'ORIGINAL'>('PROCESSED');
  const [videoConfidenceThreshold, setVideoConfidenceThreshold] = useState<number>(0.70);
  const [frameRatio, setFrameRatio] = useState<number>(0.20);
  const [sampleRate, setSampleRate] = useState<number>(5);
  const [isVideoAnalyzing, setIsVideoAnalyzing] = useState<boolean>(false);
  const [videoProgress, setVideoProgress] = useState<number>(0);
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const videoFileInputRef = useRef<HTMLInputElement | null>(null);

  // ===================== IMAGE HANDLERS =====================
  const handleImageFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelected(e.dataTransfer.files[0]);
    }
  };

  const handleImageSelected = (file: File) => {
    setImageError(null);
    setImageResult(null);

    const allowedExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const isImageMime = file.type.startsWith('image/');

    if (!isImageMime && !allowedExts.includes(ext)) {
      setImageError(`Unsupported format '${ext || file.type}'. Please upload a JPG, JPEG, or PNG image.`);
      return;
    }

    setSelectedImageFile(file);
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setImagePreviewUrl(URL.createObjectURL(file));
  };

  const clearImage = () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setSelectedImageFile(null);
    setImagePreviewUrl(null);
    setImageResult(null);
    setImageError(null);
    if (imageFileInputRef.current) {
      imageFileInputRef.current.value = '';
    }
  };

  const loadSampleImage = async (filename: string, displayName: string) => {
    setImageError(null);
    setImageResult(null);
    try {
      const url = getMediaUrl(`/results/${filename}`);
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Could not fetch ${displayName} sample image.`);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
      handleImageSelected(file);
    } catch (err: any) {
      setImageError(err.message || 'Failed to load sample image.');
    }
  };

  const analyzeImage = async () => {
    if (!selectedImageFile) return;

    setIsImageAnalyzing(true);
    setImageError(null);
    setImageResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImageFile);
      formData.append('confidence_threshold', String(imageConfidenceThreshold));

      const res = await fetch(`${API_BASE}/fire/analyze-image`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(err.detail || 'Image analysis failed');
      }

      const data: ImageFireResult = await res.json();
      setImageResult(data);
    } catch (err: any) {
      setImageError(err.message || 'Failed to analyze image. Please ensure backend is running.');
    } finally {
      setIsImageAnalyzing(false);
    }
  };

  // ===================== VIDEO HANDLERS =====================
  const handleVideoFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleVideoSelected(e.dataTransfer.files[0]);
    }
  };

  const handleVideoSelected = (file: File) => {
    setSelectedVideoFile(file);
    if (originalVideoUrl) {
      URL.revokeObjectURL(originalVideoUrl);
    }
    setOriginalVideoUrl(URL.createObjectURL(file));
    setIsVideoAnalyzing(false);
    setVideoProgress(0);
    setVideoResult(null);
    setVideoError(null);
    setActiveVideoTab('PROCESSED');
  };

  const startVideoAnalysis = async () => {
    if (!selectedVideoFile) return;

    setIsVideoAnalyzing(true);
    setVideoProgress(0);
    setVideoResult(null);
    setVideoError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedVideoFile);
      formData.append('confidence_threshold', String(videoConfidenceThreshold));
      formData.append('frame_ratio', String(frameRatio));
      formData.append('sample_rate', String(sampleRate));

      const res = await fetch(`${API_BASE}/fire/analyze-video`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Video upload failed' }));
        throw new Error(err.detail || 'Video upload failed');
      }

      const data = await res.json();
      const jobId = data.job_id;

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/fire/status/${jobId}`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setVideoProgress(statusData.progress_percentage || 0);

            if (statusData.status === 'COMPLETED') {
              clearInterval(pollInterval);
              setIsVideoAnalyzing(false);
              setVideoResult(statusData);
              setActiveVideoTab('PROCESSED');
            } else if (statusData.status === 'FAILED') {
              clearInterval(pollInterval);
              setIsVideoAnalyzing(false);
              setVideoError(statusData.error || 'Video analysis failed.');
            }
          }
        } catch {
          clearInterval(pollInterval);
          setIsVideoAnalyzing(false);
          setVideoError('Network connection lost during analysis.');
        }
      }, 800);
    } catch (err: any) {
      setIsVideoAnalyzing(false);
      setVideoError(err.message || 'Failed to initialize video analysis.');
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* ===================== TOP HEADER & MODE CONTROLS ===================== */}
      <div className="bg-[#0a0f19] border border-slate-800 p-5 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center space-x-3.5">
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.2)]">
            <Flame className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white font-mono tracking-tight flex items-center space-x-2">
              <span>FIRE DETECTION SYSTEM</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/40">
                AI Inference
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              PyTorch MobileNetV3 • <span className="text-orange-400 font-bold">YOLOv8 Flame Core</span> • Static Image Analysis
            </p>
          </div>
        </div>

        {/* Operational Mode Toggle Tabs */}
        <div className="flex items-center space-x-2 bg-slate-950/80 p-1.5 rounded-xl border border-slate-800/80">
          <button
            onClick={() => setActiveMode('IMAGE')}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold flex items-center space-x-2 transition-all ${
              activeMode === 'IMAGE'
                ? 'bg-gradient-to-r from-red-600 to-orange-600 text-white shadow-[0_0_15px_rgba(239,68,68,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ImageIcon className="w-3.5 h-3.5" />
            <span>IMAGE ANALYSIS</span>
          </button>

          <button
            onClick={() => setActiveMode('VIDEO')}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold flex items-center space-x-2 transition-all ${
              activeMode === 'VIDEO'
                ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-[0_0_15px_rgba(6,182,212,0.3)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileVideo className="w-3.5 h-3.5" />
            <span>VIDEO FORENSICS</span>
          </button>

          <button
            onClick={() => setShowImageConfig(!showImageConfig)}
            className={`p-2 rounded-lg text-xs font-mono transition-all border ${
              showImageConfig
                ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title="Configure Thresholds"
          >
            <Sliders className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ===================== CONFIGURABLE THRESHOLD DRAWER ===================== */}
      {showImageConfig && (
        <div className="bg-slate-950/90 border border-cyan-500/30 p-5 rounded-2xl shadow-[0_0_25px_rgba(0,240,255,0.06)] animate-in fade-in duration-200">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                  <Flame className="w-3.5 h-3.5 text-red-400" />
                  <span>IMAGE FIRE CONFIDENCE THRESHOLD</span>
                </span>
                <span className="text-red-400 font-bold">{(imageConfidenceThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.30"
                max="0.95"
                step="0.05"
                value={imageConfidenceThreshold}
                onChange={(e) => setImageConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full accent-red-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />
              <p className="text-[10px] font-mono text-slate-500">
                Minimum confidence score required for an image to trigger FIRE DETECTED status.
                Scores between 35% and this threshold yield 'Uncertain — Please Verify'.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                  <span>ANTI-FALSE-POSITIVE SUPPRESSION</span>
                </span>
                <span className="text-emerald-400 font-bold">ACTIVE</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
                <p>• Avoids classifying red/orange objects as fire without flame contours</p>
                <p>• Low-light / dark scene suppression active</p>
                <p>• MobileNetV3 + YOLOv8 dual cross-validation enabled</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===================== MODE 1: IMAGE FIRE DETECTION (PRIMARY) ===================== */}
      {activeMode === 'IMAGE' && (
        <div className="space-y-6">
          {/* Quick Test Samples Preset Bar */}
          <div className="bg-[#0a0f19] border border-slate-800/80 p-3.5 rounded-xl flex items-center justify-between flex-wrap gap-2 text-xs font-mono">
            <span className="text-slate-400 flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>TEST BENCH PRESETS:</span>
            </span>
            <div className="flex items-center space-x-2 flex-wrap gap-y-2">
              <button
                onClick={() => loadSampleImage('sample_fireplace.png', 'User Fireplace')}
                className="px-3 py-1.5 rounded-lg bg-red-950/40 hover:bg-red-900/50 border border-red-500/40 text-red-300 transition-all flex items-center space-x-1.5"
              >
                <span>🔥 User Fireplace (Fire)</span>
              </button>
              <button
                onClick={() => loadSampleImage('sample_normal_room.jpg', 'Normal Room')}
                className="px-3 py-1.5 rounded-lg bg-emerald-950/40 hover:bg-emerald-900/50 border border-emerald-500/40 text-emerald-300 transition-all flex items-center space-x-1.5"
              >
                <span>🏢 Normal Room (No Fire)</span>
              </button>
              <button
                onClick={() => loadSampleImage('sample_red_object.jpg', 'Red Objects')}
                className="px-3 py-1.5 rounded-lg bg-orange-950/40 hover:bg-orange-900/50 border border-orange-500/40 text-orange-300 transition-all flex items-center space-x-1.5"
              >
                <span>🔴 Red/Orange Objects</span>
              </button>
              <button
                onClick={() => loadSampleImage('sample_dark_scene.jpg', 'Dark Scene')}
                className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-all flex items-center space-x-1.5"
              >
                <span>🌑 Dark Scene</span>
              </button>
            </div>
          </div>

          {/* Hidden File Input */}
          <input
            ref={imageFileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleImageSelected(e.target.files[0]);
              }
            }}
          />

          {/* Error Message Display */}
          {imageError && (
            <div className="bg-red-950/60 border border-red-500/50 p-4 rounded-xl flex items-center space-x-3 text-red-200 font-mono text-xs animate-in fade-in">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <div className="flex-1">
                <span className="font-bold">Error: </span>
                <span>{imageError}</span>
              </div>
              <button
                onClick={() => setImageError(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {/* Upload Dropzone (When no image is chosen) */}
          {!imagePreviewUrl && (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleImageFileDrop}
              onClick={() => imageFileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-red-500/60 bg-[#070b12] hover:bg-[#0a0f19] rounded-2xl p-14 text-center cursor-pointer transition-all duration-300 group shadow-lg"
            >
              <div className="w-20 h-20 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform shadow-[0_0_30px_rgba(239,68,68,0.15)]">
                <Upload className="w-10 h-10 text-red-400 group-hover:animate-bounce" />
              </div>
              <h3 className="text-base font-bold text-white font-mono mb-1">
                UPLOAD AN IMAGE FOR FIRE DETECTION
              </h3>
              <p className="text-xs text-slate-400 font-mono max-w-md mx-auto mb-4">
                Drag and drop your image here, or browse from your computer.
                Accurately distinguishes real fire from red/orange objects and low-light scenes.
              </p>
              <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-400">
                <span>Supported Formats:</span>
                <span className="text-red-400 font-bold">JPG</span>
                <span>•</span>
                <span className="text-orange-400 font-bold">JPEG</span>
                <span>•</span>
                <span className="text-amber-400 font-bold">PNG</span>
              </div>
            </div>
          )}

          {/* Image Workspace (When an image is loaded) */}
          {imagePreviewUrl && (
            <div className="space-y-6">
              {/* Image Control Header Bar */}
              <div className="bg-[#0a0f19] border border-slate-800 p-4 rounded-xl flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
                    <ImageIcon className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <span className="text-xs font-mono font-bold text-white block">
                      {selectedImageFile?.name || 'Uploaded Image'}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {selectedImageFile ? `${(selectedImageFile.size / 1024).toFixed(1)} KB` : ''} • Image-Based Inspection
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-2.5">
                  {/* Bounding box toggle if analyzed */}
                  {imageResult && imageResult.boxes_count > 0 && (
                    <button
                      onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all flex items-center space-x-1.5 ${
                        showBoundingBoxes
                          ? 'bg-red-950/40 border-red-500/50 text-red-300'
                          : 'bg-slate-900 border-slate-800 text-slate-400'
                      }`}
                    >
                      <Eye className="w-3.5 h-3.5 text-red-400" />
                      <span>{showBoundingBoxes ? 'BOUNDING BOXES ON' : 'BOXES OFF'}</span>
                    </button>
                  )}

                  {/* Primary CTA: Analyze Image */}
                  <button
                    onClick={analyzeImage}
                    disabled={isImageAnalyzing}
                    className="px-5 py-2 rounded-xl bg-gradient-to-r from-red-600 via-orange-600 to-amber-600 hover:from-red-500 hover:to-orange-500 disabled:opacity-50 text-white text-xs font-mono font-bold flex items-center space-x-2 shadow-lg shadow-red-900/40 transition-all active:scale-95"
                  >
                    <Flame className="w-4 h-4" />
                    <span>{isImageAnalyzing ? 'ANALYZING IMAGE...' : 'ANALYZE IMAGE'}</span>
                  </button>

                  {/* Upload Another Image */}
                  <button
                    onClick={() => imageFileInputRef.current?.click()}
                    disabled={isImageAnalyzing}
                    className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono flex items-center space-x-1.5 transition-all"
                  >
                    <Upload className="w-3.5 h-3.5 text-cyan-400" />
                    <span>UPLOAD ANOTHER</span>
                  </button>

                  {/* Clear Image */}
                  <button
                    onClick={clearImage}
                    disabled={isImageAnalyzing}
                    className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-red-400 text-xs font-mono flex items-center space-x-1.5 transition-all"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>CLEAR IMAGE</span>
                  </button>
                </div>
              </div>

              {/* Grid: Left Image Preview, Right Result Panel */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Image Display Canvas Viewport */}
                <div className="lg:col-span-7 bg-[#070b12] border border-slate-800 rounded-2xl overflow-hidden relative shadow-2xl flex flex-col justify-center items-center min-h-[420px] max-h-[560px]">
                  {/* Image View */}
                  <img
                    src={
                      imageResult && showBoundingBoxes && imageResult.annotated_image
                        ? imageResult.annotated_image
                        : imagePreviewUrl
                    }
                    alt="Inspection Preview"
                    className="max-h-[540px] w-auto max-w-full object-contain select-none"
                  />

                  {/* Neural Processing Loading Indicator Overlay */}
                  {isImageAnalyzing && (
                    <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center space-y-4 p-6 z-20">
                      <div className="relative w-16 h-16">
                        <div className="absolute inset-0 rounded-full border-4 border-red-500/20 border-t-red-500 animate-spin" />
                        <div className="absolute inset-2 rounded-full border-4 border-orange-500/20 border-b-orange-400 animate-spin [animation-direction:reverse]" />
                        <Flame className="w-6 h-6 text-red-400 absolute inset-0 m-auto animate-pulse" />
                      </div>
                      <div className="text-center font-mono space-y-1">
                        <p className="text-sm font-bold text-white tracking-widest uppercase">
                          EVALUATING FIRE DETECTION MODEL...
                        </p>
                        <p className="text-[11px] text-slate-400">
                          Resizing 224x224 • Normalizing • Running MobileNetV3 & YOLOv8 Flame Core
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Right: Results Panel */}
                <div className="lg:col-span-5 space-y-4">
                  {imageResult ? (
                    <div
                      className={`p-6 rounded-2xl border transition-all duration-300 shadow-2xl space-y-5 ${
                        imageResult.status_code === 'FIRE'
                          ? 'bg-red-950/40 border-red-500/60 shadow-[0_0_35px_rgba(239,68,68,0.25)]'
                          : imageResult.status_code === 'UNCERTAIN'
                          ? 'bg-amber-950/30 border-amber-500/50 shadow-[0_0_25px_rgba(245,158,11,0.2)]'
                          : 'bg-emerald-950/30 border-emerald-500/40 shadow-[0_0_25px_rgba(16,185,129,0.15)]'
                      }`}
                    >
                      {/* Status Header Badge */}
                      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                        <div className="flex items-center space-x-2.5">
                          {imageResult.status_code === 'FIRE' ? (
                            <div className="p-2 rounded-xl bg-red-500/20 border border-red-500/50 text-red-400 animate-pulse">
                              <Flame className="w-6 h-6" />
                            </div>
                          ) : imageResult.status_code === 'UNCERTAIN' ? (
                            <div className="p-2 rounded-xl bg-amber-500/20 border border-amber-500/50 text-amber-400">
                              <AlertTriangle className="w-6 h-6" />
                            </div>
                          ) : (
                            <div className="p-2 rounded-xl bg-emerald-500/20 border border-emerald-500/50 text-emerald-400">
                              <CheckCircle className="w-6 h-6" />
                            </div>
                          )}
                          <div>
                            <div
                              className={`text-lg font-mono font-black tracking-wide ${
                                imageResult.status_code === 'FIRE'
                                  ? 'text-red-400'
                                  : imageResult.status_code === 'UNCERTAIN'
                                  ? 'text-amber-400'
                                  : 'text-emerald-400'
                              }`}
                            >
                              {imageResult.status}
                            </div>
                            <span className="text-[10px] font-mono text-slate-400">
                              Inference Latency: {imageResult.processing_time_ms} ms
                            </span>
                          </div>
                        </div>

                        {/* Confidence Score Pill */}
                        <div className="text-right">
                          <div
                            className={`text-2xl font-mono font-black ${
                              imageResult.status_code === 'FIRE'
                                ? 'text-red-400'
                                : imageResult.status_code === 'UNCERTAIN'
                                ? 'text-amber-400'
                                : 'text-emerald-400'
                            }`}
                          >
                            {imageResult.confidence}%
                          </div>
                          <span className="text-[10px] font-mono text-slate-500">
                            Confidence Score
                          </span>
                        </div>
                      </div>

                      {/* Description Summary */}
                      <p className="text-xs font-mono text-slate-300 leading-relaxed bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                        {imageResult.description}
                      </p>

                      {/* Model & Telemetry Metrics Cards */}
                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">MobileNetV3 FireNet</span>
                          <span className="font-bold text-white text-sm">
                            {imageResult.firenet_fire_confidence}% Fire
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-0.5">
                            Clean: {imageResult.firenet_no_fire_confidence}%
                          </span>
                        </div>

                        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">YOLOv8 Flame Boxes</span>
                          <span className="font-bold text-white text-sm">
                            {imageResult.boxes_count} Zone{imageResult.boxes_count !== 1 ? 's' : ''}
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-0.5">
                            Spatial Verification
                          </span>
                        </div>

                        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">Decision Threshold</span>
                          <span className="font-bold text-cyan-400 text-sm">
                            {imageResult.confidence_threshold}%
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-0.5">
                            Configurable Slider
                          </span>
                        </div>

                        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block">Image Dimensions</span>
                          <span className="font-bold text-purple-400 text-sm">
                            {imageResult.image_width} × {imageResult.image_height}
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-0.5">
                            Native Resolution
                          </span>
                        </div>
                      </div>

                      {/* Bounding Box Detail List (If fire zones detected) */}
                      {imageResult.boxes_count > 0 && (
                        <div className="space-y-2">
                          <span className="text-[11px] font-mono font-bold text-slate-300 block">
                            DETECTED FLAME REGIONS:
                          </span>
                          <div className="space-y-1.5 max-h-36 overflow-y-auto">
                            {imageResult.boxes.map((box, idx) => (
                              <div
                                key={idx}
                                className="p-2 rounded-lg bg-slate-900/80 border border-red-500/30 flex items-center justify-between text-[11px] font-mono"
                              >
                                <span className="text-red-400 font-bold flex items-center space-x-1.5">
                                  <Flame className="w-3.5 h-3.5" />
                                  <span>Zone #{idx + 1}: {box.label}</span>
                                </span>
                                <span className="text-slate-400">
                                  [{box.x1}, {box.y1}, {box.width}×{box.height}] • <strong className="text-white">{box.confidence}%</strong>
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Reset / New Analysis CTA */}
                      <button
                        onClick={clearImage}
                        className="w-full py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-mono font-bold flex items-center justify-center space-x-2 transition-all"
                      >
                        <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
                        <span>CLEAR & ANALYZE ANOTHER IMAGE</span>
                      </button>
                    </div>
                  ) : (
                    /* Initial Guidance Panel before analysis */
                    <div className="bg-[#0a0f19] border border-slate-800 p-6 rounded-2xl space-y-4 text-center py-12">
                      <div className="w-14 h-14 rounded-2xl bg-red-500/10 border border-red-500/30 flex items-center justify-center mx-auto text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.15)]">
                        <Flame className="w-7 h-7" />
                      </div>
                      <div className="space-y-1 font-mono">
                        <h4 className="text-sm font-bold text-white">READY FOR FIRE DETECTION</h4>
                        <p className="text-xs text-slate-400 max-w-xs mx-auto">
                          Click "ANALYZE IMAGE" to run the PyTorch MobileNetV3 and YOLOv8 fire models.
                        </p>
                      </div>
                      <button
                        onClick={analyzeImage}
                        disabled={isImageAnalyzing}
                        className="w-full py-3 rounded-xl bg-gradient-to-r from-red-600 via-orange-600 to-amber-600 hover:from-red-500 hover:to-orange-500 text-white text-xs font-mono font-bold flex items-center justify-center space-x-2 shadow-lg shadow-red-900/40 transition-all active:scale-95"
                      >
                        <Flame className="w-4 h-4" />
                        <span>ANALYZE IMAGE NOW</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===================== MODE 2: VIDEO FIRE FORENSICS (SECONDARY) ===================== */}
      {activeMode === 'VIDEO' && (
        <div className="space-y-6">
          <input
            ref={videoFileInputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.webm,video/mp4,video/avi"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleVideoSelected(e.target.files[0]);
              }
            }}
          />

          {!selectedVideoFile && (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleVideoFileDrop}
              onClick={() => videoFileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 bg-[#070b12] hover:bg-[#0a0f19] rounded-2xl p-14 text-center cursor-pointer transition-all duration-300 group shadow-lg"
            >
              <div className="w-20 h-20 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform shadow-[0_0_30px_rgba(6,182,212,0.15)]">
                <FileVideo className="w-10 h-10 text-cyan-400 group-hover:animate-bounce" />
              </div>
              <h3 className="text-base font-bold text-white font-mono mb-1">
                UPLOAD VIDEO FOR FORENSIC SCAN
              </h3>
              <p className="text-xs text-slate-400 font-mono max-w-md mx-auto mb-4">
                Drag and drop your video file here (MP4, AVI, MOV, MKV, WEBM).
                Runs frame-by-frame temporal aggregation and saves annotated video.
              </p>
            </div>
          )}

          {selectedVideoFile && (
            <div className="space-y-6">
              <div className="bg-[#0a0f19] border border-slate-800 p-4 rounded-xl flex items-center justify-between flex-wrap gap-3">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
                    <FileVideo className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <span className="text-xs font-mono font-bold text-white block">
                      {selectedVideoFile.name}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {(selectedVideoFile.size / (1024 * 1024)).toFixed(2)} MB • Forensic Video
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  {!videoResult && (
                    <button
                      onClick={startVideoAnalysis}
                      disabled={isVideoAnalyzing}
                      className="px-5 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 text-white text-xs font-mono font-bold flex items-center space-x-2 shadow-lg"
                    >
                      <Flame className="w-4 h-4" />
                      <span>{isVideoAnalyzing ? `SCANNING (${videoProgress.toFixed(0)}%)...` : 'START VIDEO ANALYSIS'}</span>
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setSelectedVideoFile(null);
                      setVideoResult(null);
                      setOriginalVideoUrl(null);
                    }}
                    className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Video Player */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-8 bg-[#070b12] border border-slate-800 rounded-2xl overflow-hidden p-2">
                  <video
                    controls
                    className="w-full rounded-xl bg-black"
                    src={
                      activeVideoTab === 'PROCESSED' && videoResult?.processed_video_url
                        ? getMediaUrl(videoResult.processed_video_url)
                        : originalVideoUrl || undefined
                    }
                  />
                </div>

                <div className="lg:col-span-4 bg-[#0a0f19] border border-slate-800 p-5 rounded-2xl space-y-4">
                  <h4 className="text-xs font-mono font-bold text-slate-400 tracking-wider uppercase">
                    VIDEO SCAN METRICS
                  </h4>
                  {videoResult ? (
                    <div className="space-y-4 font-mono text-xs">
                      <div className={`p-4 rounded-xl border ${videoResult.fire_detected ? 'bg-red-950/40 border-red-500/60' : 'bg-emerald-950/40 border-emerald-500/60'}`}>
                        <span className="text-sm font-bold block">{videoResult.final_result}</span>
                        <span className="text-[10px] text-slate-400">Confidence: {videoResult.overall_confidence}%</span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-slate-400">
                          <span>Fire Frames:</span>
                          <span className="text-white font-bold">{videoResult.fire_frames} / {videoResult.analyzed_frames}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Fire Ratio:</span>
                          <span className="text-orange-400 font-bold">{videoResult.fire_percentage}%</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Duration:</span>
                          <span className="text-cyan-400 font-bold">{videoResult.duration_sec}s</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs font-mono text-slate-500">
                      Click 'START VIDEO ANALYSIS' to process video.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoAnalysis;
