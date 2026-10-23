import React, { useState, useRef } from 'react';
import {
  Image as ImageIcon,
  Upload,
  Cpu,
  Copy,
  AlertTriangle,
  Layers,
  Sparkles,
  Box,
  Link as LinkIcon
} from 'lucide-react';
import { CameraCanvasOverlay } from '../components/video/CameraCanvasOverlay';
import { TrackedPerson, DetectedObject } from '../types';
import { API_BASE } from '../services/api';

export const ImageAnalysis: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [copied, setCopied] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 1280, height: 720 });
  const [detections, setDetections] = useState<TrackedPerson[]>([]);
  const [sceneObjects, setSceneObjects] = useState<DetectedObject[]>([]);
  const [showObjectsOverlay, setShowObjectsOverlay] = useState(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleImageSelected = async (file: File) => {
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = async (e) => {
      setSelectedImage(e.target?.result as string);
      setIsProcessing(true);

      try {
        // Send real image to backend endpoint
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE}/detection/image`, {
          method: 'POST',
          body: formData
        });

        if (res.ok) {
          const data = await res.json();
          setImageDimensions({ width: data.width || 1280, height: data.height || 720 });
          setDetections(data.detections || []);
          setSceneObjects(data.objects || []);
        }
      } catch (err) {
        console.error("Error analyzing image:", err);
      } finally {
        setIsProcessing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleCopyJson = () => {
    const payload = {
      people_count: detections.length,
      people: detections,
      objects_count: sceneObjects.length,
      objects: sceneObjects
    };
    navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-[#0a0f19] border border-slate-800 p-4 rounded-xl flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <ImageIcon className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              REAL IMAGE AI FORENSIC & OBJECT-ACTIVITY ANALYZER
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              <span className="text-purple-400 font-bold">YOLOv11 Multi-Object Detection</span> • Person-Object Interaction • <span className="text-cyan-400 font-bold">best_human_activity_v2.keras</span>
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {selectedImage && (
            <button
              onClick={() => setShowObjectsOverlay(!showObjectsOverlay)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                showObjectsOverlay 
                  ? 'bg-purple-950/40 border-purple-500/50 text-purple-300' 
                  : 'bg-slate-900 border-slate-800 text-slate-400'
              }`}
            >
              {showObjectsOverlay ? 'HIDE OBJECTS' : 'SHOW OBJECTS'}
            </button>
          )}

          {selectedImage && (
            <button
              onClick={() => { setSelectedImage(null); setImageFile(null); setDetections([]); setSceneObjects([]); }}
              className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono"
            >
              UPLOAD ANOTHER IMAGE
            </button>
          )}
        </div>
      </div>

      {/* Upload Zone */}
      {!selectedImage && (
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-cyan-500/60 bg-slate-950/40 hover:bg-slate-900/40 rounded-2xl p-12 text-center cursor-pointer transition-all space-y-4"
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            onChange={(e) => e.target.files && handleImageSelected(e.target.files[0])}
          />
          <div className="w-16 h-16 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mx-auto text-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.15)]">
            <Upload className="w-8 h-8" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white font-mono">
              Upload Image for YOLOv11 Activity & Object Detection
            </h3>
            <p className="text-xs text-slate-400 font-mono mt-1">
              Supports JPEG, PNG, WEBP, BMP • Simultaneous person tracking, object spatial association, and activity recognition
            </p>
          </div>
        </div>
      )}

      {/* Result Viewport */}
      {selectedImage && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="relative aspect-video w-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl surveillance-feed">
              <img
                src={selectedImage}
                alt="Analyzed Evidence"
                className="w-full h-full object-contain"
              />
              <CameraCanvasOverlay
                people={detections}
                objects={sceneObjects}
                videoWidth={imageDimensions.width}
                videoHeight={imageDimensions.height}
                showBoundingBoxes={true}
                showTrackingIds={true}
                showActivityLabels={true}
                showObjects={showObjectsOverlay}
              />
            </div>

            {/* Scene Objects Summary Bar */}
            {sceneObjects.length > 0 && (
              <div className="soc-panel p-3.5 rounded-xl flex items-center justify-between flex-wrap gap-2 text-xs font-mono">
                <div className="flex items-center space-x-2 text-purple-400 font-bold">
                  <Box className="w-4 h-4" />
                  <span>YOLOv11 DETECTED SCENE OBJECTS ({sceneObjects.length}):</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {sceneObjects.map((obj, i) => (
                    <span key={i} className="px-2 py-0.5 rounded bg-purple-950/60 border border-purple-500/30 text-purple-300 text-[11px]">
                      {obj.label} ({obj.confidence.toFixed(0)}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Detections JSON & Metadata */}
          <div className="space-y-4">
            <div className="soc-panel p-4 rounded-xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                  Person & Object Interactions ({detections.length})
                </span>
                {(detections.length > 0 || sceneObjects.length > 0) && (
                  <button
                    onClick={handleCopyJson}
                    className="text-xs font-mono text-slate-400 hover:text-cyan-300 flex items-center space-x-1"
                  >
                    <Copy className="w-3.5 h-3.5" />
                    <span>{copied ? 'COPIED' : 'COPY JSON'}</span>
                  </button>
                )}
              </div>

              {detections.length === 0 ? (
                <div className="py-8 text-center text-xs font-mono text-slate-500">
                  {isProcessing ? "Running YOLOv11 neural inference..." : "No persons detected in this image."}
                </div>
              ) : (
                <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                  {detections.map((d) => (
                    <div key={d.track_id} className="bg-slate-900/70 p-3 rounded-lg border border-slate-800 text-xs font-mono space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-white">Person #{d.track_id}</span>
                        <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-bold">
                          {d.activity}
                        </span>
                      </div>
                      <div className="flex justify-between text-slate-400 text-[11px]">
                        <span>Confidence: <strong className="text-emerald-400">{d.confidence}%</strong></span>
                        <span>Threat: <strong className={d.threat_level === 'NORMAL' ? 'text-slate-300' : 'text-red-400'}>{d.threat_level}</strong></span>
                      </div>

                      {/* Interacting Objects breakdown */}
                      {d.interacting_objects && d.interacting_objects.length > 0 ? (
                        <div className="pt-1.5 border-t border-slate-800/80 flex items-start space-x-1.5">
                          <LinkIcon className="w-3.5 h-3.5 text-purple-400 mt-0.5 shrink-0" />
                          <div className="space-y-1">
                            <span className="text-[10px] text-slate-400 block">Interacting Objects:</span>
                            <div className="flex flex-wrap gap-1">
                              {d.interacting_objects.map((io, idx) => (
                                <span key={idx} className="px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-300 border border-purple-500/30 text-[10px] font-bold">
                                  {io.label} ({io.confidence}%)
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="pt-1 border-t border-slate-800/80 text-[10px] text-slate-500">
                          No direct object interaction detected.
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Raw JSON Dump */}
              {(detections.length > 0 || sceneObjects.length > 0) && (
                <div className="pt-2 border-t border-slate-800">
                  <span className="text-[10px] font-mono text-slate-500 block mb-1">RAW YOLOv11 INFERENCE PAYLOAD</span>
                  <pre className="bg-slate-950 p-2.5 rounded-lg text-[10px] font-mono text-slate-300 max-h-36 overflow-y-auto border border-slate-800">
                    {JSON.stringify({ detections, scene_objects: sceneObjects }, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
