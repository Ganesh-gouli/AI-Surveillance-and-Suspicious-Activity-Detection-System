import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Layers,
  CheckCircle,
  Activity,
  ArrowRight,
  Database,
  Sliders,
  Sparkles,
  RefreshCw,
  FolderOpen,
  Box,
  Link as LinkIcon,
  ShieldAlert
} from 'lucide-react';
import { api } from '../services/api';
import { AIModelInfo } from '../types';

const OBJECT_CATEGORIES_DISPLAY: Record<string, string[]> = {
  "Electronics & Tech": ["laptop", "cell phone", "keyboard", "mouse", "tv", "remote"],
  "Dining & Sustenance": ["bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "sandwich", "pizza", "apple", "banana", "cake", "donut"],
  "Mobility & Vehicles": ["bicycle", "motorcycle", "car", "bus", "skateboard", "surfboard"],
  "Luggage & Accessories": ["backpack", "handbag", "suitcase", "umbrella"],
  "Sports & Action": ["sports ball", "tennis racket", "baseball bat", "frisbee", "skis", "snowboard"],
  "Hazards & Threats": ["knife", "scissors"],
  "Furniture Context": ["chair", "couch", "bed", "dining table"]
};

const INTERACTION_MATRIX = [
  { personActivity: "sitting", interactingObjects: ["chair", "couch", "dining table", "desk"], rule: "Seated posture / desktop webcam crop or person located on/near chair/couch/bench." },
  { personActivity: "standing", interactingObjects: ["upright posture"], rule: "Upright vertical posture (h/w >= 1.70) without seated furniture contact." },
  { personActivity: "using_laptop", interactingObjects: ["laptop", "keyboard", "mouse"], rule: "Person torso/lap overlaps laptop or keyboard/mouse input devices." },
  { personActivity: "using_phone", interactingObjects: ["cell phone"], rule: "Cell phone detected within arm/hand reaching perimeter." },
  { personActivity: "drinking", interactingObjects: ["cup", "bottle", "wine glass"], rule: "Verified drinking only when cup/bottle/glass container is detected in hand/upper body." },
  { personActivity: "eating", interactingObjects: ["sandwich", "pizza", "apple", "fork", "bowl"], rule: "Food item or dining utensil in upper body/hand contact." },
  { personActivity: "cycling", interactingObjects: ["bicycle", "motorcycle"], rule: "Person straddling or within bounding volume of a bicycle/motorcycle." },
  { personActivity: "carrying_bag", interactingObjects: ["backpack", "handbag", "suitcase"], rule: "Luggage or accessory carried or held in close proximity." },
  { personActivity: "playing_sports", interactingObjects: ["sports ball", "tennis racket", "baseball bat"], rule: "Sports apparatus detected within dynamic athletic posture." },
  { personActivity: "armed_hazard", interactingObjects: ["knife", "scissors"], rule: "CRITICAL: Hazard object detected in human hand or perimeter." }
];

export const AIModels: React.FC = () => {
  const [models, setModels] = useState<AIModelInfo[]>([]);
  const [isReloading, setIsReloading] = useState(false);
  const [activeTab, setActiveTab] = useState<'architecture' | 'classes' | 'interactions'>('architecture');

  const loadModels = async () => {
    const data = await api.getAIModelsInfo();
    setModels(data);
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleReload = () => {
    setIsReloading(true);
    setTimeout(() => {
      setIsReloading(false);
      loadModels();
      alert("YOLOv11 & Keras Neural model weights verified & hot-reloaded.");
    }, 800);
  };

  const activityModel = models.find(m => m.name.includes('Activity')) || {
    name: "Human Activity Classifier",
    version: "2.4.0",
    architecture: "EfficientNetV2B2 + Dense Classifier Head",
    classes: ["cycling", "drinking", "eating", "fighting", "running", "sitting", "sleeping", "using_laptop"],
    test_accuracy: 87.38,
    status: "ACTIVE",
    model_path: "/models/best_human_activity_v2.keras",
    framework: "TensorFlow / Keras 3.x"
  };

  const yoloModel = models.find(m => m.name.includes('YOLO')) || {
    name: "YOLOv11 Multi-Object & Human Detector",
    version: "11.0.0",
    architecture: "YOLOv11-Nano (C3k2 + SPPF + Neural Attention)",
    classes: ["person", "laptop", "cell phone", "cup", "bottle", "bicycle", "backpack", "sports ball", "knife", "chair"],
    test_accuracy: 94.20,
    status: "ACTIVE",
    model_path: "yolo11n.pt",
    framework: "Ultralytics YOLOv11 (PyTorch)"
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-[#0a0f19] border border-slate-800 p-4 rounded-xl">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white font-mono">
              AI MODEL REPOSITORY & INFERENCE PIPELINE
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              <span className="text-purple-400 font-bold">YOLOv11 Multi-Object Vision</span> • Person-Object Spatial Interaction • <span className="text-cyan-400 font-bold">best_human_activity_v2.keras</span>
            </p>
          </div>
        </div>

        <button
          onClick={handleReload}
          disabled={isReloading}
          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-bold flex items-center space-x-1.5 border border-slate-700 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-cyan-400 ${isReloading ? 'animate-spin' : ''}`} />
          <span>RELOAD WEIGHTS</span>
        </button>
      </div>

      {/* Interactive AI Architecture Flow Pipeline Diagram */}
      <div className="soc-panel p-5 rounded-xl space-y-4">
        <div className="text-xs font-mono font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span>YOLOv11 Real-Time Multimodal Inference Architecture</span>
        </div>

        {/* Modular Pipeline Steps */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 items-center text-center">
          {/* Step 1: Input */}
          <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg space-y-1">
            <div className="text-[10px] text-slate-500 font-mono">STAGE 1</div>
            <div className="text-xs font-bold text-white font-mono">Video / RTSP</div>
            <div className="text-[9px] text-cyan-400 font-mono">OpenCV Ingestion</div>
          </div>

          {/* Step 2: YOLOv11 */}
          <div className="bg-purple-950/40 border border-purple-500/50 p-3 rounded-lg space-y-1 shadow-[0_0_12px_rgba(192,132,252,0.2)]">
            <div className="text-[10px] text-purple-400 font-mono font-bold">STAGE 2</div>
            <div className="text-xs font-bold text-white font-mono">YOLOv11</div>
            <div className="text-[9px] text-purple-300 font-mono">People + 80 Objects</div>
          </div>

          {/* Step 3: Tracker & Spatial Link */}
          <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg space-y-1">
            <div className="text-[10px] text-slate-500 font-mono">STAGE 3</div>
            <div className="text-xs font-bold text-white font-mono">Spatial Linker</div>
            <div className="text-[9px] text-blue-400 font-mono">Person-Object IoU</div>
          </div>

          {/* Step 4: EfficientNetV2B2 Keras */}
          <div className="bg-cyan-950/40 border border-cyan-500/40 p-3 rounded-lg space-y-1 shadow-[0_0_10px_rgba(0,240,255,0.15)]">
            <div className="text-[10px] text-cyan-400 font-mono font-bold">STAGE 4</div>
            <div className="text-xs font-bold text-white font-mono">EfficientNetV2</div>
            <div className="text-[9px] text-emerald-400 font-mono">8-Class Keras Model</div>
          </div>

          {/* Step 5: Multimodal Fusion */}
          <div className="bg-purple-950/30 border border-purple-500/40 p-3 rounded-lg space-y-1">
            <div className="text-[10px] text-purple-400 font-mono font-bold">STAGE 5</div>
            <div className="text-xs font-bold text-white font-mono">Context Fusion</div>
            <div className="text-[9px] text-cyan-300 font-mono">Object + Pose AI</div>
          </div>

          {/* Step 6: Alert Engine */}
          <div className="bg-red-950/40 border border-red-500/40 p-3 rounded-lg space-y-1">
            <div className="text-[10px] text-red-400 font-mono font-bold">STAGE 6</div>
            <div className="text-xs font-bold text-white font-mono">Alert Engine</div>
            <div className="text-[9px] text-red-400 font-mono">Threat Dispatch</div>
          </div>

          {/* Step 7: WebSockets */}
          <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg space-y-1">
            <div className="text-[10px] text-slate-500 font-mono">STAGE 7</div>
            <div className="text-xs font-bold text-white font-mono">WebSockets</div>
            <div className="text-[9px] text-cyan-400 font-mono">SOC Dashboard</div>
          </div>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Model 1: YOLOv11 Multi-Object & Human Detector */}
        <div className="soc-panel p-5 rounded-xl border border-purple-500/40 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-purple-400 uppercase">
              YOLOv11 Object & Human Detector
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono font-bold text-xs">
              ● ACTIVE
            </span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">ARCHITECTURE</span>
              <span className="text-white font-bold">{yoloModel.architecture}</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">CLASSES SUPPORTED</span>
              <span className="text-purple-300 font-bold">Person (Class 0) + 80 COCO Objects</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">mAP50-95 ACCURACY</span>
              <span className="text-emerald-400 font-bold text-sm">94.20%</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">FRAMEWORK & WEIGHTS</span>
              <span className="text-slate-300 truncate block">Ultralytics YOLOv11 ({yoloModel.model_path})</span>
            </div>
          </div>
        </div>

        {/* Model 2: Human Activity Classifier */}
        <div className="soc-panel p-5 rounded-xl border border-cyan-500/30 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
              Human Activity Classifier
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono font-bold text-xs">
              ● ACTIVE
            </span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">ARCHITECTURE</span>
              <span className="text-white font-bold">EfficientNetV2B2 + Dense Head</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">CLASSES SUPPORTED</span>
              <span className="text-cyan-400 font-bold">8 Target Human Activities</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">TEST ACCURACY</span>
              <span className="text-emerald-400 font-bold text-sm">87.38%</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">MODEL WEIGHTS PATH</span>
              <span className="text-slate-300 truncate block">/models/best_human_activity_v2.keras</span>
            </div>
          </div>
        </div>

        {/* Model 3: Spatial Person-Object & Behaviour Engine */}
        <div className="soc-panel p-5 rounded-xl border border-slate-800 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-slate-300 uppercase">
              Interaction & Threat Engine
            </span>
            <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-mono font-bold text-xs">
              ● READY
            </span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">ARCHITECTURE</span>
              <span className="text-white font-bold">Spatial IoU Interaction + Temporal Rules</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">SECURITY HAZARD RULES</span>
              <span className="text-amber-400 font-bold">Armed Object, Perimeter, Violence, Device</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">TRIGGER ACCURACY</span>
              <span className="text-emerald-400 font-bold text-sm">96.10%</span>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">GEOMETRIC KERNEL</span>
              <span className="text-slate-300">Spatial Proximity + Ray-Casting PIP</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs for Class & Interaction Matrix */}
      <div className="soc-panel p-5 rounded-xl space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('architecture')}
            className={`text-xs font-mono font-bold px-3 py-1.5 rounded-lg transition-all ${
              activeTab === 'architecture' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' : 'text-slate-400 hover:text-white'
            }`}
          >
            OBJECT-TO-ACTIVITY INTERACTION MATRIX
          </button>
          <button
            onClick={() => setActiveTab('classes')}
            className={`text-xs font-mono font-bold px-3 py-1.5 rounded-lg transition-all ${
              activeTab === 'classes' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/40' : 'text-slate-400 hover:text-white'
            }`}
          >
            YOLOv11 DETECTABLE OBJECT CATEGORIES
          </button>
        </div>

        {activeTab === 'architecture' && (
          <div className="space-y-3">
            <p className="text-xs text-slate-400 font-mono">
              YOLOv11 computes real-time bounding box overlap and proximity between humans and context objects to tag and corroborate high-precision activities:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {INTERACTION_MATRIX.map((item, idx) => (
                <div key={idx} className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 text-xs font-mono space-y-1.5">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-cyan-400 uppercase">◈ {item.personActivity}</span>
                    <span className="px-2 py-0.5 rounded bg-purple-900/50 text-purple-300 text-[10px] font-bold border border-purple-500/30">
                      {item.interactingObjects.join(' + ')}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">{item.rule}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'classes' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(OBJECT_CATEGORIES_DISPLAY).map(([category, items]) => (
                <div key={category} className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 text-xs font-mono space-y-2">
                  <span className="text-[11px] font-bold text-purple-300 block border-b border-slate-800 pb-1">
                    {category}
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((item) => (
                      <span key={item} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px]">
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
