import React from 'react';
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react';

interface PTZControlsProps {
  onPTZCommand?: (cmd: string) => void;
}

export const PTZControls: React.FC<PTZControlsProps> = ({ onPTZCommand }) => {
  const handleCmd = (cmd: string) => {
    if (onPTZCommand) onPTZCommand(cmd);
  };

  return (
    <div className="bg-[#0c121d] border border-slate-800 p-3 rounded-lg flex flex-col items-center space-y-3">
      <div className="text-[10px] font-mono font-bold tracking-wider text-slate-400 uppercase">
        Digital PTZ Joystick
      </div>

      {/* Directional Pad */}
      <div className="relative w-28 h-28 bg-slate-900 border border-slate-800 rounded-full flex items-center justify-center shadow-inner">
        {/* Up */}
        <button
          onClick={() => handleCmd('UP')}
          className="absolute top-1 p-1 text-slate-400 hover:text-cyan-400 active:scale-90 transition-transform"
          title="Pan Up"
        >
          <ChevronUp className="w-5 h-5" />
        </button>
        {/* Down */}
        <button
          onClick={() => handleCmd('DOWN')}
          className="absolute bottom-1 p-1 text-slate-400 hover:text-cyan-400 active:scale-90 transition-transform"
          title="Pan Down"
        >
          <ChevronDown className="w-5 h-5" />
        </button>
        {/* Left */}
        <button
          onClick={() => handleCmd('LEFT')}
          className="absolute left-1 p-1 text-slate-400 hover:text-cyan-400 active:scale-90 transition-transform"
          title="Pan Left"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        {/* Right */}
        <button
          onClick={() => handleCmd('RIGHT')}
          className="absolute right-1 p-1 text-slate-400 hover:text-cyan-400 active:scale-90 transition-transform"
          title="Pan Right"
        >
          <ChevronRight className="w-5 h-5" />
        </button>

        {/* Center Recenter */}
        <button
          onClick={() => handleCmd('HOME')}
          className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-400 hover:text-cyan-400 hover:border-cyan-500/50"
          title="Recenter Camera Home"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Zoom Controls */}
      <div className="flex items-center space-x-2 w-full">
        <button
          onClick={() => handleCmd('ZOOM_IN')}
          className="flex-1 flex items-center justify-center space-x-1 py-1.5 px-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono border border-slate-700"
        >
          <ZoomIn className="w-3.5 h-3.5 text-cyan-400" />
          <span>ZOOM +</span>
        </button>
        <button
          onClick={() => handleCmd('ZOOM_OUT')}
          className="flex-1 flex items-center justify-center space-x-1 py-1.5 px-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono border border-slate-700"
        >
          <ZoomOut className="w-3.5 h-3.5 text-cyan-400" />
          <span>ZOOM -</span>
        </button>
      </div>
    </div>
  );
};
