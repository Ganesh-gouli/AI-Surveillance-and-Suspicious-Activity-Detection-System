import React, { createContext, useContext, useState, useCallback } from 'react';
import { CheckCircle2, AlertTriangle, AlertOctagon, Info, X } from 'lucide-react';

export type ToastType = 'success' | 'warning' | 'error' | 'info';

export interface ToastItem {
  id: string;
  title: string;
  message?: string;
  type: ToastType;
  duration?: number;
}

interface ToastContextType {
  toast: (options: { title: string; message?: string; type?: ToastType; duration?: number }) => void;
  success: (title: string, message?: string) => void;
  warning: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    ({
      title,
      message,
      type = 'info',
      duration = 4000
    }: {
      title: string;
      message?: string;
      type?: ToastType;
      duration?: number;
    }) => {
      const id = `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
      const newToast: ToastItem = { id, title, message, type, duration };
      setToasts((prev) => [...prev, newToast]);

      if (duration > 0) {
        setTimeout(() => {
          removeToast(id);
        }, duration);
      }
    },
    [removeToast]
  );

  const success = useCallback((title: string, message?: string) => toast({ title, message, type: 'success' }), [toast]);
  const warning = useCallback((title: string, message?: string) => toast({ title, message, type: 'warning' }), [toast]);
  const error = useCallback((title: string, message?: string) => toast({ title, message, type: 'error' }), [toast]);
  const info = useCallback((title: string, message?: string) => toast({ title, message, type: 'info' }), [toast]);

  return (
    <ToastContext.Provider value={{ toast, success, warning, error, info }}>
      {children}
      {/* Toast Render Viewport */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col space-y-2.5 max-w-sm w-full pointer-events-none">
        {toasts.map((t) => {
          const isSuccess = t.type === 'success';
          const isWarning = t.type === 'warning';
          const isError = t.type === 'error';

          return (
            <div
              key={t.id}
              className={`pointer-events-auto p-3.5 rounded-xl border backdrop-blur-md shadow-2xl flex items-start space-x-3 transition-all animate-in slide-in-from-bottom-5 duration-200 ${
                isSuccess
                  ? 'bg-slate-900/95 border-emerald-500/40 text-emerald-200 shadow-emerald-950/40'
                  : isWarning
                  ? 'bg-slate-900/95 border-amber-500/40 text-amber-200 shadow-amber-950/40'
                  : isError
                  ? 'bg-slate-900/95 border-red-500/40 text-red-200 shadow-red-950/40'
                  : 'bg-slate-900/95 border-cyan-500/40 text-cyan-200 shadow-cyan-950/40'
              }`}
            >
              <div className="mt-0.5 shrink-0">
                {isSuccess && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                {isWarning && <AlertTriangle className="w-4 h-4 text-amber-400" />}
                {isError && <AlertOctagon className="w-4 h-4 text-red-400" />}
                {t.type === 'info' && <Info className="w-4 h-4 text-cyan-400" />}
              </div>

              <div className="flex-1 min-w-0 font-mono">
                <div className="text-xs font-bold text-white tracking-wide">{t.title}</div>
                {t.message && <div className="text-[11px] text-slate-300 mt-0.5 leading-relaxed">{t.message}</div>}
              </div>

              <button
                onClick={() => removeToast(t.id)}
                className="text-slate-400 hover:text-white p-1 transition-colors"
                aria-label="Close notification"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};
