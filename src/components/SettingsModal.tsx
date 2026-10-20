import React, { useState, useEffect } from 'react';
import {
  X,
  Settings,
  Server,
  Cpu,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Sun,
  Moon,
  Trash2,
} from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  isDark: boolean;
  setIsDark: (val: boolean) => void;
  onResetProgress: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  isDark,
  setIsDark,
  onResetProgress,
}) => {
  const [pythonStatus, setPythonStatus] = useState<{
    online: boolean;
    checking: boolean;
    message?: string;
  }>({ online: false, checking: false });

  const checkPythonBackend = async () => {
    setPythonStatus((p) => ({ ...p, checking: true }));
    try {
      const res = await fetch('/api/python-status');
      const data = await res.json();
      setPythonStatus({
        online: data.online,
        checking: false,
        message: data.message,
      });
    } catch {
      setPythonStatus({
        online: false,
        checking: false,
        message: 'Could not connect to external Python FastAPI service.',
      });
    }
  };

  useEffect(() => {
    if (isOpen) {
      checkPythonBackend();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg p-6 space-y-6 shadow-2xl relative">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2 text-white font-bold text-base">
            <Settings className="w-5 h-5 text-indigo-400" />
            <span>Simulator Settings & Engine Diagnostics</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4 text-xs">
          {/* Dual Execution Engine Architecture */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white uppercase tracking-wider text-[11px]">
                Machine Learning Execution Engines
              </span>
              <button
                onClick={checkPythonBackend}
                className="text-[11px] text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
              >
                <RefreshCw className={`w-3 h-3 ${pythonStatus.checking ? 'animate-spin' : ''}`} />
                <span>Check Status</span>
              </button>
            </div>

            {/* Engine 1: Built-in */}
            <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/60 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <div className="font-semibold text-emerald-300">
                  Built-in Pure Mathematical Engine (Active)
                </div>
                <p className="text-slate-400 text-[11px]">
                  Zero-dependency analytical least squares, Gini recursive decision trees, and Lloyd&apos;s K-Means clustering running in-browser with instantaneous execution.
                </p>
              </div>
            </div>

            {/* Engine 2: Python FastAPI Microservice */}
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-start gap-3">
              <Server className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
              <div className="space-y-0.5 flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">
                    External Python FastAPI (Scikit-Learn)
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded font-mono ${pythonStatus.online ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>
                    {pythonStatus.online ? 'Online' : 'Standby / Optional'}
                  </span>
                </div>
                <p className="text-slate-400 text-[11px]">
                  Optional separate FastAPI service running in <code>/backend</code>.
                  To launch: <code>uvicorn main:app --port 8000</code>.
                </p>
              </div>
            </div>
          </div>

          {/* Theme Mode Toggle */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <div className="font-bold text-white">Visual Theme</div>
              <div className="text-slate-400 text-[11px]">
                Toggle between dark and light appearance
              </div>
            </div>

            <button
              onClick={() => setIsDark(!isDark)}
              className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-200 font-medium flex items-center gap-2 hover:bg-slate-800 transition-colors"
            >
              {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-400" />}
              <span>{isDark ? 'Dark Mode' : 'Light Mode'}</span>
            </button>
          </div>

          {/* Reset Course Progress */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
            <div>
              <div className="font-bold text-white">Reset Course Progress</div>
              <div className="text-slate-400 text-[11px]">
                Clear completed lessons, quiz scores, and saved experiments
              </div>
            </div>

            <button
              onClick={() => {
                if (confirm('Are you sure you want to reset all progress?')) {
                  onResetProgress();
                  onClose();
                }
              }}
              className="px-3 py-1.5 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 font-medium flex items-center gap-1.5 hover:bg-rose-900/60 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
