import React from 'react';
import {
  BrainCircuit,
  BookOpen,
  FlaskConical,
  Database,
  Award,
  Layers,
  Bot,
  Settings,
  Sun,
  Moon,
} from 'lucide-react';

export type ActiveTab =
  | 'dashboard'
  | 'learn'
  | 'simulator'
  | 'datasets'
  | 'quizzes'
  | 'experiments'
  | 'tutor';

interface TopNavProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  isDark: boolean;
  setIsDark: (dark: boolean) => void;
  onOpenSettings: () => void;
  onQuickSimulate: () => void;
}

export const TopNav: React.FC<TopNavProps> = ({
  activeTab,
  setActiveTab,
  isDark,
  setIsDark,
  onOpenSettings,
  onQuickSimulate,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/90 backdrop-blur-md px-6 py-3 transition-colors">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-8">
        {/* Zone 1: Brand Wordmark (Single text element) */}
        <button
          onClick={() => setActiveTab('dashboard')}
          className="flex items-center gap-2.5 text-left text-slate-100 hover:text-indigo-400 transition-colors whitespace-nowrap shrink-0 group"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center text-white shadow-sm shadow-indigo-500/20 group-hover:scale-105 transition-transform">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <span className="font-bold text-base tracking-tight font-display">
            ML Algorithm Simulator
          </span>
        </button>

        {/* Zone 2: Navigation Links (Single line, text with active state) */}
        <nav className="hidden lg:flex items-center gap-1 xl:gap-2">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'dashboard'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Dashboard
          </button>

          <button
            onClick={() => setActiveTab('learn')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'learn'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Learn ML
          </button>

          <button
            onClick={() => setActiveTab('simulator')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'simulator'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Algorithm Simulator
          </button>

          <button
            onClick={() => setActiveTab('datasets')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'datasets'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Dataset Explorer
          </button>

          <button
            onClick={() => setActiveTab('quizzes')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'quizzes'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Quizzes & Badges
          </button>

          <button
            onClick={() => setActiveTab('experiments')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'experiments'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            Experiments
          </button>

          <button
            onClick={() => setActiveTab('tutor')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
              activeTab === 'tutor'
                ? 'bg-slate-800 text-indigo-400 font-semibold'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/60'
            }`}
          >
            AI Tutor
          </button>
        </nav>

        {/* Zone 3: Primary Action & Controls */}
        <div className="flex items-center gap-2.5 shrink-0">
          <button
            onClick={() => setIsDark(!isDark)}
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-400" />}
          </button>

          <button
            onClick={onOpenSettings}
            title="Simulator Settings"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>

          <button
            onClick={onQuickSimulate}
            className="px-3.5 py-1.5 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow-sm shadow-indigo-600/30 transition-all hover:scale-[1.02] active:scale-[0.98] whitespace-nowrap shrink-0"
          >
            Start Simulation
          </button>
        </div>
      </div>
    </header>
  );
};
