import React, { useState } from 'react';
import {
  FlaskConical,
  Trash2,
  Copy,
  Download,
  ArrowRight,
  TrendingUp,
  Sliders,
  CheckCircle,
  FileSpreadsheet,
} from 'lucide-react';
import { Experiment } from '../types/ml';

interface ExperimentWorkspaceProps {
  experiments: Experiment[];
  onDeleteExperiment: (id: string) => void;
  onDuplicateExperiment: (exp: Experiment) => void;
  onLaunchExperiment: (exp: Experiment) => void;
}

export const ExperimentWorkspace: React.FC<ExperimentWorkspaceProps> = ({
  experiments,
  onDeleteExperiment,
  onDuplicateExperiment,
  onLaunchExperiment,
}) => {
  const [selectedForComparison, setSelectedForComparison] = useState<string[]>([]);

  const toggleComparison = (id: string) => {
    if (selectedForComparison.includes(id)) {
      setSelectedForComparison(selectedForComparison.filter((i) => i !== id));
    } else {
      if (selectedForComparison.length >= 4) return;
      setSelectedForComparison([...selectedForComparison, id]);
    }
  };

  const comparedExperiments = experiments.filter((e) => selectedForComparison.includes(e.id));

  const exportAllJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(experiments, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `ml_experiments_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-indigo-400">
            <FlaskConical className="w-4 h-4" />
            <span>Machine Learning Experiment Benchmarking</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            My Experiments & Comparisons
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Compare model hyperparameters, feature scaling choices, and evaluation metrics side-by-side.
          </p>
        </div>

        {experiments.length > 0 && (
          <button
            onClick={exportAllJSON}
            className="px-3.5 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Experiments (JSON)</span>
          </button>
        )}
      </div>

      {experiments.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <FlaskConical className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-base font-bold text-white">No Saved Experiments Yet</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
            Run any algorithm simulation in the Algorithm Simulator and click &quot;Save Experiment&quot; to inspect parameter benchmarks here!
          </p>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Side-by-Side Comparison Matrix */}
          {comparedExperiments.length >= 2 && (
            <div className="p-6 rounded-2xl bg-slate-900 border border-indigo-500/40 space-y-4 animate-fadeIn">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs font-bold text-indigo-400">
                  <TrendingUp className="w-4 h-4" />
                  <span>Side-by-Side Benchmark Matrix ({comparedExperiments.length} selected)</span>
                </div>
                <button
                  onClick={() => setSelectedForComparison([])}
                  className="text-xs text-slate-400 hover:text-white"
                >
                  Clear Selection
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border border-slate-800 rounded-xl overflow-hidden">
                  <thead className="bg-slate-950 text-slate-400 text-[11px] uppercase tracking-wider">
                    <tr>
                      <th className="p-3">Attribute</th>
                      {comparedExperiments.map((exp) => (
                        <th key={exp.id} className="p-3 text-white font-bold">
                          {exp.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 bg-slate-950/40 font-mono">
                    <tr>
                      <td className="p-3 font-sans text-slate-400">Algorithm</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3 text-indigo-400 font-bold">{exp.algorithmName}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans text-slate-400">Dataset</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3">{exp.datasetName}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans text-slate-400">Scaling</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3 capitalize">{exp.preprocessing.scaleMethod}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans text-slate-400">Accuracy / R²</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3 text-emerald-400 font-bold text-sm">
                          {exp.metrics.r2 !== undefined && `R²: ${exp.metrics.r2}`}
                          {exp.metrics.accuracy !== undefined && `Acc: ${(exp.metrics.accuracy * 100).toFixed(1)}%`}
                          {exp.metrics.silhouetteScore !== undefined && `Sil: ${exp.metrics.silhouetteScore}`}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans text-slate-400">MSE / Error</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3">{exp.metrics.mse ?? exp.metrics.inertia ?? 'N/A'}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans text-slate-400">Training Time</td>
                      {comparedExperiments.map((exp) => (
                        <td key={exp.id} className="p-3">{exp.metrics.trainTimeMs} ms</td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Experiments Table */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>All Saved Experiments ({experiments.length})</span>
              <span>Check boxes to compare up to 4 models side-by-side</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[11px] font-semibold">
                  <tr>
                    <th className="py-3 px-4 w-10">Compare</th>
                    <th className="py-3 px-4">Experiment Name</th>
                    <th className="py-3 px-4">Algorithm</th>
                    <th className="py-3 px-4">Dataset</th>
                    <th className="py-3 px-4">Primary Metric</th>
                    <th className="py-3 px-4">Date</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {experiments.map((exp) => {
                    const isChecked = selectedForComparison.includes(exp.id);
                    return (
                      <tr key={exp.id} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 px-4">
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => toggleComparison(exp.id)}
                            className="rounded accent-indigo-600"
                          />
                        </td>
                        <td className="py-3 px-4 font-sans font-semibold text-white">
                          {exp.name}
                        </td>
                        <td className="py-3 px-4 text-indigo-400 font-sans font-medium">
                          {exp.algorithmName}
                        </td>
                        <td className="py-3 px-4 font-sans text-slate-400">
                          {exp.datasetName}
                        </td>
                        <td className="py-3 px-4 text-emerald-400 font-bold">
                          {exp.metrics.r2 !== undefined && `R² = ${exp.metrics.r2}`}
                          {exp.metrics.accuracy !== undefined && `Acc = ${(exp.metrics.accuracy * 100).toFixed(1)}%`}
                          {exp.metrics.silhouetteScore !== undefined && `Sil = ${exp.metrics.silhouetteScore}`}
                        </td>
                        <td className="py-3 px-4 text-slate-500 font-sans text-[11px]">
                          {new Date(exp.createdAt).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td className="py-3 px-4 text-right font-sans">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={() => onDuplicateExperiment(exp)}
                              title="Duplicate Experiment"
                              className="p-1.5 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                            >
                              <Copy className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => onDeleteExperiment(exp.id)}
                              title="Delete Experiment"
                              className="p-1.5 rounded text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
