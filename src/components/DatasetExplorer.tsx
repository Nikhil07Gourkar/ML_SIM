import React, { useState, useMemo } from 'react';
import {
  Database,
  Upload,
  Search,
  Filter,
  BarChart2,
  Table as TableIcon,
  ScatterChart,
  Lightbulb,
  CheckCircle,
  AlertCircle,
  Play,
  FileSpreadsheet,
} from 'lucide-react';
import { DatasetFeature, DatasetInfo, DatasetRecord } from '../types/ml';
import { BUILT_IN_DATASETS } from '../ml/datasets';
import { parseCSVString } from '../ml/csvParser';
import { mean, stdDev, quantile, pearsonCorrelation } from '../ml/mathUtils';

interface DatasetExplorerProps {
  selectedDataset: DatasetInfo;
  onSelectDataset: (dataset: DatasetInfo) => void;
  onLaunchSimulation: (dataset: DatasetInfo) => void;
  onCustomDatasetAdded: (dataset: DatasetInfo) => void;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({
  selectedDataset,
  onSelectDataset,
  onLaunchSimulation,
  onCustomDatasetAdded,
}) => {
  const [activeTab, setActiveTab] = useState<'table' | 'statistics' | 'visualizer' | 'upload'>('table');
  const [searchQuery, setSearchQuery] = useState('');
  const [tableSearch, setTableSearch] = useState('');
  const [page, setPage] = useState(1);
  const rowsPerPage = 12;

  // Visualizer axes state
  const numericFeatures = useMemo(
    () => selectedDataset.features.filter((f) => f.type === 'numeric'),
    [selectedDataset]
  );
  const [axisX, setAxisX] = useState<string>(numericFeatures[0]?.name || '');
  const [axisY, setAxisY] = useState<string>(numericFeatures[1]?.name || numericFeatures[0]?.name || '');
  const [histogramFeature, setHistogramFeature] = useState<string>(numericFeatures[0]?.name || '');

  // CSV Upload state
  const [uploadText, setUploadText] = useState('');
  const [uploadErrors, setUploadErrors] = useState<string[]>([]);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [customName, setCustomName] = useState('My Custom Dataset');
  const [customTask, setCustomTask] = useState<'classification' | 'regression' | 'clustering'>('classification');
  const [customTarget, setCustomTarget] = useState<string>('');

  // Update axes when dataset changes
  React.useEffect(() => {
    if (numericFeatures.length > 0) {
      setAxisX(numericFeatures[0].name);
      setAxisY(numericFeatures[1]?.name || numericFeatures[0].name);
      setHistogramFeature(numericFeatures[0].name);
    }
  }, [selectedDataset.id]);

  // Filter datasets list
  const filteredDatasets = BUILT_IN_DATASETS.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.taskType.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Table pagination and filter
  const filteredRows = useMemo(() => {
    if (!tableSearch.trim()) return selectedDataset.data;
    const q = tableSearch.toLowerCase();
    return selectedDataset.data.filter((row) =>
      Object.values(row).some((val) => String(val).toLowerCase().includes(q))
    );
  }, [selectedDataset.data, tableSearch]);

  const totalPages = Math.ceil(filteredRows.length / rowsPerPage);
  const paginatedRows = filteredRows.slice((page - 1) * rowsPerPage, page * rowsPerPage);

  // Descriptive Statistics Calculation
  const statsTable = useMemo(() => {
    return numericFeatures.map((feat) => {
      const vals = selectedDataset.data
        .map((r) => r[feat.name])
        .filter((v) => typeof v === 'number' && !isNaN(v)) as number[];

      if (vals.length === 0) {
        return {
          name: feat.name,
          count: 0,
          mean: 0,
          std: 0,
          min: 0,
          p25: 0,
          median: 0,
          p75: 0,
          max: 0,
          missing: feat.missingCount,
        };
      }

      const sorted = [...vals].sort((a, b) => a - b);
      return {
        name: feat.name,
        count: vals.length,
        mean: +mean(vals).toFixed(2),
        std: +stdDev(vals).toFixed(2),
        min: +sorted[0].toFixed(2),
        p25: +quantile(sorted, 0.25).toFixed(2),
        median: +quantile(sorted, 0.5).toFixed(2),
        p75: +quantile(sorted, 0.75).toFixed(2),
        max: +sorted[sorted.length - 1].toFixed(2),
        missing: feat.missingCount,
      };
    });
  }, [selectedDataset, numericFeatures]);

  // Correlation Matrix Calculation
  const correlationMatrix = useMemo(() => {
    const matrix: { [f1: string]: { [f2: string]: number } } = {};
    for (const f1 of numericFeatures) {
      matrix[f1.name] = {};
      const v1 = selectedDataset.data.map((r) => Number(r[f1.name]) || 0);
      for (const f2 of numericFeatures) {
        const v2 = selectedDataset.data.map((r) => Number(r[f2.name]) || 0);
        matrix[f1.name][f2.name] = +pearsonCorrelation(v1, v2).toFixed(2);
      }
    }
    return matrix;
  }, [selectedDataset, numericFeatures]);

  // Calculated Pattern Insights
  const patternInsights = useMemo(() => {
    const insights: string[] = [];
    // 1. Find strongest positive and negative correlations
    let maxCorr = 0;
    let maxPair: [string, string] = ['', ''];
    let minCorr = 0;
    let minPair: [string, string] = ['', ''];

    for (let i = 0; i < numericFeatures.length; i++) {
      for (let j = i + 1; j < numericFeatures.length; j++) {
        const f1 = numericFeatures[i].name;
        const f2 = numericFeatures[j].name;
        const r = correlationMatrix[f1]?.[f2] ?? 0;
        if (r > maxCorr && r < 0.999) {
          maxCorr = r;
          maxPair = [f1, f2];
        }
        if (r < minCorr) {
          minCorr = r;
          minPair = [f1, f2];
        }
      }
    }

    if (maxCorr >= 0.7) {
      insights.push(
        `Strong positive correlation (r = ${maxCorr}) detected between "${maxPair[0]}" and "${maxPair[1]}". As one increases, the other systematically rises.`
      );
    }
    if (minCorr <= -0.6) {
      insights.push(
        `Strong negative correlation (r = ${minCorr}) between "${minPair[0]}" and "${minPair[1]}". These features move in opposing directions.`
      );
    }

    // 2. Class distribution check for classification
    if (selectedDataset.taskType === 'classification' && selectedDataset.targetColumn) {
      const counts: Record<string, number> = {};
      for (const row of selectedDataset.data) {
        const val = String(row[selectedDataset.targetColumn]);
        counts[val] = (counts[val] || 0) + 1;
      }
      const values = Object.values(counts);
      const minCount = Math.min(...values);
      const maxCount = Math.max(...values);
      if (maxCount / (minCount || 1) >= 2.0) {
        insights.push(
          `Moderate class imbalance detected in target "${selectedDataset.targetColumn}" (${maxCount} vs ${minCount} samples). Consider evaluating Precision/Recall rather than Accuracy.`
        );
      } else {
        insights.push(
          `Balanced class distribution across all target categories (${Object.keys(counts).join(', ')}). Accuracy is an appropriate primary evaluation metric.`
        );
      }
    }

    // 3. Recommended algorithm insight
    insights.push(
      `Based on ${selectedDataset.taskType} task and ${numericFeatures.length} numeric features, recommended models: ${selectedDataset.recommendedAlgorithms.join(', ')}.`
    );

    return insights;
  }, [selectedDataset, numericFeatures, correlationMatrix]);

  // Handle Custom CSV File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
      setUploadErrors(['Only CSV files are supported. Please upload a .csv file.']);
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setUploadErrors(['File size exceeds 5MB limit.']);
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      setUploadText(text);
      processUploadedCSV(text, file.name.replace(/\.csv$/i, ''));
    };
    reader.readAsText(file);
  };

  const processUploadedCSV = (csvStr: string, defaultName: string) => {
    const result = parseCSVString(csvStr);
    if (result.errors.length > 0) {
      setUploadErrors(result.errors);
      setUploadSuccess(null);
      return;
    }

    setUploadErrors([]);
    setCustomName(defaultName);
    setCustomTarget(result.detectedTarget || '');

    // Guess task type based on target
    const targetFeat = result.features.find((f) => f.name === result.detectedTarget);
    if (targetFeat) {
      setCustomTask(targetFeat.type === 'categorical' ? 'classification' : 'regression');
    }

    setUploadSuccess(`Successfully parsed ${result.rows} rows and ${result.columns} columns (${result.duplicateCount} duplicates found).`);
  };

  const finalizeCustomDataset = () => {
    const parsed = parseCSVString(uploadText);
    if (parsed.records.length === 0) return;

    const newDataset: DatasetInfo = {
      id: `custom_${Date.now()}`,
      name: customName || 'Custom Uploaded Dataset',
      taskType: customTask,
      difficulty: 'Beginner',
      description: `User uploaded dataset with ${parsed.rows} rows and ${parsed.columns} features.`,
      rows: parsed.rows,
      columns: parsed.columns,
      features: parsed.features,
      targetColumn: customTask !== 'clustering' ? customTarget : undefined,
      recommendedAlgorithms: customTask === 'classification'
        ? ['decision_tree', 'random_forest', 'logistic_regression']
        : customTask === 'regression'
        ? ['linear_regression', 'decision_tree']
        : ['kmeans', 'pca'],
      challenge: 'Evaluate model performance on this custom uploaded dataset and optimize hyperparameters.',
      data: parsed.records,
    };

    onCustomDatasetAdded(newDataset);
    onSelectDataset(newDataset);
    setActiveTab('table');
  };

  // Scatter plot points for SVG canvas
  const scatterPoints = useMemo(() => {
    if (!axisX || !axisY) return [];
    const pts = selectedDataset.data.map((r) => {
      const x = Number(r[axisX]) || 0;
      const y = Number(r[axisY]) || 0;
      const targetVal = selectedDataset.targetColumn ? String(r[selectedDataset.targetColumn]) : 'Cluster';
      return { x, y, targetVal };
    });

    const xVals = pts.map((p) => p.x);
    const yVals = pts.map((p) => p.y);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals) || minX + 1;
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals) || minY + 1;

    return pts.map((p) => ({
      ...p,
      normX: (p.x - minX) / (maxX - minX),
      normY: 1 - (p.y - minY) / (maxY - minY), // invert for SVG Y
    }));
  }, [selectedDataset, axisX, axisY]);

  // Histogram bins calculation
  const histogramBins = useMemo(() => {
    if (!histogramFeature) return [];
    const vals = selectedDataset.data
      .map((r) => Number(r[histogramFeature]))
      .filter((v) => !isNaN(v));

    if (vals.length === 0) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals) || min + 1;
    const numBins = 10;
    const binWidth = (max - min) / numBins;

    const bins: { label: string; count: number }[] = Array.from({ length: numBins }, (_, i) => ({
      label: `${(min + i * binWidth).toFixed(1)}`,
      count: 0,
    }));

    for (const v of vals) {
      let bIdx = Math.floor((v - min) / (binWidth || 1));
      if (bIdx >= numBins) bIdx = numBins - 1;
      if (bIdx < 0) bIdx = 0;
      bins[bIdx].count++;
    }

    return bins;
  }, [selectedDataset, histogramFeature]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fadeIn">
      {/* Header & Dataset Selector Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-indigo-400">
            <Database className="w-4 h-4" />
            <span>Dataset Exploration & Preprocessing Hub</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            {selectedDataset.name}
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            {selectedDataset.description}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-3.5 py-1.5 text-xs font-medium rounded-lg border transition-colors flex items-center gap-1.5 ${
              activeTab === 'upload'
                ? 'bg-indigo-600 text-white border-indigo-500'
                : 'bg-slate-900 text-slate-300 border-slate-800 hover:border-slate-700'
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload CSV</span>
          </button>

          <button
            onClick={() => onLaunchSimulation(selectedDataset)}
            className="px-4 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg shadow-sm shadow-indigo-600/30 transition-all flex items-center gap-1.5"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Launch in Simulator</span>
          </button>
        </div>
      </div>

      {/* Dataset Picker Cards Strip */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>Select built-in educational dataset:</span>
          <div className="relative w-48">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-slate-500" />
            <input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-md pl-8 pr-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
          {filteredDatasets.map((ds) => (
            <button
              key={ds.id}
              onClick={() => onSelectDataset(ds)}
              className={`p-2.5 rounded-lg text-left transition-all border ${
                selectedDataset.id === ds.id
                  ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-sm shadow-indigo-500/20'
                  : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="text-xs font-semibold truncate">{ds.name}</div>
              <div className="text-[11px] text-slate-500 capitalize flex items-center gap-1 mt-0.5">
                <span>{ds.taskType}</span>
                <span>·</span>
                <span>{ds.rows} rows</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Tabs Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-800 pt-2">
        <button
          onClick={() => setActiveTab('table')}
          className={`pb-2.5 px-3 text-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
            activeTab === 'table'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <TableIcon className="w-3.5 h-3.5" />
          <span>Interactive Data Table</span>
        </button>

        <button
          onClick={() => setActiveTab('statistics')}
          className={`pb-2.5 px-3 text-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
            activeTab === 'statistics'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileSpreadsheet className="w-3.5 h-3.5" />
          <span>Descriptive Statistics & Correlation</span>
        </button>

        <button
          onClick={() => setActiveTab('visualizer')}
          className={`pb-2.5 px-3 text-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
            activeTab === 'visualizer'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ScatterChart className="w-3.5 h-3.5" />
          <span>Scatter & Distribution Charts</span>
        </button>

        <button
          onClick={() => setActiveTab('upload')}
          className={`pb-2.5 px-3 text-xs font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
            activeTab === 'upload'
              ? 'border-indigo-500 text-indigo-400 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Upload className="w-3.5 h-3.5" />
          <span>Upload Custom CSV</span>
        </button>
      </div>

      {/* Tab 1: Interactive Data Table */}
      {activeTab === 'table' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="relative w-72">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search values in table..."
                value={tableSearch}
                onChange={(e) => {
                  setTableSearch(e.target.value);
                  setPage(1);
                }}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div className="text-xs text-slate-400">
              Showing {(page - 1) * rowsPerPage + 1} to{' '}
              {Math.min(page * rowsPerPage, filteredRows.length)} of {filteredRows.length} records
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 overflow-x-auto bg-slate-900/40">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold text-[11px]">
                <tr>
                  <th className="py-2.5 px-3 w-12 text-slate-500">#</th>
                  {selectedDataset.features.map((feat) => (
                    <th key={feat.name} className="py-2.5 px-3">
                      <div className="flex items-center gap-1">
                        <span>{feat.name}</span>
                        {feat.name === selectedDataset.targetColumn && (
                          <span className="text-[10px] text-indigo-400 font-bold lowercase">
                            (target)
                          </span>
                        )}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {paginatedRows.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2 px-3 text-slate-500 text-[11px]">
                      {(page - 1) * rowsPerPage + idx + 1}
                    </td>
                    {selectedDataset.features.map((feat) => (
                      <td key={feat.name} className="py-2 px-3 whitespace-nowrap">
                        {row[feat.name] !== undefined && row[feat.name] !== '' ? (
                          typeof row[feat.name] === 'number' ? (
                            Number(row[feat.name]).toFixed(2)
                          ) : (
                            String(row[feat.name])
                          )
                        ) : (
                          <span className="text-amber-500/70 italic font-sans text-[11px]">
                            NaN
                          </span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 border border-slate-800 rounded text-xs text-slate-300 transition-colors"
              >
                Previous
              </button>
              <span className="text-xs text-slate-400">
                Page {page} of {totalPages}
              </span>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="px-3 py-1 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 border border-slate-800 rounded text-xs text-slate-300 transition-colors"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Statistics & Correlation Matrix */}
      {activeTab === 'statistics' && (
        <div className="space-y-6">
          {/* Descriptive Statistics Table */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white">Descriptive Statistics (Numerical Features)</h3>
            <div className="rounded-xl border border-slate-800 overflow-x-auto bg-slate-900/40">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 text-[11px] font-semibold">
                  <tr>
                    <th className="py-2.5 px-3">Feature</th>
                    <th className="py-2.5 px-3">Count</th>
                    <th className="py-2.5 px-3">Mean</th>
                    <th className="py-2.5 px-3">Std Dev</th>
                    <th className="py-2.5 px-3">Min</th>
                    <th className="py-2.5 px-3">25% (Q1)</th>
                    <th className="py-2.5 px-3">Median</th>
                    <th className="py-2.5 px-3">75% (Q3)</th>
                    <th className="py-2.5 px-3">Max</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {statsTable.map((st) => (
                    <tr key={st.name} className="hover:bg-slate-800/30">
                      <td className="py-2 px-3 font-sans font-medium text-white">{st.name}</td>
                      <td className="py-2 px-3">{st.count}</td>
                      <td className="py-2 px-3">{st.mean}</td>
                      <td className="py-2 px-3">{st.std}</td>
                      <td className="py-2 px-3">{st.min}</td>
                      <td className="py-2 px-3">{st.p25}</td>
                      <td className="py-2 px-3">{st.median}</td>
                      <td className="py-2 px-3">{st.p75}</td>
                      <td className="py-2 px-3">{st.max}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Correlation Heatmap Grid */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">Pearson Correlation Heatmap (r)</h3>
                <p className="text-xs text-slate-400">
                  Measures linear association from -1.0 (inverse) to +1.0 (direct). Remember: Correlation does NOT imply causation.
                </p>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 overflow-x-auto p-4 bg-slate-900/40">
              <table className="w-full text-center text-xs font-mono">
                <thead>
                  <tr>
                    <th className="text-left py-2 px-2 text-slate-500 font-sans"></th>
                    {numericFeatures.map((f) => (
                      <th key={f.name} className="py-2 px-2 text-slate-400 font-sans truncate max-w-[100px]" title={f.name}>
                        {f.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {numericFeatures.map((f1) => (
                    <tr key={f1.name}>
                      <td className="text-left py-2 px-2 text-slate-300 font-sans font-medium truncate max-w-[120px]" title={f1.name}>
                        {f1.name}
                      </td>
                      {numericFeatures.map((f2) => {
                        const r = correlationMatrix[f1.name]?.[f2.name] ?? 0;
                        // Color scale from blue (-1) through slate (0) to amber/emerald (+1)
                        let bgClass = 'bg-slate-800/40 text-slate-300';
                        if (r > 0.7) bgClass = 'bg-emerald-600/30 text-emerald-300 font-bold';
                        else if (r > 0.4) bgClass = 'bg-emerald-500/15 text-emerald-400';
                        else if (r < -0.7) bgClass = 'bg-rose-600/30 text-rose-300 font-bold';
                        else if (r < -0.4) bgClass = 'bg-rose-500/15 text-rose-400';

                        return (
                          <td key={f2.name} className="p-1">
                            <div className={`py-1.5 px-2 rounded ${bgClass} transition-colors`}>
                              {r.toFixed(2)}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Automatic Pattern Insights Box */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-indigo-400">
              <Lightbulb className="w-4 h-4" />
              <span>Calculated Pattern & Relationship Discoveries</span>
            </div>
            <ul className="space-y-1.5 text-xs text-slate-300 list-disc list-inside">
              {patternInsights.map((ins, i) => (
                <li key={i}>{ins}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Tab 3: Interactive Visualizer (Scatter & Histogram) */}
      {activeTab === 'visualizer' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 2D Feature Scatter Plot */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h3 className="text-sm font-bold text-white">2D Feature Scatter Plot</h3>
              <div className="flex items-center gap-2 text-xs">
                <div>
                  <span className="text-slate-400 mr-1">X:</span>
                  <select
                    value={axisX}
                    onChange={(e) => setAxisX(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  >
                    {numericFeatures.map((f) => (
                      <option key={f.name} value={f.name}>{f.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <span className="text-slate-400 mr-1">Y:</span>
                  <select
                    value={axisY}
                    onChange={(e) => setAxisY(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
                  >
                    {numericFeatures.map((f) => (
                      <option key={f.name} value={f.name}>{f.name}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* SVG Interactive Scatter Canvas */}
            <div className="w-full h-72 bg-slate-950/80 rounded-xl border border-slate-800/80 p-4 relative flex items-center justify-center">
              <svg className="w-full h-full overflow-visible">
                {/* Axis lines */}
                <line x1="10" y1="90%" x2="95%" y2="90%" stroke="#334155" strokeWidth="1" />
                <line x1="10" y1="5%" x2="10" y2="90%" stroke="#334155" strokeWidth="1" />

                {/* Data points */}
                {scatterPoints.map((pt, idx) => {
                  const cx = 15 + pt.normX * 80 + '%';
                  const cy = 10 + pt.normY * 75 + '%';

                  // Color based on target value
                  let fill = '#6366f1';
                  if (pt.targetVal === 'Setosa' || pt.targetVal === 'Pass' || pt.targetVal === 'Benign') fill = '#10b981';
                  else if (pt.targetVal === 'Versicolor' || pt.targetVal === 'Fail' || pt.targetVal === 'Malignant') fill = '#f43f5e';
                  else if (pt.targetVal === 'Virginica' || pt.targetVal === 'Class 2') fill = '#0ea5e9';

                  return (
                    <circle
                      key={idx}
                      cx={cx}
                      cy={cy}
                      r="4.5"
                      fill={fill}
                      opacity="0.8"
                      className="hover:r-6 hover:opacity-100 transition-all cursor-pointer"
                    >
                      <title>{`${axisX}: ${pt.x}, ${axisY}: ${pt.y} (${pt.targetVal})`}</title>
                    </circle>
                  );
                })}
              </svg>
            </div>

            <div className="text-xs text-slate-400 text-center flex items-center justify-center gap-4">
              <span>Points: {scatterPoints.length}</span>
              <span className="text-slate-600">·</span>
              <span>Colored by: {selectedDataset.targetColumn || 'Cluster ID'}</span>
            </div>
          </div>

          {/* Feature Distribution Histogram */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">Feature Histogram Distribution</h3>
              <select
                value={histogramFeature}
                onChange={(e) => setHistogramFeature(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200"
              >
                {numericFeatures.map((f) => (
                  <option key={f.name} value={f.name}>{f.name}</option>
                ))}
              </select>
            </div>

            <div className="w-full h-72 bg-slate-950/80 rounded-xl border border-slate-800/80 p-4 flex items-end gap-2">
              {histogramBins.map((bin, idx) => {
                const maxCount = Math.max(...histogramBins.map((b) => b.count)) || 1;
                const heightPercent = Math.max(8, (bin.count / maxCount) * 85);
                return (
                  <div key={idx} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end group">
                    <span className="text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
                      {bin.count}
                    </span>
                    <div
                      className="w-full bg-indigo-600/70 hover:bg-indigo-500 rounded-t transition-all"
                      style={{ height: `${heightPercent}%` }}
                    />
                    <span className="text-[9px] text-slate-500 font-mono truncate w-full text-center">
                      {bin.label}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="text-xs text-slate-400 text-center">
              Distribution of values for &quot;{histogramFeature}&quot; across 10 equal-width bins
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Custom CSV Upload & Data Ingestion Pipeline */}
      {activeTab === 'upload' && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6">
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white">Upload Your Own CSV Dataset</h3>
            <p className="text-xs text-slate-400">
              Drag-and-drop or select any CSV file. The pipeline parses headers, detects data types, audits missing values, and checks for data leakage.
            </p>
          </div>

          {/* File Picker & Drop Zone */}
          <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-8 text-center transition-colors flex flex-col items-center justify-center space-y-3 cursor-pointer relative bg-slate-950/50">
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            <div className="p-3 bg-indigo-500/10 rounded-full text-indigo-400">
              <Upload className="w-6 h-6" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white">Click to select or drop CSV file here</div>
              <div className="text-xs text-slate-500 mt-0.5">Maximum file size: 5MB (comma, semicolon, or tab delimited)</div>
            </div>
          </div>

          {/* Error & Success Messages */}
          {uploadErrors.length > 0 && (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs text-rose-300 space-y-1 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold">Dataset Ingestion Error</div>
                {uploadErrors.map((err, i) => (
                  <div key={i}>{err}</div>
                ))}
              </div>
            </div>
          )}

          {uploadSuccess && (
            <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800 text-xs text-emerald-300 flex items-center gap-2.5">
              <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{uploadSuccess}</span>
            </div>
          )}

          {/* Manual CSV Text Area */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300">
              Or paste raw CSV text directly:
            </label>
            <textarea
              rows={4}
              value={uploadText}
              onChange={(e) => {
                setUploadText(e.target.value);
                processUploadedCSV(e.target.value, 'Custom Dataset');
              }}
              placeholder="feature1,feature2,feature3,target&#10;1.2,3.4,5.6,ClassA&#10;2.3,4.5,6.7,ClassB"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* Configuration Form if CSV is parsed */}
          {uploadSuccess && (
            <div className="pt-4 border-t border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Dataset Name</label>
                <input
                  type="text"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Task Type</label>
                <select
                  value={customTask}
                  onChange={(e) => setCustomTask(e.target.value as any)}
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200"
                >
                  <option value="classification">Classification (Categories)</option>
                  <option value="regression">Regression (Continuous Numbers)</option>
                  <option value="clustering">Clustering (Unsupervised)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Target Column</label>
                <input
                  type="text"
                  value={customTarget}
                  onChange={(e) => setCustomTarget(e.target.value)}
                  placeholder="e.g. target or label"
                  className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200"
                />
              </div>

              <div className="md:col-span-3 flex justify-end">
                <button
                  onClick={finalizeCustomDataset}
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow-sm transition-colors"
                >
                  Confirm & Ingest Dataset
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
