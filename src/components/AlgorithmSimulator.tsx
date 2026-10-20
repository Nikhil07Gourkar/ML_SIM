import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  SkipBack,
  Save,
  CheckCircle,
  HelpCircle,
  Sparkles,
  Layers,
  Sliders,
  ChevronRight,
  TrendingUp,
  Cpu,
  BarChart,
  Binary,
  GitBranch,
  Target,
  ArrowRight,
  ArrowLeft,
  Database,
  Table,
  CheckSquare,
  Square,
  Activity,
  Calculator,
  PieChart,
  Zap,
} from 'lucide-react';
import {
  AlgorithmId,
  DatasetInfo,
  Experiment,
  PreparedData,
  PreprocessingConfig,
  SimulationStep,
  TreeNode,
} from '../types/ml';
import { BUILT_IN_DATASETS } from '../ml/datasets';
import { preprocessDataset } from '../ml/preprocessing';
import { executeModelTraining, TrainResult } from '../ml/engine';

interface AlgorithmSimulatorProps {
  initialAlgorithm?: AlgorithmId;
  initialDataset?: DatasetInfo;
  onSaveExperiment: (exp: Experiment) => void;
  onAskTutor: (context: any) => void;
}

export const AlgorithmSimulator: React.FC<AlgorithmSimulatorProps> = ({
  initialAlgorithm = 'linear_regression',
  initialDataset = BUILT_IN_DATASETS[1], // Study hours by default
  onSaveExperiment,
  onAskTutor,
}) => {
  // 1. Workflow Step (1 to 10)
  const [activeStepTab, setActiveStepTab] = useState<number>(7); // Start on simulation if preconfigured

  // 2. Selection states
  const [selectedDataset, setSelectedDataset] = useState<DatasetInfo>(initialDataset);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmId>(initialAlgorithm);

  // 3. Preprocessing Config
  const [preprocessingConfig, setPreprocessingConfig] = useState<PreprocessingConfig>({
    missingStrategy: 'impute_mean',
    removeDuplicates: true,
    scaleMethod: 'none', // none or standard
    encodeCategorical: 'one_hot',
    testSplitRatio: 0.2,
    randomSeed: 42,
    selectedFeatures: initialDataset.features.filter((f) => f.name !== initialDataset.targetColumn).map((f) => f.name),
    targetColumn: initialDataset.targetColumn,
  });

  // 4. Algorithm Hyperparameters
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({
    fitIntercept: true,
    learningRate: 0.05,
    maxIterations: 100,
    maxDepth: 3,
    minSamplesSplit: 2,
    criterion: 'gini',
    k: 3,
    numTrees: 7,
    C: 1.0,
    threshold: 0.5,
    numComponents: 2,
  });

  // 5. Interactive Demo Override for Linear Regression (Manual slope/intercept mode)
  const [manualMode, setManualMode] = useState(false);
  const [manualSlope, setManualSlope] = useState(1.0);
  const [manualIntercept, setManualIntercept] = useState(10.0);

  // 6. Interactive Sample Prediction inputs
  const [userInputs, setUserInputs] = useState<Record<string, number>>({});
  const [livePrediction, setLivePrediction] = useState<string | number | null>(null);

  // 7. Prepared Data & Train Results
  const [preparedData, setPreparedData] = useState<PreparedData | null>(null);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [prepError, setPrepError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  // 8. Simulation Playback State
  const [currentStepIdx, setCurrentStepIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const timerRef = useRef<any>(null);

  // Sync state when props change
  useEffect(() => {
    if (initialAlgorithm) setSelectedAlgorithm(initialAlgorithm);
  }, [initialAlgorithm]);

  useEffect(() => {
    if (initialDataset) {
      setSelectedDataset(initialDataset);
      setPreprocessingConfig((prev) => ({
        ...prev,
        selectedFeatures: initialDataset.features
          .filter((f) => f.name !== initialDataset.targetColumn)
          .map((f) => f.name),
        targetColumn: initialDataset.targetColumn,
      }));
    }
  }, [initialDataset]);

  // Run preprocessing and training
  const runModelTraining = () => {
    try {
      setPrepError(null);
      const prepared = preprocessDataset(
        selectedDataset.data,
        preprocessingConfig,
        selectedDataset.taskType
      );
      setPreparedData(prepared);

      const result = executeModelTraining(
        selectedAlgorithm,
        prepared,
        hyperparameters,
        selectedDataset.taskType
      );
      setTrainResult(result);
      setCurrentStepIdx(0);
      setIsPlaying(false);

      // Prepopulate manual mode with OLS result
      if (result.modelDetails.slope !== undefined) {
        setManualSlope(Number(result.modelDetails.slope.toFixed(2)));
        setManualIntercept(Number(result.modelDetails.intercept.toFixed(2)));
      }

      // Prepopulate user prediction inputs with means
      const initialInputs: Record<string, number> = {};
      prepared.featureNames.forEach((feat, idx) => {
        const meanVal = prepared.X_train.reduce((acc, row) => acc + (row[idx] || 0), 0) / (prepared.X_train.length || 1);
        initialInputs[feat] = +meanVal.toFixed(2);
      });
      setUserInputs(initialInputs);
      setLivePrediction(result.predict(Object.values(initialInputs)));
    } catch (err: any) {
      setPrepError(err?.message || 'Error occurred during preprocessing or training.');
    }
  };

  // Re-train when parameters or datasets change
  useEffect(() => {
    runModelTraining();
  }, [
    selectedDataset.id,
    selectedAlgorithm,
    preprocessingConfig.scaleMethod,
    preprocessingConfig.missingStrategy,
    preprocessingConfig.selectedFeatures.length,
    hyperparameters.k,
    hyperparameters.maxDepth,
    hyperparameters.threshold,
    hyperparameters.fitIntercept,
    hyperparameters.criterion,
    hyperparameters.numTrees,
    hyperparameters.C,
  ]);

  // Step Playback Loop
  useEffect(() => {
    if (isPlaying && trainResult && trainResult.simulationSteps.length > 0) {
      const intervalMs = Math.max(250, 1500 / playbackSpeed);
      timerRef.current = setInterval(() => {
        setCurrentStepIdx((prev) => {
          if (prev >= trainResult.simulationSteps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, trainResult, playbackSpeed]);

  const currentStep: SimulationStep | undefined =
    trainResult?.simulationSteps[currentStepIdx] || trainResult?.simulationSteps[0];

  // Handle User Input Change for Real-Time Prediction
  const handleInputChange = (feat: string, val: number) => {
    const updated = { ...userInputs, [feat]: val };
    setUserInputs(updated);
    if (trainResult && preparedData) {
      const vec = preparedData.featureNames.map((f) => updated[f] || 0);
      setLivePrediction(trainResult.predict(vec));
    }
  };

  // Save Experiment Handler
  const handleSave = () => {
    if (!trainResult || !preparedData) return;
    const exp: Experiment = {
      id: `exp_${Date.now()}`,
      name: `${selectedDataset.name} - ${selectedAlgorithm}`,
      createdAt: new Date().toISOString(),
      datasetId: selectedDataset.id,
      datasetName: selectedDataset.name,
      algorithmId: selectedAlgorithm,
      algorithmName: selectedAlgorithm.replace('_', ' ').toUpperCase(),
      hyperparameters: { ...hyperparameters },
      preprocessing: { ...preprocessingConfig },
      metrics: { ...trainResult.metrics },
      status: 'completed',
    };
    onSaveExperiment(exp);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2500);
  };

  // Toggle Feature Selection
  const toggleFeature = (featName: string) => {
    const current = preprocessingConfig.selectedFeatures;
    if (current.includes(featName)) {
      if (current.length <= 1) return; // Prevent 0 features
      setPreprocessingConfig({
        ...preprocessingConfig,
        selectedFeatures: current.filter((f) => f !== featName),
      });
    } else {
      setPreprocessingConfig({
        ...preprocessingConfig,
        selectedFeatures: [...current, featName],
      });
    }
  };

  const workflowSteps = [
    { num: 1, label: 'Dataset', desc: 'Choose training data' },
    { num: 2, label: 'Algorithm', desc: 'Choose ML model' },
    { num: 3, label: 'Features', desc: 'Select inputs & target' },
    { num: 4, label: 'Preprocessing', desc: 'Scaling & imputation' },
    { num: 5, label: 'Hyperparameters', desc: 'Model parameters' },
    { num: 6, label: 'Data Preview', desc: 'Inspect transformed matrix' },
    { num: 7, label: 'Simulation', desc: 'Step-by-step visual canvas' },
    { num: 8, label: 'Math Steps', desc: 'Formulas & trace log' },
    { num: 9, label: 'Evaluation', desc: 'Metrics & confusion matrix' },
    { num: 10, label: 'Prediction', desc: 'Real-time test bench' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6 animate-fadeIn">
      {/* 10-Step Interactive Workflow Ribbon Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-3 shadow-md">
        <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80 text-xs">
          <span className="font-semibold text-slate-300 flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-indigo-400" />
            <span>Workflow Stages (Click any step to view)</span>
          </span>
          <span className="text-slate-500 font-mono">
            Active: Step {activeStepTab} of 10
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-1.5">
          {workflowSteps.map((st) => (
            <button
              key={st.num}
              onClick={() => setActiveStepTab(st.num)}
              className={`flex flex-col items-center justify-center py-2 px-1 rounded-xl text-center transition-all ${
                activeStepTab === st.num
                  ? 'bg-indigo-600 text-white font-semibold shadow-md shadow-indigo-600/30 ring-1 ring-indigo-400'
                  : 'bg-slate-950/50 hover:bg-slate-800/80 text-slate-400 hover:text-slate-200 border border-slate-800/60'
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] mb-1 font-bold ${
                  activeStepTab === st.num
                    ? 'bg-white text-indigo-700'
                    : 'bg-slate-800 text-slate-400'
                }`}
              >
                {st.num}
              </span>
              <span className="text-[11px] truncate w-full px-1">{st.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Top Header Strip with Action Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400">
            <Cpu className="w-4 h-4" />
            <span className="capitalize">{selectedAlgorithm.replace(/_/g, ' ')}</span>
            <span className="text-slate-600">·</span>
            <span>{selectedDataset.name}</span>
            <span className="text-slate-600">·</span>
            <span className="capitalize">{selectedDataset.taskType}</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-0.5">
            {workflowSteps[activeStepTab - 1].label}: {workflowSteps[activeStepTab - 1].desc}
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() =>
              onAskTutor({
                algorithmId: selectedAlgorithm,
                datasetName: selectedDataset.name,
                taskType: selectedDataset.taskType,
                metrics: trainResult?.metrics,
                hyperparameters,
              })
            }
            className="px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
            <span>Ask Tutor About Step {activeStepTab}</span>
          </button>

          <button
            onClick={handleSave}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm ${
              saveSuccess
                ? 'bg-emerald-600 text-white'
                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30'
            }`}
          >
            {saveSuccess ? <CheckCircle className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
            <span>{saveSuccess ? 'Saved!' : 'Save Experiment'}</span>
          </button>
        </div>
      </div>

      {prepError && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-xs text-rose-300">
          {prepError}
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 1: CHOOSE DATASET                                                    */}
      {/* ========================================================================= */}
      {activeStepTab === 1 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 1: Choose Dataset</h2>
            <p className="text-xs text-slate-400">
              Select one of the 10 built-in benchmark datasets or an uploaded dataset for simulation.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {BUILT_IN_DATASETS.map((ds) => (
              <button
                key={ds.id}
                onClick={() => {
                  setSelectedDataset(ds);
                  setPreprocessingConfig((p) => ({
                    ...p,
                    selectedFeatures: ds.features.filter((f) => f.name !== ds.targetColumn).map((f) => f.name),
                    targetColumn: ds.targetColumn,
                  }));
                }}
                className={`p-4 rounded-xl border text-left transition-all space-y-2 ${
                  selectedDataset.id === ds.id
                    ? 'bg-indigo-950/70 border-indigo-500 text-white shadow-md shadow-indigo-500/20 ring-1 ring-indigo-500'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-sm">{ds.name}</span>
                  <span className="capitalize px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-mono">
                    {ds.taskType}
                  </span>
                </div>
                <p className="text-xs text-slate-400 line-clamp-2">{ds.description}</p>
                <div className="text-[11px] text-slate-500 flex items-center gap-2 pt-1">
                  <span>{ds.rows} rows</span>
                  <span>·</span>
                  <span>{ds.columns} features</span>
                  <span>·</span>
                  <span>{ds.difficulty}</span>
                </div>
              </button>
            ))}
          </div>

          <div className="flex justify-end pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(2)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <span>Next: Select Algorithm</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 2: SELECT ALGORITHM                                                  */}
      {/* ========================================================================= */}
      {activeStepTab === 2 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 2: Select Machine Learning Algorithm</h2>
            <p className="text-xs text-slate-400">
              Pick an estimator to train. Recommended algorithms for {selectedDataset.name}: {selectedDataset.recommendedAlgorithms.join(', ')}.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { id: 'linear_regression', name: 'Linear Regression', task: 'Regression', desc: 'Fit best-fit line via Ordinary Least Squares & gradient descent' },
              { id: 'logistic_regression', name: 'Logistic Regression', task: 'Classification', desc: 'Predict discrete classes via Sigmoid activation & log-loss' },
              { id: 'decision_tree', name: 'Decision Tree', task: 'Classification/Regression', desc: 'Recursive orthogonal splits maximizing Gini/Entropy Information Gain' },
              { id: 'random_forest', name: 'Random Forest', task: 'Classification/Regression', desc: 'Ensemble bagging of randomized decision trees with committee voting' },
              { id: 'kmeans', name: 'K-Means Clustering', task: 'Clustering', desc: 'Iterative cluster assignment and centroid recalculation via Lloyd\'s algorithm' },
              { id: 'knn', name: 'K-Nearest Neighbors', task: 'Classification/Regression', desc: 'Instance-based classification voting among K closest neighbors' },
              { id: 'svm', name: 'Support Vector Machine', task: 'Classification', desc: 'Max-margin separating hyperplane with soft margin hinge loss' },
              { id: 'pca', name: 'Principal Component Analysis', task: 'Dimensionality', desc: 'Orthogonal projection to directions of maximal covariance variance' },
            ].map((algo) => (
              <button
                key={algo.id}
                onClick={() => setSelectedAlgorithm(algo.id as AlgorithmId)}
                className={`p-4 rounded-xl border text-left transition-all space-y-2 ${
                  selectedAlgorithm === algo.id
                    ? 'bg-indigo-950/70 border-indigo-500 text-white shadow-md shadow-indigo-500/20 ring-1 ring-indigo-500'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-sm">{algo.name}</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-indigo-400 font-mono">
                    {algo.task}
                  </span>
                </div>
                <p className="text-xs text-slate-400 line-clamp-3">{algo.desc}</p>
              </button>
            ))}
          </div>

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(1)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Dataset</span>
            </button>
            <button
              onClick={() => setActiveStepTab(3)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Select Features</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 3: FEATURES & TARGET SELECTION                                       */}
      {/* ========================================================================= */}
      {activeStepTab === 3 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 3: Select Features & Target Column</h2>
            <p className="text-xs text-slate-400">
              Choose which columns to include as model inputs (X) and designate the target variable (y).
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Features checkboxes */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-white uppercase tracking-wider block">
                Input Features (X)
              </label>
              <div className="space-y-2 max-h-72 overflow-y-auto p-1">
                {selectedDataset.features
                  .filter((f) => f.name !== selectedDataset.targetColumn)
                  .map((feat) => {
                    const isSelected = preprocessingConfig.selectedFeatures.includes(feat.name);
                    return (
                      <div
                        key={feat.name}
                        onClick={() => toggleFeature(feat.name)}
                        className={`p-3 rounded-xl border flex items-center justify-between cursor-pointer transition-all ${
                          isSelected
                            ? 'bg-slate-950 border-indigo-500/70 text-white'
                            : 'bg-slate-950/40 border-slate-800 text-slate-500'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          {isSelected ? (
                            <CheckSquare className="w-4 h-4 text-indigo-400" />
                          ) : (
                            <Square className="w-4 h-4 text-slate-600" />
                          )}
                          <span className="text-xs font-semibold">{feat.name}</span>
                        </div>
                        <span className="text-[10px] text-slate-500 capitalize">{feat.type}</span>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Target column selector */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-white uppercase tracking-wider block">
                Target Column (y)
              </label>
              {selectedDataset.taskType === 'clustering' ? (
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-2">
                  <div className="font-semibold text-amber-400">Unsupervised Task</div>
                  <p>
                    Clustering does not use a ground-truth target column during training. Centroids are derived entirely from input feature geometry.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <select
                    value={preprocessingConfig.targetColumn || selectedDataset.targetColumn}
                    onChange={(e) =>
                      setPreprocessingConfig({ ...preprocessingConfig, targetColumn: e.target.value })
                    }
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-indigo-500"
                  >
                    {selectedDataset.features.map((feat) => (
                      <option key={feat.name} value={feat.name}>
                        {feat.name} ({feat.type})
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-400">
                    Currently predicting: <strong className="text-indigo-400">{preprocessingConfig.targetColumn}</strong>
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(2)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Algorithm</span>
            </button>
            <button
              onClick={() => setActiveStepTab(4)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Preprocessing</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 4: PREPROCESSING CONFIG                                              */}
      {/* ========================================================================= */}
      {activeStepTab === 4 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 4: Configure Data Preprocessing</h2>
            <p className="text-xs text-slate-400">
              Clean and transform data without data leakage. Transformations are strictly fitted on the training split only.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <label className="font-bold text-white block">Feature Scaling</label>
              <select
                value={preprocessingConfig.scaleMethod}
                onChange={(e) =>
                  setPreprocessingConfig({ ...preprocessingConfig, scaleMethod: e.target.value as any })
                }
                className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200"
              >
                <option value="none">None (Raw Values)</option>
                <option value="standard">StandardScaler (z = (x - μ) / σ)</option>
                <option value="minmax">MinMaxScaler ([0, 1])</option>
              </select>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Vital for distance-based models (KNN, SVM, K-Means). Trees are scale-invariant.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <label className="font-bold text-white block">Missing Value Strategy</label>
              <select
                value={preprocessingConfig.missingStrategy}
                onChange={(e) =>
                  setPreprocessingConfig({ ...preprocessingConfig, missingStrategy: e.target.value as any })
                }
                className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200"
              >
                <option value="impute_mean">Impute Mean</option>
                <option value="impute_median">Impute Median</option>
                <option value="drop">Drop Rows with Missing</option>
              </select>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Calculates replacement values using only training partition statistics.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <label className="font-bold text-white block">Train/Test Split: {(1 - preprocessingConfig.testSplitRatio) * 100}% / {preprocessingConfig.testSplitRatio * 100}%</label>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={preprocessingConfig.testSplitRatio}
                onChange={(e) =>
                  setPreprocessingConfig({ ...preprocessingConfig, testSplitRatio: Number(e.target.value) })
                }
                className="w-full accent-indigo-500"
              />
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Reserve 10-40% of records strictly for out-of-sample evaluation.
              </p>
            </div>
          </div>

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(3)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Features</span>
            </button>
            <button
              onClick={() => setActiveStepTab(5)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Hyperparameters</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 5: HYPERPARAMETERS                                                   */}
      {/* ========================================================================= */}
      {activeStepTab === 5 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 5: Configure Model Hyperparameters</h2>
            <p className="text-xs text-slate-400">
              Adjust algorithm-specific controls to observe their direct impact on model complexity, bias, and variance.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
            {selectedAlgorithm === 'kmeans' && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <label className="font-bold text-white block">Number of Clusters (K): {hyperparameters.k}</label>
                <input
                  type="range"
                  min="2"
                  max="6"
                  value={hyperparameters.k}
                  onChange={(e) => setHyperparameters({ ...hyperparameters, k: Number(e.target.value) })}
                  className="w-full accent-indigo-500"
                />
                <p className="text-[11px] text-slate-400">How many cluster centroids the algorithm will initialize.</p>
              </div>
            )}

            {selectedAlgorithm === 'decision_tree' && (
              <>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <label className="font-bold text-white block">Maximum Tree Depth: {hyperparameters.maxDepth}</label>
                  <input
                    type="range"
                    min="1"
                    max="6"
                    value={hyperparameters.maxDepth}
                    onChange={(e) => setHyperparameters({ ...hyperparameters, maxDepth: Number(e.target.value) })}
                    className="w-full accent-indigo-500"
                  />
                  <p className="text-[11px] text-slate-400">Controls tree complexity. Lower depths prevent overfitting.</p>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <label className="font-bold text-white block">Split Criterion</label>
                  <select
                    value={hyperparameters.criterion}
                    onChange={(e) => setHyperparameters({ ...hyperparameters, criterion: e.target.value })}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200"
                  >
                    <option value="gini">Gini Impurity (1 - ∑ p_i²)</option>
                    <option value="entropy">Entropy (-∑ p_i log₂ p_i)</option>
                  </select>
                </div>
              </>
            )}

            {selectedAlgorithm === 'logistic_regression' && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <label className="font-bold text-white block">Decision Threshold: {hyperparameters.threshold}</label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={hyperparameters.threshold}
                  onChange={(e) => setHyperparameters({ ...hyperparameters, threshold: Number(e.target.value) })}
                  className="w-full accent-indigo-500"
                />
                <p className="text-[11px] text-slate-400">Adjusts the boundary between positive and negative predictions.</p>
              </div>
            )}

            {selectedAlgorithm === 'linear_regression' && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <label className="font-bold text-white block">Fit Intercept (b)</label>
                <input
                  type="checkbox"
                  checked={hyperparameters.fitIntercept}
                  onChange={(e) => setHyperparameters({ ...hyperparameters, fitIntercept: e.target.checked })}
                  className="accent-indigo-500 mr-2"
                />
                <span className="text-slate-300">Allow non-zero intercept in y = mx + b</span>
              </div>
            )}

            {selectedAlgorithm === 'random_forest' && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <label className="font-bold text-white block">Number of Trees: {hyperparameters.numTrees}</label>
                <input
                  type="range"
                  min="3"
                  max="15"
                  value={hyperparameters.numTrees}
                  onChange={(e) => setHyperparameters({ ...hyperparameters, numTrees: Number(e.target.value) })}
                  className="w-full accent-indigo-500"
                />
              </div>
            )}

            {selectedAlgorithm === 'svm' && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <label className="font-bold text-white block">Regularization C: {hyperparameters.C}</label>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={hyperparameters.C}
                  onChange={(e) => setHyperparameters({ ...hyperparameters, C: Number(e.target.value) })}
                  className="w-full accent-indigo-500"
                />
              </div>
            )}
          </div>

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(4)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Preprocessing</span>
            </button>
            <button
              onClick={() => setActiveStepTab(6)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Preview Data</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 6: PREPARED DATA PREVIEW                                             */}
      {/* ========================================================================= */}
      {activeStepTab === 6 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white">Step 6: Preview Prepared Data</h2>
            <p className="text-xs text-slate-400">
              Inspect the exact numeric matrix ready to feed into the training algorithms.
            </p>
          </div>

          {preparedData && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Train Samples</div>
                  <div className="text-white font-bold text-sm">{preparedData.X_train.length}</div>
                </div>
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Test Samples</div>
                  <div className="text-white font-bold text-sm">{preparedData.X_test.length}</div>
                </div>
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Feature Count</div>
                  <div className="text-white font-bold text-sm">{preparedData.featureNames.length}</div>
                </div>
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
                  <div className="text-slate-500 text-[10px]">Scaling Applied</div>
                  <div className="text-indigo-400 font-bold text-sm capitalize">{preprocessingConfig.scaleMethod}</div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 overflow-x-auto bg-slate-950 p-2 font-mono text-[11px]">
                <table className="w-full text-left text-slate-300">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-800 text-[10px] uppercase">
                      <th className="p-2">#</th>
                      {preparedData.featureNames.map((f) => (
                        <th key={f} className="p-2">{f}</th>
                      ))}
                      {preparedData.targetName && <th className="p-2 text-indigo-400">Target (y)</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {preparedData.X_train.slice(0, 8).map((row, rIdx) => (
                      <tr key={rIdx} className="border-b border-slate-900 hover:bg-slate-900/50">
                        <td className="p-2 text-slate-600">{rIdx + 1}</td>
                        {row.map((val, cIdx) => (
                          <td key={cIdx} className="p-2">{val.toFixed(2)}</td>
                        ))}
                        {preparedData.targetName && (
                          <td className="p-2 text-indigo-300 font-bold">
                            {preparedData.y_train[rIdx]?.toFixed(1)}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(5)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Hyperparameters</span>
            </button>
            <button
              onClick={() => setActiveStepTab(7)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Launch Simulation (Step 7)</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 7: INTERACTIVE SIMULATION STAGE                                      */}
      {/* ========================================================================= */}
      {activeStepTab === 7 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fadeIn">
          {/* Left Zone: Visual Stage (2 cols) */}
          <div className="lg:col-span-2 rounded-2xl bg-slate-900 border border-slate-800 p-6 flex flex-col justify-between space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
                <h2 className="text-sm font-bold text-white tracking-wide">
                  Step 7: Visual Algorithm Stage
                </h2>
              </div>

              {selectedAlgorithm === 'linear_regression' && (
                <button
                  onClick={() => setManualMode(!manualMode)}
                  className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                    manualMode
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                      : 'bg-slate-800 text-slate-400 border-slate-700'
                  }`}
                >
                  {manualMode ? 'Manual Slope Mode' : 'OLS Mode'}
                </button>
              )}
            </div>

            {/* SVG Visual Stage Canvas */}
            <div className="w-full h-80 bg-slate-950 rounded-xl border border-slate-800/80 p-4 relative overflow-hidden flex items-center justify-center">
              {/* Linear Regression */}
              {selectedAlgorithm === 'linear_regression' && preparedData && (
                <svg className="w-full h-full overflow-visible">
                  <line x1="5%" y1="92%" x2="95%" y2="92%" stroke="#334155" strokeWidth="1" />
                  <line x1="5%" y1="5%" x2="5%" y2="92%" stroke="#334155" strokeWidth="1" />

                  {preparedData.X_train.slice(0, 45).map((xRow, i) => {
                    const xVal = xRow[0] || 0;
                    const yVal = preparedData.y_train[i] || 0;
                    const cx = 8 + (xVal / 20) * 80 + '%';
                    const cy = 90 - (yVal / 100) * 80 + '%';

                    const slope = manualMode ? manualSlope : (currentStep?.visualState?.regressionLine?.slope ?? trainResult?.modelDetails.slope ?? 3.5);
                    const intercept = manualMode ? manualIntercept : (currentStep?.visualState?.regressionLine?.intercept ?? trainResult?.modelDetails.intercept ?? 12);
                    const predY = slope * xVal + intercept;
                    const lineCy = 90 - (predY / 100) * 80 + '%';

                    return (
                      <g key={i}>
                        <line x1={cx} y1={cy} x2={cx} y2={lineCy} stroke="#f43f5e" strokeWidth="1" strokeDasharray="2,2" opacity="0.6" />
                        <circle cx={cx} cy={cy} r="4" fill="#6366f1" opacity="0.8" />
                      </g>
                    );
                  })}

                  {(() => {
                    const slope = manualMode ? manualSlope : (currentStep?.visualState?.regressionLine?.slope ?? trainResult?.modelDetails.slope ?? 3.5);
                    const intercept = manualMode ? manualIntercept : (currentStep?.visualState?.regressionLine?.intercept ?? trainResult?.modelDetails.intercept ?? 12);
                    const yStart = intercept;
                    const yEnd = slope * 20 + intercept;
                    return (
                      <line
                        x1="8%"
                        y1={`${90 - (yStart / 100) * 80}%`}
                        x2="88%"
                        y2={`${90 - (yEnd / 100) * 80}%`}
                        stroke={manualMode ? '#f59e0b' : '#10b981'}
                        strokeWidth="3"
                        className="transition-all duration-300"
                      />
                    );
                  })()}
                </svg>
              )}

              {/* K-Means */}
              {selectedAlgorithm === 'kmeans' && preparedData && (
                <svg className="w-full h-full overflow-visible">
                  <line x1="5%" y1="92%" x2="95%" y2="92%" stroke="#334155" strokeWidth="1" />
                  <line x1="5%" y1="5%" x2="5%" y2="92%" stroke="#334155" strokeWidth="1" />

                  {preparedData.X_train.slice(0, 60).map((pt, i) => {
                    const clusterId = currentStep?.visualState.assignments?.[i] ?? 0;
                    const colors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#06b6d4'];
                    const color = colors[clusterId % colors.length];
                    const cx = 10 + (Math.abs(pt[0]) / 12) * 80 + '%';
                    const cy = 90 - (Math.abs(pt[1] || pt[0]) / 12) * 80 + '%';
                    return <circle key={i} cx={cx} cy={cy} r="4" fill={color} opacity="0.75" />;
                  })}

                  {(currentStep?.visualState.centroids || trainResult?.modelDetails.centroids || []).map((c: any, cIdx: number) => {
                    const colors = ['#818cf8', '#34d399', '#fbbf24', '#f472b6', '#22d3ee'];
                    const color = colors[cIdx % colors.length];
                    const cx = 10 + (Math.abs(c.x || c[0] || 0) / 12) * 80 + '%';
                    const cy = 90 - (Math.abs(c.y || c[1] || 0) / 12) * 80 + '%';
                    return (
                      <g key={cIdx} className="transition-all duration-500">
                        <circle cx={cx} cy={cy} r="14" fill={color} opacity="0.25" />
                        <circle cx={cx} cy={cy} r="7" fill={color} stroke="#ffffff" strokeWidth="2" />
                        <text x={cx} y={cy} textAnchor="middle" dy="3" fontSize="9" fill="#000" fontWeight="bold">
                          {cIdx + 1}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}

              {/* Decision Tree */}
              {selectedAlgorithm === 'decision_tree' && trainResult?.modelDetails.treeRoot && (
                <div className="w-full h-full flex flex-col items-center justify-center p-2 text-xs">
                  {(() => {
                    const root = trainResult.modelDetails.treeRoot as TreeNode;
                    return (
                      <div className="flex flex-col items-center space-y-4 w-full">
                        <div className="p-2.5 rounded-lg bg-indigo-950/80 border border-indigo-500 text-center text-indigo-200">
                          <div className="font-bold">{root.featureName} ≤ {root.threshold}</div>
                          <div className="text-[10px] text-slate-400">Gini: {root.impurity} · Samples: {root.samples}</div>
                        </div>
                        <div className="grid grid-cols-2 gap-8 w-full max-w-md">
                          <div className="p-2 rounded-lg bg-emerald-950/70 border border-emerald-600 text-center text-emerald-200">
                            <div className="text-[11px] font-bold">Left (True)</div>
                            <div className="text-[10px] text-slate-300">
                              {root.left?.isLeaf ? `Leaf: ${root.left.predictedClass}` : `${root.left?.featureName} ≤ ${root.left?.threshold}`}
                            </div>
                          </div>
                          <div className="p-2 rounded-lg bg-rose-950/70 border border-rose-600 text-center text-rose-200">
                            <div className="text-[11px] font-bold">Right (False)</div>
                            <div className="text-[10px] text-slate-300">
                              {root.right?.isLeaf ? `Leaf: ${root.right.predictedClass}` : `${root.right?.featureName} ≤ ${root.right?.threshold}`}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* Logistic Regression */}
              {selectedAlgorithm === 'logistic_regression' && (
                <svg className="w-full h-full overflow-visible">
                  <line x1="5%" y1="92%" x2="95%" y2="92%" stroke="#334155" strokeWidth="1" />
                  <line x1="5%" y1="8%" x2="95%" y2="8%" stroke="#334155" strokeWidth="1" strokeDasharray="3,3" />
                  <line x1="50%" y1="5%" x2="50%" y2="92%" stroke="#334155" strokeWidth="1" />
                  <line
                    x1="5%"
                    y1={`${92 - (hyperparameters.threshold || 0.5) * 84}%`}
                    x2="95%"
                    y2={`${92 - (hyperparameters.threshold || 0.5) * 84}%`}
                    stroke="#f59e0b"
                    strokeWidth="1.5"
                    strokeDasharray="4,4"
                  />
                  <path d="M 50,260 C 200,260 250,30 450,30" fill="none" stroke="#6366f1" strokeWidth="3" />
                </svg>
              )}

              {/* Default PCA / SVM / KNN */}
              {['svm', 'knn', 'random_forest', 'pca'].includes(selectedAlgorithm) && (
                <div className="text-center space-y-2">
                  <Target className="w-10 h-10 text-indigo-400 mx-auto opacity-70" />
                  <div className="text-xs font-semibold text-slate-300">
                    {selectedAlgorithm.toUpperCase().replace('_', ' ')} Execution Space
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Trained across {preparedData?.X_train.length} samples with {preparedData?.featureNames.length} features.
                  </div>
                </div>
              )}
            </div>

            {/* Playback Transport Bar */}
            <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentStepIdx(0);
                  }}
                  className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  disabled={currentStepIdx <= 0}
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentStepIdx((p) => Math.max(0, p - 1));
                  }}
                  className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-300 border border-slate-800"
                >
                  <SkipBack className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs flex items-center gap-1.5"
                >
                  {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
                  <span>{isPlaying ? 'Pause' : 'Play'}</span>
                </button>
                <button
                  disabled={!trainResult || currentStepIdx >= trainResult.simulationSteps.length - 1}
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentStepIdx((p) => p + 1);
                  }}
                  className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-slate-300 border border-slate-800"
                >
                  <SkipForward className="w-4 h-4" />
                </button>
                <button
                  onClick={() => {
                    if (trainResult) {
                      setIsPlaying(false);
                      setCurrentStepIdx(trainResult.simulationSteps.length - 1);
                    }
                  }}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs font-medium"
                >
                  Run All
                </button>
              </div>

              <div className="flex items-center gap-3 text-xs text-slate-400">
                <span className="font-mono tabular-nums">
                  Step {currentStepIdx + 1} of {trainResult?.simulationSteps.length || 1}
                </span>
                <div className="flex items-center gap-1 bg-slate-900 px-2 py-1 rounded border border-slate-800">
                  <span className="text-[11px] text-slate-500">Speed:</span>
                  {[0.5, 1, 2].map((spd) => (
                    <button
                      key={spd}
                      onClick={() => setPlaybackSpeed(spd)}
                      className={`px-1.5 py-0.5 rounded text-[11px] font-medium ${
                        playbackSpeed === spd ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right Zone: Math Explanation & Step Inspector Deck */}
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                  Live Step Calculation
                </div>
                <span className="text-[11px] text-slate-500 font-mono">
                  {currentStep?.operation || 'Step Operation'}
                </span>
              </div>

              <div className="space-y-2">
                <h3 className="text-base font-bold text-white">
                  {currentStep?.title || 'Execution Step'}
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {currentStep?.description}
                </p>
              </div>

              {currentStep?.formula && (
                <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                  <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Mathematical Formula
                  </div>
                  <div className="font-mono text-xs text-amber-300 overflow-x-auto py-1">
                    {currentStep.formula}
                  </div>
                </div>
              )}

              {currentStep?.intermediateValues && (
                <div className="space-y-1.5">
                  <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                    Current Internal State
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    {Object.entries(currentStep.intermediateValues).map(([k, v]) => (
                      <div key={k} className="p-2 rounded bg-slate-950 border border-slate-800">
                        <div className="text-[10px] text-slate-400 font-sans">{k}</div>
                        <div className="text-slate-200 font-bold truncate">
                          {typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
              <button
                onClick={() => setActiveStepTab(8)}
                className="w-full py-2 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 border border-indigo-500/40 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
              >
                <span>View Full Math Steps (Step 8)</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 8: DEDICATED FULL MATH STEPS & TIMELINE TRACE                        */}
      {/* ========================================================================= */}
      {activeStepTab === 8 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Calculator className="w-5 h-5 text-indigo-400" />
                <span>Step 8: Mathematical Steps & Derivations</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Full chronological sequence of algorithmic operations, matrices, and calculations.
              </p>
            </div>

            <button
              onClick={() => setActiveStepTab(7)}
              className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-lg font-medium flex items-center gap-1.5 transition-colors"
            >
              <Activity className="w-3.5 h-3.5 text-indigo-400" />
              <span>Back to Visual Stage</span>
            </button>
          </div>

          {trainResult && trainResult.simulationSteps.length > 0 ? (
            <div className="space-y-4">
              {trainResult.simulationSteps.map((step, idx) => (
                <div
                  key={idx}
                  onClick={() => setCurrentStepIdx(idx)}
                  className={`p-4 rounded-xl border transition-all cursor-pointer space-y-3 ${
                    currentStepIdx === idx
                      ? 'bg-slate-950 border-indigo-500 shadow-md ring-1 ring-indigo-500/50'
                      : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="w-6 h-6 rounded-full bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 text-xs font-bold flex items-center justify-center">
                        {step.stepNumber}
                      </span>
                      <h3 className="text-sm font-bold text-white">{step.title}</h3>
                    </div>
                    <span className="px-2 py-0.5 rounded text-[10px] bg-slate-900 border border-slate-800 text-slate-400 font-mono">
                      {step.operation}
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed pl-8">
                    {step.description}
                  </p>

                  {step.formula && (
                    <div className="ml-8 p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-amber-300">
                      <div className="text-[10px] uppercase font-bold text-slate-500 mb-1">Formula Applied</div>
                      {step.formula}
                    </div>
                  )}

                  {step.mathExplanation && (
                    <div className="ml-8 text-xs text-slate-400 italic">
                      {step.mathExplanation}
                    </div>
                  )}

                  {step.intermediateValues && (
                    <div className="ml-8 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                      {Object.entries(step.intermediateValues).map(([k, v]) => (
                        <div key={k} className="p-2 bg-slate-900 rounded border border-slate-800/80">
                          <div className="text-[10px] text-slate-500 font-sans">{k}</div>
                          <div className="text-white font-bold truncate">
                            {typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-xs text-slate-400">
              No math steps generated yet. Launch training to inspect state.
            </div>
          )}

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(7)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Simulation</span>
            </button>
            <button
              onClick={() => setActiveStepTab(9)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Model Evaluation (Step 9)</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 9: DEDICATED MODEL EVALUATION & METRICS CENTER                       */}
      {/* ========================================================================= */}
      {activeStepTab === 9 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="space-y-1">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart className="w-5 h-5 text-indigo-400" />
              <span>Step 9: Model Evaluation & Performance Metrics</span>
            </h2>
            <p className="text-xs text-slate-400">
              Diagnostic metrics computed on the out-of-sample test split ({preparedData?.X_test.length} unseen samples).
            </p>
          </div>

          {trainResult && (
            <div className="space-y-6">
              {/* Primary KPI Metric Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {trainResult.metrics.accuracy !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Classification Accuracy</span>
                    <div className="text-2xl font-bold font-mono text-emerald-400">
                      {(trainResult.metrics.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-[11px] text-slate-500">(TP + TN) / Total</div>
                  </div>
                )}
                {trainResult.metrics.precision !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Precision</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.precision}
                    </div>
                    <div className="text-[11px] text-slate-500">TP / (TP + FP)</div>
                  </div>
                )}
                {trainResult.metrics.recall !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Recall (Sensitivity)</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.recall}
                    </div>
                    <div className="text-[11px] text-slate-500">TP / (TP + FN)</div>
                  </div>
                )}
                {trainResult.metrics.f1Score !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">F1-Score</span>
                    <div className="text-2xl font-bold font-mono text-indigo-400">
                      {trainResult.metrics.f1Score}
                    </div>
                    <div className="text-[11px] text-slate-500">Harmonic Mean</div>
                  </div>
                )}

                {/* Regression Metrics */}
                {trainResult.metrics.r2 !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">R² Coefficient of Determination</span>
                    <div className="text-2xl font-bold font-mono text-emerald-400">
                      {trainResult.metrics.r2}
                    </div>
                    <div className="text-[11px] text-slate-500">1 - (SS_res / SS_tot)</div>
                  </div>
                )}
                {trainResult.metrics.mse !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Mean Squared Error (MSE)</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.mse}
                    </div>
                    <div className="text-[11px] text-slate-500">(1/n) ∑ (y - ŷ)²</div>
                  </div>
                )}
                {trainResult.metrics.mae !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Mean Absolute Error (MAE)</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.mae}
                    </div>
                    <div className="text-[11px] text-slate-500">(1/n) ∑ |y - ŷ|</div>
                  </div>
                )}
                {trainResult.metrics.rmse !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Root MSE (RMSE)</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.rmse}
                    </div>
                    <div className="text-[11px] text-slate-500">√MSE in original units</div>
                  </div>
                )}

                {/* Clustering Metrics */}
                {trainResult.metrics.inertia !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Inertia (WCSS)</span>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {trainResult.metrics.inertia}
                    </div>
                    <div className="text-[11px] text-slate-500">Within-Cluster Sum of Squares</div>
                  </div>
                )}
                {trainResult.metrics.silhouetteScore !== undefined && (
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                    <span className="text-xs text-slate-400">Silhouette Score</span>
                    <div className="text-2xl font-bold font-mono text-cyan-400">
                      {trainResult.metrics.silhouetteScore}
                    </div>
                    <div className="text-[11px] text-slate-500">Cluster separation [-1, +1]</div>
                  </div>
                )}
              </div>

              {/* Confusion Matrix Visualization for Classification */}
              {trainResult.metrics.confusionMatrix && (
                <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">Confusion Matrix</h3>
                      <p className="text-xs text-slate-400">Actual (Rows) versus Predicted (Columns)</p>
                    </div>
                  </div>

                  <div className="max-w-md mx-auto overflow-x-auto">
                    <table className="w-full text-center border-collapse">
                      <thead>
                        <tr>
                          <th className="p-2"></th>
                          {trainResult.metrics.confusionMatrix.labels.map((lbl) => (
                            <th key={lbl} className="p-2 text-xs font-semibold text-indigo-400">
                              Pred: {lbl}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {trainResult.metrics.confusionMatrix.matrix.map((row, rIdx) => (
                          <tr key={rIdx}>
                            <th className="p-2 text-xs font-semibold text-slate-300 text-left">
                              Act: {trainResult.metrics.confusionMatrix!.labels[rIdx]}
                            </th>
                            {row.map((val, cIdx) => {
                              const isDiagonal = rIdx === cIdx;
                              return (
                                <td key={cIdx} className="p-1">
                                  <div
                                    className={`p-3 rounded-lg font-mono font-bold text-sm ${
                                      isDiagonal
                                        ? 'bg-emerald-950/60 border border-emerald-500/50 text-emerald-300'
                                        : val > 0
                                        ? 'bg-rose-950/60 border border-rose-500/50 text-rose-300'
                                        : 'bg-slate-900 border border-slate-800 text-slate-500'
                                    }`}
                                  >
                                    {val}
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
              )}
            </div>
          )}

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(8)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Math Steps</span>
            </button>
            <button
              onClick={() => setActiveStepTab(10)}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <span>Next: Live Prediction Bench (Step 10)</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* STEP 10: DEDICATED REAL-TIME PREDICTION LAB                               */}
      {/* ========================================================================= */}
      {activeStepTab === 10 && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-6 animate-fadeIn">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-indigo-400" />
                <span>Step 10: Real-Time Interactive Prediction Lab</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Adjust feature numbers or click presets to see the model predict instantly.
              </p>
            </div>

            <div className="px-5 py-2.5 bg-indigo-950 border border-indigo-500/50 rounded-xl flex items-center gap-3">
              <span className="text-xs text-indigo-300 font-medium">Model Output:</span>
              <span className="font-mono font-bold text-white text-lg">
                {livePrediction !== null ? String(livePrediction) : 'Calculating...'}
              </span>
            </div>
          </div>

          {/* Feature Inputs Grid */}
          <div className="space-y-4">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider">
              Input Vector Attributes
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {preparedData?.featureNames.map((feat) => {
                const currentVal = userInputs[feat] !== undefined ? userInputs[feat] : 0;
                return (
                  <div key={feat} className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-300 truncate max-w-[180px]" title={feat}>
                        {feat}
                      </span>
                      <span className="font-mono text-indigo-400 font-bold">{currentVal}</span>
                    </div>

                    <input
                      type="number"
                      step="0.1"
                      value={currentVal}
                      onChange={(e) => handleInputChange(feat, Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs font-mono text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                );
              })}
            </div>
          </div>

          <div className="flex justify-between pt-4 border-t border-slate-800">
            <button
              onClick={() => setActiveStepTab(9)}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium flex items-center gap-1.5"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Previous: Evaluation</span>
            </button>

            <button
              onClick={handleSave}
              className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Save Finished Experiment</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
