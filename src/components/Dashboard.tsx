import React from 'react';
import {
  BrainCircuit,
  BookOpen,
  FlaskConical,
  Database,
  Award,
  ArrowRight,
  TrendingUp,
  Sparkles,
  Play,
  Upload,
  CheckCircle2,
  Clock,
  Layers,
} from 'lucide-react';
import { AlgorithmId, Experiment, UserProgress } from '../types/ml';
import { BUILT_IN_DATASETS } from '../ml/datasets';
import { CURRICULUM_LESSONS } from '../ml/curriculum';

interface DashboardProps {
  progress: UserProgress;
  experiments: Experiment[];
  onSelectTab: (tab: any) => void;
  onLaunchAlgorithm: (algoId: AlgorithmId, datasetId?: string) => void;
  onSelectLesson: (lessonId: string) => void;
  onSelectDataset: (datasetId: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  progress,
  experiments,
  onSelectTab,
  onLaunchAlgorithm,
  onSelectLesson,
  onSelectDataset,
}) => {
  const completedLessonCount = progress.completedLessons.length;
  const totalLessons = CURRICULUM_LESSONS.length;
  const progressPercent = Math.round((completedLessonCount / totalLessons) * 100);

  // Determine next recommended lesson
  const nextLesson =
    CURRICULUM_LESSONS.find((l) => !progress.completedLessons.includes(l.id)) ||
    CURRICULUM_LESSONS[0];

  const algorithmCards: Array<{
    id: AlgorithmId;
    name: string;
    type: 'Supervised' | 'Unsupervised';
    task: 'Regression' | 'Classification' | 'Clustering' | 'Dimensionality';
    difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
    description: string;
    defaultDataset: string;
  }> = [
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      type: 'Supervised',
      task: 'Regression',
      difficulty: 'Beginner',
      description: 'Fit the optimal best-fit line using Ordinary Least Squares and gradient descent.',
      defaultDataset: 'synthetic_linear',
    },
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      type: 'Supervised',
      task: 'Classification',
      difficulty: 'Beginner',
      description: 'Model discrete class probabilities using Sigmoid activation and decision thresholds.',
      defaultDataset: 'synthetic_binary',
    },
    {
      id: 'decision_tree',
      name: 'Decision Tree',
      type: 'Supervised',
      task: 'Classification',
      difficulty: 'Beginner',
      description: 'Recursive hierarchical splitting using Gini Impurity and Information Gain.',
      defaultDataset: 'iris',
    },
    {
      id: 'kmeans',
      name: 'K-Means Clustering',
      type: 'Unsupervised',
      task: 'Clustering',
      difficulty: 'Beginner',
      description: 'Partition data into K clusters via iterative centroid assignment and recalculation.',
      defaultDataset: 'synthetic_blobs',
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      type: 'Supervised',
      task: 'Classification',
      difficulty: 'Intermediate',
      description: 'Ensemble bagging of randomized decision trees with committee voting.',
      defaultDataset: 'wine',
    },
    {
      id: 'knn',
      name: 'K-Nearest Neighbors',
      type: 'Supervised',
      task: 'Classification',
      difficulty: 'Beginner',
      description: 'Instance-based learning finding K closest neighbors in Euclidean feature space.',
      defaultDataset: 'iris',
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      type: 'Supervised',
      task: 'Classification',
      difficulty: 'Intermediate',
      description: 'Maximize the separating margin hyperplane between distinct classes.',
      defaultDataset: 'breast_cancer',
    },
    {
      id: 'pca',
      name: 'Principal Component Analysis',
      type: 'Unsupervised',
      task: 'Dimensionality',
      difficulty: 'Intermediate',
      description: 'Project high-dimensional data onto orthogonal directions of maximal variance.',
      defaultDataset: 'wine',
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8 animate-fadeIn">
      {/* 1. Welcome Section & Learning Progress */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-indigo-950/40 border border-slate-800 p-6 md:p-8 flex flex-col justify-between relative overflow-hidden">
          <div className="space-y-3 z-10">
            <div className="flex items-center gap-2 text-xs font-medium text-indigo-400">
              <Sparkles className="w-4 h-4" />
              <span>Interactive Machine Learning Lab</span>
              <span className="text-slate-600">·</span>
              <span>Zero to Practical Understanding</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
              Welcome to the ML Algorithm Simulator
            </h1>
            <p className="text-sm md:text-base text-slate-300 max-w-2xl leading-relaxed">
              Explore how algorithms truly learn from data. Step through analytical least squares, watch decision trees split on Gini impurity, and inspect centroids migrating frame by frame.
            </p>
          </div>

          <div className="mt-6 pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4 z-10">
            <div>
              <div className="text-xs text-slate-400">Course Progress</div>
              <div className="text-xl font-bold text-white tabular-nums">
                {progressPercent}%
              </div>
              <div className="text-xs text-slate-500">{completedLessonCount} / {totalLessons} lessons</div>
            </div>

            <div>
              <div className="text-xs text-slate-400">Simulations Run</div>
              <div className="text-xl font-bold text-white tabular-nums">
                {progress.simulationsRun}
              </div>
              <div className="text-xs text-slate-500">Live executions</div>
            </div>

            <div>
              <div className="text-xs text-slate-400">Saved Experiments</div>
              <div className="text-xl font-bold text-white tabular-nums">
                {experiments.length}
              </div>
              <div className="text-xs text-slate-500">Models benchmarked</div>
            </div>

            <div>
              <div className="text-xs text-slate-400">Badges Earned</div>
              <div className="text-xl font-bold text-amber-400 tabular-nums">
                {progress.earnedBadges.length}
              </div>
              <div className="text-xs text-slate-500">Achievements</div>
            </div>
          </div>
        </div>

        {/* Continue Learning Card */}
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                Recommended Next Step
              </span>
              <BookOpen className="w-4 h-4 text-slate-400" />
            </div>
            <h3 className="text-lg font-bold text-white">
              {nextLesson.title}
            </h3>
            <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed">
              {nextLesson.summary}
            </p>
          </div>

          <div className="mt-6 space-y-3">
            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-indigo-500 h-full rounded-full transition-all duration-500"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <button
              onClick={() => {
                onSelectLesson(nextLesson.id);
                onSelectTab('learn');
              }}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors shadow-sm shadow-indigo-600/30"
            >
              <span>Continue Lesson</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Quick Actions Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <button
          onClick={() => onSelectTab('datasets')}
          className="p-4 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all text-left flex items-center gap-3.5 group"
        >
          <div className="p-2.5 rounded-lg bg-indigo-500/10 text-indigo-400 group-hover:scale-105 transition-transform">
            <Upload className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white group-hover:text-indigo-400 transition-colors">
              Explore & Upload Datasets
            </div>
            <div className="text-xs text-slate-400">
              Inspect tabular distributions or import custom CSV
            </div>
          </div>
        </button>

        <button
          onClick={() => onSelectTab('simulator')}
          className="p-4 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all text-left flex items-center gap-3.5 group"
        >
          <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400 group-hover:scale-105 transition-transform">
            <Play className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white group-hover:text-emerald-400 transition-colors">
              Step-by-Step Simulator
            </div>
            <div className="text-xs text-slate-400">
              Interactive execution with play/pause and math breakdown
            </div>
          </div>
        </button>

        <button
          onClick={() => onSelectTab('quizzes')}
          className="p-4 rounded-xl bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all text-left flex items-center gap-3.5 group"
        >
          <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400 group-hover:scale-105 transition-transform">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white group-hover:text-amber-400 transition-colors">
              Quizzes & Challenges
            </div>
            <div className="text-xs text-slate-400">
              Test concepts and unlock practical machine learning badges
            </div>
          </div>
        </button>
      </div>

      {/* 2. Core Algorithm Laboratory Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Algorithm Laboratory
            </h2>
            <p className="text-xs text-slate-400">
              Select an algorithm to configure hyperparameters and watch it execute step by step
            </p>
          </div>
          <button
            onClick={() => onSelectTab('simulator')}
            className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
          >
            Open Simulator
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {algorithmCards.map((algo) => (
            <div
              key={algo.id}
              className="rounded-xl bg-slate-900 border border-slate-800 hover:border-slate-700 p-5 flex flex-col justify-between transition-all hover:shadow-lg hover:shadow-indigo-950/20 group"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-medium text-slate-300">{algo.task}</span>
                  <span className="text-slate-500">·</span>
                  <span>{algo.difficulty}</span>
                </div>

                <h3 className="text-base font-bold text-white group-hover:text-indigo-400 transition-colors">
                  {algo.name}
                </h3>

                <p className="text-xs text-slate-400 leading-relaxed">
                  {algo.description}
                </p>
              </div>

              <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                <span className="text-[11px] text-slate-500 font-mono">
                  {algo.type}
                </span>

                <button
                  onClick={() => onLaunchAlgorithm(algo.id, algo.defaultDataset)}
                  className="px-3 py-1.5 text-xs font-medium text-white bg-slate-800 hover:bg-indigo-600 rounded-md transition-colors flex items-center gap-1.5"
                >
                  <Play className="w-3 h-3 fill-current" />
                  <span>Simulate</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Recently Explored Datasets & Recent Experiments Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Built-in Datasets Library Preview */}
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white">
                Curated Datasets
              </h3>
              <p className="text-xs text-slate-400">
                10 real & synthetic benchmark datasets ready for experimentation
              </p>
            </div>
            <button
              onClick={() => onSelectTab('datasets')}
              className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
            >
              View All
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-2.5">
            {BUILT_IN_DATASETS.slice(0, 4).map((ds) => (
              <div
                key={ds.id}
                onClick={() => onSelectDataset(ds.id)}
                className="p-3 rounded-lg bg-slate-950/60 hover:bg-slate-800/70 border border-slate-800/60 hover:border-slate-700 transition-all flex items-center justify-between cursor-pointer group"
              >
                <div className="space-y-1">
                  <div className="text-sm font-semibold text-white group-hover:text-indigo-400 transition-colors">
                    {ds.name}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="capitalize">{ds.taskType}</span>
                    <span aria-hidden="true">·</span>
                    <span>{ds.rows} samples</span>
                    <span aria-hidden="true">·</span>
                    <span>{ds.columns} features</span>
                  </div>
                </div>

                <button className="text-xs font-medium text-slate-400 group-hover:text-white px-2.5 py-1 rounded bg-slate-800 group-hover:bg-indigo-600 transition-colors">
                  Explore
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Experiments Card */}
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white">
                Recent Model Experiments
              </h3>
              <p className="text-xs text-slate-400">
                Trained configurations with benchmarked evaluation metrics
              </p>
            </div>
            <button
              onClick={() => onSelectTab('experiments')}
              className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
            >
              Compare
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {experiments.length === 0 ? (
            <div className="h-48 flex flex-col items-center justify-center text-center p-6 border border-dashed border-slate-800 rounded-xl space-y-2">
              <FlaskConical className="w-8 h-8 text-slate-600" />
              <div className="text-sm font-medium text-slate-300">No experiments saved yet</div>
              <p className="text-xs text-slate-500 max-w-xs">
                Launch any algorithm simulation and click &quot;Save Experiment&quot; to compare side-by-side!
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              {experiments.slice(-4).reverse().map((exp) => (
                <div
                  key={exp.id}
                  className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 flex items-center justify-between text-xs"
                >
                  <div className="space-y-1">
                    <div className="font-semibold text-white">
                      {exp.name}
                    </div>
                    <div className="text-slate-400 flex items-center gap-2">
                      <span>{exp.algorithmName}</span>
                      <span aria-hidden="true">·</span>
                      <span>{exp.datasetName}</span>
                    </div>
                  </div>

                  <div className="text-right">
                    {exp.metrics.r2 !== undefined && (
                      <div className="font-mono text-emerald-400 font-semibold">
                        R² = {exp.metrics.r2}
                      </div>
                    )}
                    {exp.metrics.accuracy !== undefined && (
                      <div className="font-mono text-emerald-400 font-semibold">
                        Acc = {(exp.metrics.accuracy * 100).toFixed(1)}%
                      </div>
                    )}
                    {exp.metrics.inertia !== undefined && (
                      <div className="font-mono text-cyan-400 font-semibold">
                        Inertia = {exp.metrics.inertia}
                      </div>
                    )}
                    <div className="text-[10px] text-slate-500">
                      {new Date(exp.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
