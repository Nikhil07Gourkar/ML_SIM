import React, { useState, useEffect } from 'react';
import { TopNav, ActiveTab } from './components/TopNav';
import { Dashboard } from './components/Dashboard';
import { AlgorithmSimulator } from './components/AlgorithmSimulator';
import { DatasetExplorer } from './components/DatasetExplorer';
import { LearnML } from './components/LearnML';
import { QuizChallenges } from './components/QuizChallenges';
import { ExperimentWorkspace } from './components/ExperimentWorkspace';
import { AITutor } from './components/AITutor';
import { SettingsModal } from './components/SettingsModal';
import { AlgorithmId, DatasetInfo, Experiment, UserProgress } from './types/ml';
import { BUILT_IN_DATASETS } from './ml/datasets';
import { CURRICULUM_LESSONS } from './ml/curriculum';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [isDark, setIsDark] = useState<boolean>(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Active workspace state
  const [selectedDataset, setSelectedDataset] = useState<DatasetInfo>(BUILT_IN_DATASETS[1]); // Study hours
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmId>('linear_regression');
  const [selectedLessonId, setSelectedLessonId] = useState<string>(CURRICULUM_LESSONS[0].id);

  // User Progress state with LocalStorage persistence
  const [progress, setProgress] = useState<UserProgress>(() => {
    try {
      const saved = localStorage.getItem('ml_lab_progress');
      if (saved) return JSON.parse(saved);
    } catch {}
    return {
      completedLessons: ['m1_intro_ml'],
      completedQuizzes: {},
      completedChallenges: [],
      earnedBadges: [
        {
          id: 'First Line Drawn',
          title: 'First Line Drawn',
          description: 'Solved the Study Hours Linear Regression challenge',
          icon: 'LineChart',
          unlockedAt: new Date().toISOString(),
        },
      ],
      experimentsCount: 1,
      simulationsRun: 3,
    };
  });

  // Saved Experiments state with LocalStorage persistence
  const [experiments, setExperiments] = useState<Experiment[]>(() => {
    try {
      const saved = localStorage.getItem('ml_lab_experiments');
      if (saved) return JSON.parse(saved);
    } catch {}
    return [
      {
        id: 'exp_default_1',
        name: 'Baseline Linear Fit - Study Hours',
        createdAt: new Date(Date.now() - 3600000).toISOString(),
        datasetId: 'synthetic_linear',
        datasetName: 'Study Hours vs Exam Score',
        algorithmId: 'linear_regression',
        algorithmName: 'LINEAR REGRESSION',
        hyperparameters: { fitIntercept: true, learningRate: 0.05 },
        preprocessing: {
          missingStrategy: 'impute_mean',
          removeDuplicates: true,
          scaleMethod: 'none',
          encodeCategorical: 'one_hot',
          testSplitRatio: 0.2,
          randomSeed: 42,
          selectedFeatures: ['study_hours'],
          targetColumn: 'exam_score',
        },
        metrics: {
          r2: 0.884,
          mse: 14.2,
          mae: 3.12,
          trainTimeMs: 4,
          sampleCount: 80,
        },
        status: 'completed',
      },
    ];
  });

  // Persist progress & experiments
  useEffect(() => {
    try {
      localStorage.setItem('ml_lab_progress', JSON.stringify(progress));
    } catch {}
  }, [progress]);

  useEffect(() => {
    try {
      localStorage.setItem('ml_lab_experiments', JSON.stringify(experiments));
    } catch {}
  }, [experiments]);

  // Apply dark mode class to html
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  // Handlers
  const handleToggleCompleteLesson = (lessonId: string) => {
    setProgress((prev) => {
      const exists = prev.completedLessons.includes(lessonId);
      const updated = exists
        ? prev.completedLessons.filter((id) => id !== lessonId)
        : [...prev.completedLessons, lessonId];
      return { ...prev, completedLessons: updated };
    });
  };

  const handleUnlockBadge = (badgeTitle: string) => {
    setProgress((prev) => {
      if (prev.earnedBadges.some((b) => b.title === badgeTitle)) return prev;
      const newBadge = {
        id: badgeTitle,
        title: badgeTitle,
        description: 'Completed challenge requirements',
        icon: 'Award',
        unlockedAt: new Date().toISOString(),
      };
      return {
        ...prev,
        earnedBadges: [...prev.earnedBadges, newBadge],
      };
    });
  };

  const handleSaveExperiment = (exp: Experiment) => {
    setExperiments((prev) => [exp, ...prev]);
    setProgress((prev) => ({
      ...prev,
      experimentsCount: prev.experimentsCount + 1,
      simulationsRun: prev.simulationsRun + 1,
    }));
  };

  const handleDeleteExperiment = (id: string) => {
    setExperiments((prev) => prev.filter((e) => e.id !== id));
  };

  const handleDuplicateExperiment = (exp: Experiment) => {
    const copy: Experiment = {
      ...exp,
      id: `exp_${Date.now()}`,
      name: `${exp.name} (Copy)`,
      createdAt: new Date().toISOString(),
    };
    setExperiments((prev) => [copy, ...prev]);
  };

  const handleLaunchAlgorithm = (algoId: AlgorithmId, datasetId?: string) => {
    setSelectedAlgorithm(algoId);
    if (datasetId) {
      const ds = BUILT_IN_DATASETS.find((d) => d.id === datasetId);
      if (ds) setSelectedDataset(ds);
    }
    setActiveTab('simulator');
  };

  const handleLaunchChallenge = (datasetId: string, algoId: AlgorithmId) => {
    const ds = BUILT_IN_DATASETS.find((d) => d.id === datasetId);
    if (ds) setSelectedDataset(ds);
    setSelectedAlgorithm(algoId);
    setActiveTab('simulator');
  };

  const handleResetProgress = () => {
    setProgress({
      completedLessons: [],
      completedQuizzes: {},
      completedChallenges: [],
      earnedBadges: [],
      experimentsCount: 0,
      simulationsRun: 0,
    });
    setExperiments([]);
    localStorage.removeItem('ml_lab_progress');
    localStorage.removeItem('ml_lab_experiments');
  };

  // Context passed to AI Tutor
  const tutorContext = {
    lessonTitle: CURRICULUM_LESSONS.find((l) => l.id === selectedLessonId)?.title,
    algorithmId: selectedAlgorithm,
    datasetName: selectedDataset.name,
    taskType: selectedDataset.taskType,
  };

  return (
    <div className={`min-h-screen ${isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'} transition-colors duration-200`}>
      {/* Top Bar following Top Bar Contract */}
      <TopNav
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDark={isDark}
        setIsDark={setIsDark}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onQuickSimulate={() => setActiveTab('simulator')}
      />

      {/* Main Content Area */}
      <main className="pb-16">
        {activeTab === 'dashboard' && (
          <Dashboard
            progress={progress}
            experiments={experiments}
            onSelectTab={setActiveTab}
            onLaunchAlgorithm={handleLaunchAlgorithm}
            onSelectLesson={(lessonId) => {
              setSelectedLessonId(lessonId);
              setActiveTab('learn');
            }}
            onSelectDataset={(datasetId) => {
              const ds = BUILT_IN_DATASETS.find((d) => d.id === datasetId);
              if (ds) setSelectedDataset(ds);
              setActiveTab('datasets');
            }}
          />
        )}

        {activeTab === 'simulator' && (
          <AlgorithmSimulator
            initialAlgorithm={selectedAlgorithm}
            initialDataset={selectedDataset}
            onSaveExperiment={handleSaveExperiment}
            onAskTutor={(ctx) => setActiveTab('tutor')}
          />
        )}

        {activeTab === 'datasets' && (
          <DatasetExplorer
            selectedDataset={selectedDataset}
            onSelectDataset={setSelectedDataset}
            onLaunchSimulation={(ds) => {
              setSelectedDataset(ds);
              if (ds.recommendedAlgorithms.length > 0) {
                setSelectedAlgorithm(ds.recommendedAlgorithms[0]);
              }
              setActiveTab('simulator');
            }}
            onCustomDatasetAdded={(newDs) => {
              setSelectedDataset(newDs);
              if (newDs.recommendedAlgorithms.length > 0) {
                setSelectedAlgorithm(newDs.recommendedAlgorithms[0]);
              }
            }}
          />
        )}

        {activeTab === 'learn' && (
          <LearnML
            currentLessonId={selectedLessonId}
            completedLessons={progress.completedLessons}
            onToggleCompleteLesson={handleToggleCompleteLesson}
            onLaunchTask={(task) => setActiveTab('simulator')}
            onAskTutor={(ctx) => setActiveTab('tutor')}
          />
        )}

        {activeTab === 'quizzes' && (
          <QuizChallenges
            progress={progress}
            onUnlockBadge={handleUnlockBadge}
            onLaunchChallenge={handleLaunchChallenge}
          />
        )}

        {activeTab === 'experiments' && (
          <ExperimentWorkspace
            experiments={experiments}
            onDeleteExperiment={handleDeleteExperiment}
            onDuplicateExperiment={handleDuplicateExperiment}
            onLaunchExperiment={(exp) => {
              const ds = BUILT_IN_DATASETS.find((d) => d.id === exp.datasetId);
              if (ds) setSelectedDataset(ds);
              setSelectedAlgorithm(exp.algorithmId);
              setActiveTab('simulator');
            }}
          />
        )}

        {activeTab === 'tutor' && (
          <AITutor context={tutorContext} />
        )}
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        isDark={isDark}
        setIsDark={setIsDark}
        onResetProgress={handleResetProgress}
      />
    </div>
  );
}
