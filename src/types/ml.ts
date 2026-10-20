export type TaskType = 'classification' | 'regression' | 'clustering';

export type AlgorithmId =
  | 'linear_regression'
  | 'logistic_regression'
  | 'decision_tree'
  | 'random_forest'
  | 'kmeans'
  | 'knn'
  | 'svm'
  | 'pca';

export interface DatasetFeature {
  name: string;
  type: 'numeric' | 'categorical';
  description: string;
  min?: number;
  max?: number;
  mean?: number;
  categories?: string[];
  missingCount: number;
}

export interface DatasetRecord {
  [key: string]: string | number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  taskType: TaskType;
  description: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  rows: number;
  columns: number;
  features: DatasetFeature[];
  targetColumn?: string;
  targetClasses?: string[];
  recommendedAlgorithms: AlgorithmId[];
  challenge: string;
  disclaimer?: string;
  data: DatasetRecord[];
}

export interface PreprocessingConfig {
  missingStrategy: 'drop' | 'impute_mean' | 'impute_median' | 'impute_mode' | 'none';
  removeDuplicates: boolean;
  scaleMethod: 'standard' | 'minmax' | 'none';
  encodeCategorical: 'one_hot' | 'label' | 'none';
  testSplitRatio: number; // e.g. 0.2
  randomSeed: number;
  selectedFeatures: string[];
  targetColumn?: string;
}

export interface PreparedData {
  featureNames: string[];
  targetName?: string;
  targetClasses?: string[];
  X_train: number[][];
  y_train: number[];
  X_test: number[][];
  y_test: number[];
  rawTrainRows: DatasetRecord[];
  rawTestRows: DatasetRecord[];
  scalerParams?: {
    means: number[];
    stds: number[];
    mins: number[];
    maxs: number[];
  };
  droppedRowsCount: number;
  duplicateRowsRemoved: number;
  outlierWarnings: string[];
}

export interface ConfusionMatrixData {
  matrix: number[][]; // [actual][predicted]
  labels: string[];
}

export interface EvaluationMetrics {
  // Classification
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  confusionMatrix?: ConfusionMatrixData;

  // Regression
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;

  // Clustering
  inertia?: number;
  silhouetteScore?: number;
  clusterCounts?: Record<number, number>;

  // Common
  trainTimeMs: number;
  sampleCount: number;
}

export interface SimulationStep {
  stepNumber: number;
  title: string;
  description: string;
  operation: string;
  mathExplanation?: string;
  formula?: string;
  dataHighlights?: {
    sampleIndices?: number[];
    featureNames?: string[];
    centroidCoords?: number[][];
  };
  intermediateValues?: Record<string, string | number | number[] | number[][] | string[] | any>;
  visualState: {
    // For regression: line params [slope, intercept]
    regressionLine?: { slope: number; intercept: number };
    // For classification: decision boundary line or contours
    decisionBoundary?: { w: number[]; b: number };
    // For K-Means: current centroids and point assignments
    centroids?: { x: number; y: number; id: number }[];
    assignments?: number[];
    // For tree: current node being split or built
    activeNodeId?: string;
    // For KNN: target point and highlighted nearest neighbors
    queryPoint?: number[];
    neighborIndices?: number[];
  };
}

export interface TreeNode {
  id: string;
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  impurity: number;
  samples: number;
  value: number[] | number; // class counts or mean value
  predictedClass?: string | number;
  left?: TreeNode;
  right?: TreeNode;
  isLeaf: boolean;
  gain?: number;
}

export interface Experiment {
  id: string;
  name: string;
  createdAt: string;
  datasetId: string;
  datasetName: string;
  algorithmId: AlgorithmId;
  algorithmName: string;
  hyperparameters: Record<string, number | string | boolean>;
  preprocessing: PreprocessingConfig;
  metrics: EvaluationMetrics;
  predictionsSample?: { actual: number | string; predicted: number | string }[];
  status: 'completed' | 'failed';
  notes?: string;
}

export interface CurriculumLesson {
  id: string;
  moduleId: number;
  moduleTitle: string;
  title: string;
  order: number;
  summary: string;
  simpleExplanation: string;
  technicalDefinition: string;
  realWorldExample: string;
  stepByStep: string[];
  codeSnippet?: string;
  keyTakeaways: string[];
  practicalTask: string;
  quizQuestions: {
    id: string;
    question: string;
    options: string[];
    correctIndex: number;
    explanation: string;
    misconception: string;
  }[];
}

export interface QuizQuestion {
  id: string;
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  type: 'multiple_choice' | 'true_false' | 'output_prediction' | 'calculation' | 'algorithm_match';
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
  misconception: string;
  relatedLessonId?: string;
}

export interface PracticalChallenge {
  id: string;
  title: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  description: string;
  datasetId: string;
  recommendedAlgorithm: AlgorithmId;
  targetMetricName: string;
  targetMetricThreshold: number; // e.g. R2 > 0.65 or Accuracy > 0.9
  hint: string;
  badgeName: string;
}

export interface UserProgress {
  completedLessons: string[];
  completedQuizzes: Record<string, number>; // quizId -> bestScore
  completedChallenges: string[];
  earnedBadges: {
    id: string;
    title: string;
    description: string;
    icon: string;
    unlockedAt: string;
  }[];
  experimentsCount: number;
  simulationsRun: number;
  lastActiveLessonId?: string;
}
