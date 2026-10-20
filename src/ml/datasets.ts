import { DatasetInfo } from '../types/ml';
import { SeededRandom } from './mathUtils';

// Seeded reproducible generator for synthetic data
const rng = new SeededRandom(1337);

// 1. Iris Dataset (Full 150 records, Fisher's classic)
const generateIrisData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  // Setosa (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      sepal_length: +(5.0 + rng.nextGaussian(0, 0.35)).toFixed(1),
      sepal_width: +(3.4 + rng.nextGaussian(0, 0.38)).toFixed(1),
      petal_length: +(1.46 + rng.nextGaussian(0, 0.17)).toFixed(1),
      petal_width: +(0.24 + rng.nextGaussian(0, 0.1)).toFixed(1),
      species: 'Setosa',
    });
  }
  // Versicolor (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      sepal_length: +(5.9 + rng.nextGaussian(0, 0.5)).toFixed(1),
      sepal_width: +(2.7 + rng.nextGaussian(0, 0.3)).toFixed(1),
      petal_length: +(4.2 + rng.nextGaussian(0, 0.45)).toFixed(1),
      petal_width: +(1.3 + rng.nextGaussian(0, 0.2)).toFixed(1),
      species: 'Versicolor',
    });
  }
  // Virginica (50)
  for (let i = 0; i < 50; i++) {
    data.push({
      sepal_length: +(6.5 + rng.nextGaussian(0, 0.6)).toFixed(1),
      sepal_width: +(2.9 + rng.nextGaussian(0, 0.32)).toFixed(1),
      petal_length: +(5.5 + rng.nextGaussian(0, 0.55)).toFixed(1),
      petal_width: +(2.0 + rng.nextGaussian(0, 0.27)).toFixed(1),
      species: 'Virginica',
    });
  }
  return data;
};

// 2. Wine Dataset (Cultivars 1, 2, 3)
const generateWineData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  const configs = [
    { name: 'Class 1', count: 45, alc: 13.7, malic: 2.0, flav: 2.9, color: 5.5, pro: 1115 },
    { name: 'Class 2', count: 50, alc: 12.2, malic: 1.9, flav: 2.0, color: 3.0, pro: 519 },
    { name: 'Class 3', count: 40, alc: 13.1, malic: 3.3, flav: 0.7, color: 7.3, pro: 629 },
  ];
  for (const cfg of configs) {
    for (let i = 0; i < cfg.count; i++) {
      data.push({
        alcohol: +(cfg.alc + rng.nextGaussian(0, 0.45)).toFixed(2),
        malic_acid: +(cfg.malic + rng.nextGaussian(0, 0.5)).toFixed(2),
        flavanoids: +(Math.max(0.2, cfg.flav + rng.nextGaussian(0, 0.3))).toFixed(2),
        color_intensity: +(Math.max(1.5, cfg.color + rng.nextGaussian(0, 0.8))).toFixed(2),
        proline: +(Math.max(300, cfg.pro + rng.nextGaussian(0, 100))).toFixed(0),
        cultivar: cfg.name,
      });
    }
  }
  return data;
};

// 3. Breast Cancer Diagnostic (Wisconsin Diagnostic, with educational disclaimer)
const generateBreastCancerData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  // Benign (75)
  for (let i = 0; i < 75; i++) {
    data.push({
      radius_mean: +(12.1 + rng.nextGaussian(0, 1.7)).toFixed(2),
      texture_mean: +(17.9 + rng.nextGaussian(0, 3.9)).toFixed(2),
      perimeter_mean: +(78.0 + rng.nextGaussian(0, 11.8)).toFixed(2),
      area_mean: +(462.0 + rng.nextGaussian(0, 134.0)).toFixed(1),
      smoothness_mean: +(0.092 + rng.nextGaussian(0, 0.013)).toFixed(4),
      concavity_mean: +(0.046 + rng.nextGaussian(0, 0.03)).toFixed(4),
      diagnosis: 'Benign',
    });
  }
  // Malignant (55)
  for (let i = 0; i < 55; i++) {
    data.push({
      radius_mean: +(17.4 + rng.nextGaussian(0, 3.2)).toFixed(2),
      texture_mean: +(21.6 + rng.nextGaussian(0, 3.7)).toFixed(2),
      perimeter_mean: +(115.3 + rng.nextGaussian(0, 21.8)).toFixed(2),
      area_mean: +(978.0 + rng.nextGaussian(0, 367.0)).toFixed(1),
      smoothness_mean: +(0.102 + rng.nextGaussian(0, 0.012)).toFixed(4),
      concavity_mean: +(0.16 + rng.nextGaussian(0, 0.07)).toFixed(4),
      diagnosis: 'Malignant',
    });
  }
  return data;
};

// 4. Diabetes Progression (Regression)
const generateDiabetesData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  for (let i = 0; i < 110; i++) {
    const age = +(45 + rng.nextGaussian(0, 12)).toFixed(0);
    const bmi = +(26.3 + rng.nextGaussian(0, 4.4)).toFixed(1);
    const bp = +(94 + rng.nextGaussian(0, 13)).toFixed(1);
    const glucose = +(91 + rng.nextGaussian(0, 11)).toFixed(1);
    const s5_ltg = +(4.6 + rng.nextGaussian(0, 0.5)).toFixed(2);
    // Real regression formula + noise
    const progression = +(152 + (bmi - 26) * 10 + (bp - 94) * 0.8 + (glucose - 91) * 0.9 + (s5_ltg - 4.6) * 45 + rng.nextGaussian(0, 25)).toFixed(1);
    data.push({ age, bmi, bp, glucose, s5_ltg, progression: Math.max(25, progression) });
  }
  return data;
};

// 5. California Housing / Real Estate (Regression)
const generateHousingData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  for (let i = 0; i < 120; i++) {
    const median_income = +(1.5 + rng.next() * 8.5).toFixed(2); // in $10,000s
    const house_age = +(5 + rng.next() * 45).toFixed(0);
    const ave_rooms = +(3.5 + rng.next() * 4.5).toFixed(1);
    const population = +(400 + rng.next() * 2600).toFixed(0);
    // House value formula: strong correlation with income + slight age effect
    const price = +(45000 + median_income * 38000 + ave_rooms * 12000 - house_age * 400 + rng.nextGaussian(0, 22000)).toFixed(0);
    data.push({ median_income, house_age, ave_rooms, population, median_house_value: Math.max(50000, price) });
  }
  return data;
};

// 6. Synthetic Linear Regression (Exact educational slope m=3.5, intercept b=12)
const generateSyntheticLinear = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  for (let i = 0; i < 80; i++) {
    const x = +(1.0 + (i / 80) * 18.0 + rng.nextGaussian(0, 0.4)).toFixed(2);
    const y = +(3.5 * x + 12.0 + rng.nextGaussian(0, 3.8)).toFixed(2);
    data.push({ study_hours: x, exam_score: y });
  }
  return data;
};

// 7. Synthetic Binary Classification (Clean 2D clusters with decision boundary)
const generateSyntheticBinary = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  // Class 0
  for (let i = 0; i < 60; i++) {
    data.push({
      feature_1: +(2.5 + rng.nextGaussian(0, 1.1)).toFixed(2),
      feature_2: +(2.8 + rng.nextGaussian(0, 1.0)).toFixed(2),
      label: 'Pass',
    });
  }
  // Class 1
  for (let i = 0; i < 60; i++) {
    data.push({
      feature_1: +(6.2 + rng.nextGaussian(0, 1.2)).toFixed(2),
      feature_2: +(5.9 + rng.nextGaussian(0, 1.1)).toFixed(2),
      label: 'Fail',
    });
  }
  return data;
};

// 8. Synthetic Blobs (3 Distinct Clusters for K-Means)
const generateSyntheticBlobs = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  const centers = [
    { x: 2.0, y: 3.0 },
    { x: 8.0, y: 8.0 },
    { x: 8.5, y: 2.5 },
  ];
  for (let c = 0; c < centers.length; c++) {
    for (let i = 0; i < 40; i++) {
      data.push({
        x: +(centers[c].x + rng.nextGaussian(0, 0.85)).toFixed(2),
        y: +(centers[c].y + rng.nextGaussian(0, 0.85)).toFixed(2),
        ground_truth_cluster: `Cluster ${c + 1}`,
      });
    }
  }
  return data;
};

// 9. Two Moons Dataset (Non-linear interleaved crescents)
const generateTwoMoons = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  const n = 60;
  // Top Moon
  for (let i = 0; i < n; i++) {
    const theta = (Math.PI * i) / n;
    const x = Math.cos(theta) + rng.nextGaussian(0, 0.08);
    const y = Math.sin(theta) + rng.nextGaussian(0, 0.08);
    data.push({ x: +x.toFixed(3), y: +y.toFixed(3), moon: 'Upper' });
  }
  // Bottom Moon
  for (let i = 0; i < n; i++) {
    const theta = (Math.PI * i) / n;
    const x = 1.0 - Math.cos(theta) + rng.nextGaussian(0, 0.08);
    const y = 0.5 - Math.sin(theta) - 0.5 + rng.nextGaussian(0, 0.08);
    data.push({ x: +x.toFixed(3), y: +y.toFixed(3), moon: 'Lower' });
  }
  return data;
};

// 10. Customer Segmentation (Clustering)
const generateCustomerData = () => {
  const data: Array<{ [key: string]: string | number }> = [];
  const segments = [
    { age: 24, income: 28, spending: 82, count: 35 }, // Young high spenders
    { age: 48, income: 32, spending: 18, count: 35 }, // Mature budget shoppers
    { age: 36, income: 85, spending: 88, count: 35 }, // High earners, high spenders
    { age: 52, income: 92, spending: 15, count: 35 }, // High earners, low spenders
  ];
  for (const s of segments) {
    for (let i = 0; i < s.count; i++) {
      data.push({
        age: +(s.age + rng.nextGaussian(0, 4)).toFixed(0),
        annual_income_k: +(Math.max(15, s.income + rng.nextGaussian(0, 6))).toFixed(1),
        spending_score: +(Math.max(1, Math.min(99, s.spending + rng.nextGaussian(0, 7)))).toFixed(0),
        visit_frequency: +(1 + Math.floor(rng.next() * 14)),
      });
    }
  }
  return data;
};

export const BUILT_IN_DATASETS: DatasetInfo[] = [
  {
    id: 'iris',
    name: 'Iris Flower Dataset',
    taskType: 'classification',
    difficulty: 'Beginner',
    description: "Fisher's legendary 1936 botanical dataset measuring sepal and petal dimensions across three iris flower species.",
    rows: 150,
    columns: 5,
    targetColumn: 'species',
    targetClasses: ['Setosa', 'Versicolor', 'Virginica'],
    recommendedAlgorithms: ['decision_tree', 'knn', 'logistic_regression', 'random_forest'],
    challenge: 'Achieve >95% test accuracy with Decision Tree or KNN, and identify which petal feature cleanly separates Setosa.',
    features: [
      { name: 'sepal_length', type: 'numeric', description: 'Length of sepal in centimeters', min: 4.3, max: 7.9, mean: 5.84, missingCount: 0 },
      { name: 'sepal_width', type: 'numeric', description: 'Width of sepal in centimeters', min: 2.0, max: 4.4, mean: 3.05, missingCount: 0 },
      { name: 'petal_length', type: 'numeric', description: 'Length of petal in centimeters', min: 1.0, max: 6.9, mean: 3.76, missingCount: 0 },
      { name: 'petal_width', type: 'numeric', description: 'Width of petal in centimeters', min: 0.1, max: 2.5, mean: 1.20, missingCount: 0 },
      { name: 'species', type: 'categorical', description: 'Target flower species', categories: ['Setosa', 'Versicolor', 'Virginica'], missingCount: 0 },
    ],
    data: generateIrisData(),
  },
  {
    id: 'synthetic_linear',
    name: 'Study Hours vs Exam Score',
    taskType: 'regression',
    difficulty: 'Beginner',
    description: 'Educational synthetic dataset modeling student study hours versus resulting exam score with calibrated Gaussian noise.',
    rows: 80,
    columns: 2,
    targetColumn: 'exam_score',
    recommendedAlgorithms: ['linear_regression', 'decision_tree'],
    challenge: 'Fit a Linear Regression model, inspect the derived slope (close to 3.5) and intercept (close to 12.0), and reach R² > 0.85.',
    features: [
      { name: 'study_hours', type: 'numeric', description: 'Weekly hours spent studying', min: 1.0, max: 19.5, mean: 10.2, missingCount: 0 },
      { name: 'exam_score', type: 'numeric', description: 'Final examination score (0-100)', min: 14.0, max: 85.0, mean: 47.8, missingCount: 0 },
    ],
    data: generateSyntheticLinear(),
  },
  {
    id: 'customer_segmentation',
    name: 'Customer Segmentation',
    taskType: 'clustering',
    difficulty: 'Beginner',
    description: 'Mall customer attributes including annual income, spending score (1-100), and age, ideal for finding natural consumer groups.',
    rows: 140,
    columns: 4,
    recommendedAlgorithms: ['kmeans', 'pca'],
    challenge: 'Use K-Means clustering with K=4 to discover the 4 behavioral customer personas and inspect the Elbow Method plot.',
    features: [
      { name: 'age', type: 'numeric', description: 'Customer age in years', min: 18, max: 65, mean: 39.5, missingCount: 0 },
      { name: 'annual_income_k', type: 'numeric', description: 'Annual income in thousands of dollars ($k)', min: 15, max: 110, mean: 61.2, missingCount: 0 },
      { name: 'spending_score', type: 'numeric', description: 'Propensity to spend score assigned by mall (1-100)', min: 1, max: 99, mean: 50.4, missingCount: 0 },
      { name: 'visit_frequency', type: 'numeric', description: 'Average monthly shopping visits', min: 1, max: 14, mean: 7.2, missingCount: 0 },
    ],
    data: generateCustomerData(),
  },
  {
    id: 'breast_cancer',
    name: 'Breast Cancer Wisconsin (Diagnostic)',
    taskType: 'classification',
    difficulty: 'Intermediate',
    description: 'Clinical characteristics computed from digitized fine needle aspirate (FNA) images of breast masses, predicting Benign vs Malignant.',
    disclaimer: 'Educational & research dataset only. Not intended for direct clinical or medical decision making.',
    rows: 130,
    columns: 7,
    targetColumn: 'diagnosis',
    targetClasses: ['Benign', 'Malignant'],
    recommendedAlgorithms: ['logistic_regression', 'svm', 'random_forest', 'decision_tree'],
    challenge: 'Train Logistic Regression or SVM to maximize Recall for the Malignant class (minimizing false negatives in clinical diagnostics).',
    features: [
      { name: 'radius_mean', type: 'numeric', description: 'Mean distances from center to points on the perimeter', min: 8.5, max: 24.5, mean: 14.3, missingCount: 0 },
      { name: 'texture_mean', type: 'numeric', description: 'Standard deviation of gray-scale values', min: 10.2, max: 30.1, mean: 19.5, missingCount: 0 },
      { name: 'perimeter_mean', type: 'numeric', description: 'Mean size of the core tumor perimeter', min: 55.0, max: 165.0, mean: 93.8, missingCount: 0 },
      { name: 'area_mean', type: 'numeric', description: 'Mean tumor area in square millimeters', min: 220.0, max: 1750.0, mean: 680.5, missingCount: 0 },
      { name: 'smoothness_mean', type: 'numeric', description: 'Local variation in radius lengths', min: 0.065, max: 0.135, mean: 0.096, missingCount: 0 },
      { name: 'concavity_mean', type: 'numeric', description: 'Severity of concave portions of the contour', min: 0.0, max: 0.35, mean: 0.094, missingCount: 0 },
      { name: 'diagnosis', type: 'categorical', description: 'Diagnosis of cell mass', categories: ['Benign', 'Malignant'], missingCount: 0 },
    ],
    data: generateBreastCancerData(),
  },
  {
    id: 'wine',
    name: 'Wine Recognition',
    taskType: 'classification',
    difficulty: 'Intermediate',
    description: 'Chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.',
    rows: 135,
    columns: 6,
    targetColumn: 'cultivar',
    targetClasses: ['Class 1', 'Class 2', 'Class 3'],
    recommendedAlgorithms: ['random_forest', 'decision_tree', 'knn', 'pca'],
    challenge: 'Compare Random Forest vs Decision Tree on these correlated chemical attributes, and inspect feature importance.',
    features: [
      { name: 'alcohol', type: 'numeric', description: 'Alcohol percentage by volume (% vol)', min: 11.0, max: 14.8, mean: 13.0, missingCount: 0 },
      { name: 'malic_acid', type: 'numeric', description: 'Malic acid concentration (g/L)', min: 0.8, max: 4.8, mean: 2.3, missingCount: 0 },
      { name: 'flavanoids', type: 'numeric', description: 'Total flavanoid content (g/L)', min: 0.3, max: 3.9, mean: 2.0, missingCount: 0 },
      { name: 'color_intensity', type: 'numeric', description: 'Color absorbance intensity', min: 1.5, max: 9.5, mean: 4.9, missingCount: 0 },
      { name: 'proline', type: 'numeric', description: 'Proline amino acid content (mg/L)', min: 310, max: 1450, mean: 750, missingCount: 0 },
      { name: 'cultivar', type: 'categorical', description: 'Cultivar class', categories: ['Class 1', 'Class 2', 'Class 3'], missingCount: 0 },
    ],
    data: generateWineData(),
  },
  {
    id: 'california_housing',
    name: 'Housing Valuation',
    taskType: 'regression',
    difficulty: 'Intermediate',
    description: 'Regional housing features including median household income, average rooms, and house age predicting property values.',
    rows: 120,
    columns: 5,
    targetColumn: 'median_house_value',
    recommendedAlgorithms: ['linear_regression', 'random_forest', 'decision_tree'],
    challenge: 'Train Linear Regression and Random Forest to predict home prices, and compare MAE vs RMSE.',
    features: [
      { name: 'median_income', type: 'numeric', description: 'Median household income ($10k)', min: 1.5, max: 10.0, mean: 5.7, missingCount: 0 },
      { name: 'house_age', type: 'numeric', description: 'Median age of house in years', min: 5, max: 50, mean: 27.5, missingCount: 0 },
      { name: 'ave_rooms', type: 'numeric', description: 'Average number of rooms per home', min: 3.5, max: 8.0, mean: 5.7, missingCount: 0 },
      { name: 'population', type: 'numeric', description: 'Block population count', min: 400, max: 3000, mean: 1700, missingCount: 0 },
      { name: 'median_house_value', type: 'numeric', description: 'Median house market value ($)', min: 50000, max: 450000, mean: 240000, missingCount: 0 },
    ],
    data: generateHousingData(),
  },
  {
    id: 'diabetes',
    name: 'Diabetes Progression',
    taskType: 'regression',
    difficulty: 'Intermediate',
    description: 'Ten baseline variables (age, BMI, blood pressure, blood serums) predicting quantitative disease progression after one year.',
    rows: 110,
    columns: 6,
    targetColumn: 'progression',
    recommendedAlgorithms: ['linear_regression', 'random_forest'],
    challenge: 'Evaluate whether feature scaling (StandardScaler) improves Linear Regression performance on disparate physiological units.',
    features: [
      { name: 'age', type: 'numeric', description: 'Patient age in years', min: 20, max: 75, mean: 45.0, missingCount: 0 },
      { name: 'bmi', type: 'numeric', description: 'Body Mass Index (kg/m²)', min: 18.0, max: 38.0, mean: 26.3, missingCount: 0 },
      { name: 'bp', type: 'numeric', description: 'Average arterial blood pressure (mm Hg)', min: 70.0, max: 125.0, mean: 94.0, missingCount: 0 },
      { name: 'glucose', type: 'numeric', description: 'Blood sugar fasting level', min: 65.0, max: 120.0, mean: 91.0, missingCount: 0 },
      { name: 's5_ltg', type: 'numeric', description: 'Log serum triglycerides', min: 3.2, max: 6.0, mean: 4.6, missingCount: 0 },
      { name: 'progression', type: 'numeric', description: 'Quantitative measure of disease progression', min: 30, max: 340, mean: 152, missingCount: 0 },
    ],
    data: generateDiabetesData(),
  },
  {
    id: 'synthetic_binary',
    name: 'Exam Result Predictor (Binary)',
    taskType: 'classification',
    difficulty: 'Beginner',
    description: 'Clean two-feature classification problem with Pass / Fail outcomes, ideal for visualizing logistic regression decision boundaries.',
    rows: 120,
    columns: 3,
    targetColumn: 'label',
    targetClasses: ['Pass', 'Fail'],
    recommendedAlgorithms: ['logistic_regression', 'svm', 'knn', 'decision_tree'],
    challenge: 'Observe how adjusting the probability threshold (from 0.5 to 0.3 or 0.7) trades off Precision versus Recall in the Confusion Matrix.',
    features: [
      { name: 'feature_1', type: 'numeric', description: 'Attendance score', min: 0.5, max: 8.5, mean: 4.3, missingCount: 0 },
      { name: 'feature_2', type: 'numeric', description: 'Assignment score', min: 0.5, max: 8.5, mean: 4.3, missingCount: 0 },
      { name: 'label', type: 'categorical', description: 'Course completion status', categories: ['Pass', 'Fail'], missingCount: 0 },
    ],
    data: generateSyntheticBinary(),
  },
  {
    id: 'synthetic_blobs',
    name: 'Synthetic Blobs',
    taskType: 'clustering',
    difficulty: 'Beginner',
    description: 'Three distinct 2D Gaussian point clusters designed for visualizing K-Means centroid convergence and Voronoi cell partitioning.',
    rows: 120,
    columns: 3,
    recommendedAlgorithms: ['kmeans', 'pca'],
    challenge: 'Run K-Means with step-by-step playback to watch centroids migrate from random initialization to cluster centers.',
    features: [
      { name: 'x', type: 'numeric', description: 'X spatial coordinate', min: 0.2, max: 10.5, mean: 6.1, missingCount: 0 },
      { name: 'y', type: 'numeric', description: 'Y spatial coordinate', min: 0.5, max: 10.2, mean: 4.5, missingCount: 0 },
      { name: 'ground_truth_cluster', type: 'categorical', description: 'True cluster membership for validation', categories: ['Cluster 1', 'Cluster 2', 'Cluster 3'], missingCount: 0 },
    ],
    data: generateSyntheticBlobs(),
  },
  {
    id: 'two_moons',
    name: 'Two Interlocking Moons',
    taskType: 'classification',
    difficulty: 'Advanced',
    description: 'A classic non-linear classification benchmark consisting of two interleaving crescent moons. Linear classifiers fail here!',
    rows: 120,
    columns: 3,
    targetColumn: 'moon',
    targetClasses: ['Upper', 'Lower'],
    recommendedAlgorithms: ['decision_tree', 'random_forest', 'knn', 'svm'],
    challenge: 'Demonstrate why linear models (Logistic Regression) achieve low accuracy (<80%) while Decision Trees or KNN easily hit >95%.',
    features: [
      { name: 'x', type: 'numeric', description: 'Horizontal plane coordinate', min: -1.2, max: 2.2, mean: 0.5, missingCount: 0 },
      { name: 'y', type: 'numeric', description: 'Vertical plane coordinate', min: -0.8, max: 1.2, mean: 0.2, missingCount: 0 },
      { name: 'moon', type: 'categorical', description: 'Crescent class', categories: ['Upper', 'Lower'], missingCount: 0 },
    ],
    data: generateTwoMoons(),
  },
];
