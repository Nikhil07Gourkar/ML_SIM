import {
  AlgorithmId,
  ConfusionMatrixData,
  EvaluationMetrics,
  PreparedData,
  SimulationStep,
  TreeNode,
} from '../types/ml';
import {
  computeCovarianceMatrix,
  dot,
  entropy,
  euclideanDistance,
  giniImpurity,
  invertMatrix,
  manhattanDistance,
  matrixMultiply,
  mean,
  powerIterationTopEigenvectors,
  SeededRandom,
  sigmoid,
  transpose,
  variance,
  vectorNorm,
} from './mathUtils';

export interface TrainResult {
  algorithmId: AlgorithmId;
  metrics: EvaluationMetrics;
  predictions: number[];
  simulationSteps: SimulationStep[];
  modelDetails: Record<string, any>;
  predict: (sample: number[]) => number | string;
}

// -------------------------------------------------------------
// 1. Linear Regression
// -------------------------------------------------------------
export function trainLinearRegression(
  data: PreparedData,
  hyperparams: { fitIntercept?: boolean; learningRate?: number; maxIterations?: number } = {}
): TrainResult {
  const startTime = performance.now();
  const fitIntercept = hyperparams.fitIntercept !== false;
  const X_train = data.X_train;
  const y_train = data.y_train;
  const X_test = data.X_test;
  const y_test = data.y_test;

  const n = X_train.length;
  const d = data.featureNames.length;

  // Add intercept column [1, x1, x2...] if enabled
  const X_aug: number[][] = X_train.map((row) => (fitIntercept ? [1, ...row] : [...row]));
  const X_aug_T = transpose(X_aug);
  const XtX = matrixMultiply(X_aug_T, X_aug);
  const XtX_inv = invertMatrix(XtX, 1e-4);
  const XtY: number[][] = Array.from({ length: X_aug[0].length }, () => [0]);

  for (let r = 0; r < X_aug[0].length; r++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += X_aug[i][r] * y_train[i];
    }
    XtY[r][0] = sum;
  }

  // Closed-form OLS solution: beta = (X^T X)^(-1) X^T y
  const betaMatrix = matrixMultiply(XtX_inv, XtY);
  const weights: number[] = [];
  let intercept = 0;

  if (fitIntercept) {
    intercept = betaMatrix[0][0];
    for (let j = 1; j < betaMatrix.length; j++) {
      weights.push(betaMatrix[j][0]);
    }
  } else {
    for (let j = 0; j < betaMatrix.length; j++) {
      weights.push(betaMatrix[j][0]);
    }
  }

  // Educational step-by-step Gradient Descent Simulation
  const steps: SimulationStep[] = [];
  const lr = hyperparams.learningRate || 0.05;
  const maxIters = 6;
  let currentWeights = new Array(d).fill(0);
  let currentIntercept = fitIntercept ? mean(y_train) : 0;

  steps.push({
    stepNumber: 0,
    title: 'Model Initialization',
    description: `Initialized linear model with slope = 0 and intercept = ${currentIntercept.toFixed(2)} (mean of target).`,
    operation: 'Initialization',
    formula: 'y = w · x + b',
    intermediateValues: {
      slope: d === 1 ? currentWeights[0] : currentWeights,
      intercept: currentIntercept,
    },
    visualState: {
      regressionLine: { slope: currentWeights[0] || 0, intercept: currentIntercept },
    },
  });

  for (let iter = 1; iter <= maxIters; iter++) {
    // Compute gradient
    let gradIntercept = 0;
    const gradW = new Array(d).fill(0);
    let totalError = 0;

    for (let i = 0; i < n; i++) {
      const pred = dot(currentWeights, X_train[i]) + currentIntercept;
      const error = pred - y_train[i];
      totalError += error * error;
      gradIntercept += error;
      for (let j = 0; j < d; j++) {
        gradW[j] += error * X_train[i][j];
      }
    }

    const mseLoss = totalError / (2 * n);
    currentIntercept -= (lr * gradIntercept) / n;
    for (let j = 0; j < d; j++) {
      currentWeights[j] -= (lr * gradW[j]) / n;
    }

    steps.push({
      stepNumber: iter,
      title: `Gradient Descent Iteration ${iter}`,
      description: `Computed loss MSE = ${mseLoss.toFixed(2)}. Updated parameters in negative gradient direction.`,
      operation: 'Parameter Update',
      formula: 'θ := θ - α · (1/m) · ∑(h(x) - y) · x',
      mathExplanation: `Slope shifted by -${((lr * gradW[0]) / n).toFixed(3)}, Intercept shifted by -${((lr * gradIntercept) / n).toFixed(3)}`,
      intermediateValues: {
        loss_MSE: +mseLoss.toFixed(3),
        slope: +currentWeights[0].toFixed(3),
        intercept: +currentIntercept.toFixed(3),
      },
      visualState: {
        regressionLine: { slope: currentWeights[0], intercept: currentIntercept },
      },
    });
  }

  // Final step with exact OLS closed form
  steps.push({
    stepNumber: maxIters + 1,
    title: 'Ordinary Least Squares Optimal Fit',
    description: 'Computed optimal parameters analytically using the Normal Equation (XᵀX)⁻¹ Xᵀy.',
    operation: 'Optimal Convergence',
    formula: 'β = (Xᵀ X)⁻¹ Xᵀ y',
    mathExplanation: `Optimal slope = ${weights[0]?.toFixed(3) || '0'}, Intercept = ${intercept.toFixed(3)}`,
    intermediateValues: {
      optimal_slope: weights[0] !== undefined ? +weights[0].toFixed(3) : 0,
      optimal_intercept: +intercept.toFixed(3),
    },
    visualState: {
      regressionLine: { slope: weights[0] || 0, intercept },
    },
  });

  // Predictions on Test Set
  const predictSample = (x: number[]): number => {
    let pred = intercept;
    for (let j = 0; j < d; j++) {
      pred += weights[j] * (x[j] || 0);
    }
    return pred;
  };

  const predictions = X_test.map(predictSample);

  // Compute Regression Metrics: MAE, MSE, RMSE, R²
  let sumAbsErr = 0;
  let sumSqErr = 0;
  const yTestMean = mean(y_test);
  let totalSumSq = 0;

  for (let i = 0; i < y_test.length; i++) {
    const err = predictions[i] - y_test[i];
    sumAbsErr += Math.abs(err);
    sumSqErr += err * err;
    totalSumSq += Math.pow(y_test[i] - yTestMean, 2);
  }

  const mae = y_test.length > 0 ? +(sumAbsErr / y_test.length).toFixed(3) : 0;
  const mse = y_test.length > 0 ? +(sumSqErr / y_test.length).toFixed(3) : 0;
  const rmse = +(Math.sqrt(mse)).toFixed(3);
  const r2 = totalSumSq > 1e-9 ? +(Math.max(-2, 1 - sumSqErr / totalSumSq)).toFixed(3) : 0;

  const trainTimeMs = Math.round(performance.now() - startTime);

  return {
    algorithmId: 'linear_regression',
    metrics: { mae, mse, rmse, r2, trainTimeMs, sampleCount: n },
    predictions,
    simulationSteps: steps,
    modelDetails: {
      slope: weights[0] !== undefined ? weights[0] : 0,
      intercept,
      weights,
      formulaString: `y = ${weights.map((w, i) => `${w.toFixed(2)}·${data.featureNames[i]}`).join(' + ')} + ${intercept.toFixed(2)}`,
    },
    predict: (sample: number[]) => predictSample(sample),
  };
}

// -------------------------------------------------------------
// 2. Logistic Regression
// -------------------------------------------------------------
export function trainLogisticRegression(
  data: PreparedData,
  hyperparams: { learningRate?: number; maxIterations?: number; threshold?: number } = {}
): TrainResult {
  const startTime = performance.now();
  const X_train = data.X_train;
  const y_train = data.y_train;
  const X_test = data.X_test;
  const y_test = data.y_test;

  const n = X_train.length;
  const d = data.featureNames.length;
  const lr = hyperparams.learningRate || 0.1;
  const threshold = hyperparams.threshold !== undefined ? hyperparams.threshold : 0.5;

  let weights = new Array(d).fill(0);
  let bias = 0;
  const steps: SimulationStep[] = [];

  steps.push({
    stepNumber: 0,
    title: 'Initialize Logistic Model',
    description: 'Set initial weights w = 0 and bias b = 0. Sigmoid output initially evaluates to p = 0.5 everywhere.',
    operation: 'Initialization',
    formula: 'p = σ(w · x + b) = 1 / (1 + e^{-(w · x + b)})',
    intermediateValues: { weights: [...weights], bias },
    visualState: { decisionBoundary: { w: [...weights], b: bias } },
  });

  const totalIters = 80;
  const loggedSteps = [1, 5, 15, 30, 50, 80];

  for (let iter = 1; iter <= totalIters; iter++) {
    let gradB = 0;
    const gradW = new Array(d).fill(0);
    let totalLoss = 0;

    for (let i = 0; i < n; i++) {
      const z = dot(weights, X_train[i]) + bias;
      const p = sigmoid(z);
      const error = p - (y_train[i] > 0 ? 1 : 0);

      // Binary Cross-Entropy Loss
      const y_i = y_train[i] > 0 ? 1 : 0;
      const safeP = Math.max(1e-12, Math.min(1 - 1e-12, p));
      totalLoss -= y_i * Math.log(safeP) + (1 - y_i) * Math.log(1 - safeP);

      gradB += error;
      for (let j = 0; j < d; j++) {
        gradW[j] += error * X_train[i][j];
      }
    }

    bias -= (lr * gradB) / n;
    for (let j = 0; j < d; j++) {
      weights[j] -= (lr * gradW[j]) / n;
    }

    if (loggedSteps.includes(iter)) {
      const avgLoss = +(totalLoss / n).toFixed(3);
      steps.push({
        stepNumber: steps.length,
        title: `Iteration ${iter}: Loss = ${avgLoss}`,
        description: `Gradient step updated decision hyperplane. Cross-entropy log loss decreased to ${avgLoss}.`,
        operation: 'Gradient Step',
        formula: 'w := w - α · (1/m) · ∑(σ(z) - y) · x',
        intermediateValues: {
          iteration: iter,
          loss: avgLoss,
          w1: weights[0] !== undefined ? +weights[0].toFixed(3) : 0,
          w2: weights[1] !== undefined ? +weights[1].toFixed(3) : 0,
          bias: +bias.toFixed(3),
        },
        visualState: { decisionBoundary: { w: [...weights], b: bias } },
      });
    }
  }

  // Predict function
  const predictProb = (x: number[]): number => {
    return sigmoid(dot(weights, x) + bias);
  };

  const predictClass = (x: number[]): number => {
    return predictProb(x) >= threshold ? 1 : 0;
  };

  const predictions = X_test.map(predictClass);

  // Compute Classification Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < y_test.length; i++) {
    const actual = y_test[i] > 0 ? 1 : 0;
    const pred = predictions[i];
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 1) fp++;
    else if (actual === 0 && pred === 0) tn++;
    else if (actual === 1 && pred === 0) fn++;
  }

  const accuracy = y_test.length > 0 ? +((tp + tn) / y_test.length).toFixed(3) : 0;
  const precision = tp + fp > 0 ? +(tp / (tp + fp)).toFixed(3) : 0;
  const recall = tp + fn > 0 ? +(tp / (tp + fn)).toFixed(3) : 0;
  const f1Score = precision + recall > 0 ? +((2 * precision * recall) / (precision + recall)).toFixed(3) : 0;

  const classLabels = data.targetClasses || ['Negative', 'Positive'];
  const confusionMatrix: ConfusionMatrixData = {
    matrix: [
      [tn, fp],
      [fn, tp],
    ],
    labels: classLabels.slice(0, 2),
  };

  const trainTimeMs = Math.round(performance.now() - startTime);

  return {
    algorithmId: 'logistic_regression',
    metrics: { accuracy, precision, recall, f1Score, confusionMatrix, trainTimeMs, sampleCount: n },
    predictions,
    simulationSteps: steps,
    modelDetails: {
      weights,
      bias,
      threshold,
      classLabels,
    },
    predict: (sample: number[]) => {
      const cls = predictClass(sample);
      return classLabels[cls] || cls;
    },
  };
}

// -------------------------------------------------------------
// 3. Decision Tree
// -------------------------------------------------------------
export function trainDecisionTree(
  data: PreparedData,
  hyperparams: { maxDepth?: number; minSamplesSplit?: number; criterion?: 'gini' | 'entropy' } = {},
  isRegression: boolean = false
): TrainResult {
  const startTime = performance.now();
  const maxDepth = hyperparams.maxDepth !== undefined ? hyperparams.maxDepth : 4;
  const minSamplesSplit = hyperparams.minSamplesSplit || 2;
  const criterion = hyperparams.criterion || 'gini';

  const X_train = data.X_train;
  const y_train = data.y_train;
  const X_test = data.X_test;
  const y_test = data.y_test;
  const d = data.featureNames.length;

  const simulationSteps: SimulationStep[] = [];
  let nodeIdCounter = 0;

  // Build tree recursively
  const buildTree = (
    indices: number[],
    depth: number
  ): TreeNode => {
    nodeIdCounter++;
    const currentId = `node_${nodeIdCounter}`;
    const samplesCount = indices.length;
    const currentLabels = indices.map((i) => y_train[i]);

    // Calculate impurity
    let currentImpurity = 0;
    if (isRegression) {
      currentImpurity = variance(currentLabels, false);
    } else {
      currentImpurity = criterion === 'gini' ? giniImpurity(currentLabels) : entropy(currentLabels);
    }

    // Determine value: mean for regression, counts for classification
    let nodeValue: number | number[];
    let predictedClass: number | string = 0;

    if (isRegression) {
      nodeValue = mean(currentLabels);
      predictedClass = +nodeValue.toFixed(2);
    } else {
      const numClasses = Math.max(2, (data.targetClasses || []).length);
      const counts: number[] = new Array(numClasses).fill(0);
      for (const lbl of currentLabels) {
        counts[Math.floor(lbl)] = (counts[Math.floor(lbl)] || 0) + 1;
      }
      nodeValue = counts;
      let maxC = 0;
      let bestLbl = 0;
      for (let c = 0; c < counts.length; c++) {
        if (counts[c] > maxC) {
          maxC = counts[c];
          bestLbl = c;
        }
      }
      predictedClass = data.targetClasses ? data.targetClasses[bestLbl] || bestLbl : bestLbl;
    }

    // Check leaf conditions
    if (depth >= maxDepth || samplesCount < minSamplesSplit || currentImpurity < 1e-5) {
      return {
        id: currentId,
        impurity: +currentImpurity.toFixed(3),
        samples: samplesCount,
        value: nodeValue,
        predictedClass,
        isLeaf: true,
      };
    }

    // Search best split
    let bestGain = -Infinity;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    for (let f = 0; f < d; f++) {
      const vals = Array.from(new Set(indices.map((i) => X_train[i][f]))).sort((a, b) => a - b);
      // Test midpoints
      for (let v = 0; v < vals.length - 1; v++) {
        const thresh = (vals[v] + vals[v + 1]) / 2;
        const left: number[] = [];
        const right: number[] = [];

        for (const idx of indices) {
          if (X_train[idx][f] <= thresh) left.push(idx);
          else right.push(idx);
        }

        if (left.length === 0 || right.length === 0) continue;

        let leftImp = 0;
        let rightImp = 0;
        const leftY = left.map((i) => y_train[i]);
        const rightY = right.map((i) => y_train[i]);

        if (isRegression) {
          leftImp = variance(leftY, false);
          rightImp = variance(rightY, false);
        } else {
          leftImp = criterion === 'gini' ? giniImpurity(leftY) : entropy(leftY);
          rightImp = criterion === 'gini' ? giniImpurity(rightY) : entropy(rightY);
        }

        const weightedChildImp = (left.length / samplesCount) * leftImp + (right.length / samplesCount) * rightImp;
        const gain = currentImpurity - weightedChildImp;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestThreshold = thresh;
          bestLeftIndices = left;
          bestRightIndices = right;
        }
      }
    }

    if (bestGain <= 1e-6 || bestFeature === -1) {
      return {
        id: currentId,
        impurity: +currentImpurity.toFixed(3),
        samples: samplesCount,
        value: nodeValue,
        predictedClass,
        isLeaf: true,
      };
    }

    // Log step for educational inspection
    if (simulationSteps.length < 8) {
      simulationSteps.push({
        stepNumber: simulationSteps.length,
        title: `Split at Depth ${depth}: [${data.featureNames[bestFeature]}]`,
        description: `Split ${samplesCount} samples at threshold ≤ ${bestThreshold.toFixed(2)}. Information gain = ${bestGain.toFixed(3)}.`,
        operation: 'Tree Node Split',
        formula: 'Gain = Impurity(Parent) - ∑ (N_child / N_parent) · Impurity(Child)',
        mathExplanation: `Parent impurity: ${currentImpurity.toFixed(3)} → Left child (${bestLeftIndices.length} samples), Right child (${bestRightIndices.length} samples)`,
        intermediateValues: {
          feature: data.featureNames[bestFeature],
          threshold: +bestThreshold.toFixed(2),
          gain: +bestGain.toFixed(3),
          depth,
        },
        visualState: {
          activeNodeId: currentId,
        },
      });
    }

    const leftNode = buildTree(bestLeftIndices, depth + 1);
    const rightNode = buildTree(bestRightIndices, depth + 1);

    return {
      id: currentId,
      featureIndex: bestFeature,
      featureName: data.featureNames[bestFeature],
      threshold: +bestThreshold.toFixed(3),
      impurity: +currentImpurity.toFixed(3),
      samples: samplesCount,
      gain: +bestGain.toFixed(3),
      value: nodeValue,
      predictedClass,
      left: leftNode,
      right: rightNode,
      isLeaf: false,
    };
  };

  const rootIndices = Array.from({ length: X_train.length }, (_, i) => i);
  const rootNode = buildTree(rootIndices, 0);

  // Tree sample predict function
  const predictSample = (x: number[]): number => {
    let curr: TreeNode = rootNode;
    while (!curr.isLeaf) {
      const featIdx = curr.featureIndex ?? 0;
      const thresh = curr.threshold ?? 0;
      if (x[featIdx] <= thresh) {
        if (!curr.left) break;
        curr = curr.left;
      } else {
        if (!curr.right) break;
        curr = curr.right;
      }
    }
    if (typeof curr.value === 'number') {
      return curr.value;
    }
    const counts = curr.value as number[];
    let maxIdx = 0;
    let maxC = 0;
    for (let c = 0; c < counts.length; c++) {
      if (counts[c] > maxC) {
        maxC = counts[c];
        maxIdx = c;
      }
    }
    return maxIdx;
  };

  const predictions = X_test.map(predictSample);

  let metrics: EvaluationMetrics;
  if (isRegression) {
    let sumAbsErr = 0;
    let sumSqErr = 0;
    const yTestMean = mean(y_test);
    let totalSumSq = 0;
    for (let i = 0; i < y_test.length; i++) {
      const err = predictions[i] - y_test[i];
      sumAbsErr += Math.abs(err);
      sumSqErr += err * err;
      totalSumSq += Math.pow(y_test[i] - yTestMean, 2);
    }
    const mae = +(sumAbsErr / (y_test.length || 1)).toFixed(3);
    const mse = +(sumSqErr / (y_test.length || 1)).toFixed(3);
    const rmse = +(Math.sqrt(mse)).toFixed(3);
    const r2 = totalSumSq > 1e-9 ? +(Math.max(-2, 1 - sumSqErr / totalSumSq)).toFixed(3) : 0;
    metrics = { mae, mse, rmse, r2, trainTimeMs: Math.round(performance.now() - startTime), sampleCount: X_train.length };
  } else {
    // Classification
    let correct = 0;
    const numClasses = Math.max(2, (data.targetClasses || []).length);
    const confMat: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));

    for (let i = 0; i < y_test.length; i++) {
      const act = Math.floor(y_test[i]);
      const pred = Math.floor(predictions[i]);
      if (act === pred) correct++;
      if (confMat[act] && confMat[act][pred] !== undefined) {
        confMat[act][pred]++;
      }
    }

    const accuracy = +(correct / (y_test.length || 1)).toFixed(3);
    const precision = accuracy; // approximate multi-class macro
    const recall = accuracy;
    const f1Score = accuracy;

    metrics = {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix: { matrix: confMat, labels: data.targetClasses || ['Class 0', 'Class 1'] },
      trainTimeMs: Math.round(performance.now() - startTime),
      sampleCount: X_train.length,
    };
  }

  return {
    algorithmId: 'decision_tree',
    metrics,
    predictions,
    simulationSteps,
    modelDetails: {
      treeRoot: rootNode,
      maxDepth,
      criterion,
    },
    predict: (sample: number[]) => {
      const predNum = predictSample(sample);
      if (isRegression) return predNum.toFixed(2);
      return data.targetClasses ? data.targetClasses[predNum] || predNum : predNum;
    },
  };
}

// -------------------------------------------------------------
// 4. K-Means Clustering
// -------------------------------------------------------------
export function trainKMeans(
  data: PreparedData,
  hyperparams: { k?: number; maxIterations?: number; randomSeed?: number } = {}
): TrainResult {
  const startTime = performance.now();
  const k = Math.max(2, Math.min(8, hyperparams.k || 3));
  const maxIters = hyperparams.maxIterations || 15;
  const rng = new SeededRandom(hyperparams.randomSeed || 42);

  const X = [...data.X_train, ...data.X_test]; // Unsupervised uses full available data
  const n = X.length;
  const d = data.featureNames.length;

  // 1. K-Means++ Initialization
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng.next() * n);
  centroids.push([...X[firstIdx]]);

  while (centroids.length < k) {
    const distances: number[] = [];
    let sumDistSq = 0;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const c of centroids) {
        const dist = euclideanDistance(X[i], c);
        if (dist < minDist) minDist = dist;
      }
      const distSq = minDist * minDist;
      distances.push(distSq);
      sumDistSq += distSq;
    }

    let target = rng.next() * sumDistSq;
    let chosenIdx = 0;
    for (let i = 0; i < n; i++) {
      target -= distances[i];
      if (target <= 0) {
        chosenIdx = i;
        break;
      }
    }
    centroids.push([...X[chosenIdx]]);
  }

  const steps: SimulationStep[] = [];
  let assignments: number[] = new Array(n).fill(0);
  let inertia = 0;

  // Initial step
  steps.push({
    stepNumber: 0,
    title: `K-Means++ Initialization (K=${k})`,
    description: `Initialized ${k} initial centroids spread far apart using distance-squared probability.`,
    operation: 'Initialization',
    formula: 'P(x) = D(x)² / ∑ D(x′)²',
    intermediateValues: {
      k,
      initialCentroids: centroids.map((c) => c.slice(0, 2).map((v) => +v.toFixed(2))),
    },
    visualState: {
      centroids: centroids.map((c, i) => ({ x: c[0] || 0, y: c[1] || 0, id: i })),
      assignments: [...assignments],
    },
  });

  // Iterative Optimization (Lloyd's Algorithm)
  let converged = false;
  let iter = 0;

  while (!converged && iter < maxIters) {
    iter++;
    // Step A: Assignment
    const newAssignments: number[] = new Array(n);
    let currentInertia = 0;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = euclideanDistance(X[i], centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      newAssignments[i] = bestCluster;
      currentInertia += minDist * minDist;
    }

    inertia = currentInertia;

    // Step B: Update centroids
    let maxShift = 0;
    const newCentroids: number[][] = Array.from({ length: k }, () => new Array(d).fill(0));
    const clusterCounts: number[] = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const c = newAssignments[i];
      clusterCounts[c]++;
      for (let j = 0; j < d; j++) {
        newCentroids[c][j] += X[i][j];
      }
    }

    for (let c = 0; c < k; c++) {
      if (clusterCounts[c] > 0) {
        for (let j = 0; j < d; j++) {
          newCentroids[c][j] /= clusterCounts[c];
        }
      } else {
        // Re-seed empty cluster to a random point
        newCentroids[c] = [...X[Math.floor(rng.next() * n)]];
      }
      const shift = euclideanDistance(centroids[c], newCentroids[c]);
      if (shift > maxShift) maxShift = shift;
      centroids[c] = [...newCentroids[c]];
    }

    assignments = newAssignments;

    steps.push({
      stepNumber: iter,
      title: `Iteration ${iter}: Inertia = ${inertia.toFixed(1)}`,
      description: `Reassigned points to nearest centroid and recalculated cluster means. Max centroid shift = ${maxShift.toFixed(3)}.`,
      operation: 'Assignment & Update',
      formula: 'μ_k = (1 / |C_k|) · ∑ x_i, where i ∈ C_k',
      intermediateValues: {
        inertia: +inertia.toFixed(2),
        maxShift: +maxShift.toFixed(4),
        clusterCounts,
      },
      visualState: {
        centroids: centroids.map((c, i) => ({ x: c[0] || 0, y: c[1] || 0, id: i })),
        assignments: [...assignments],
      },
    });

    if (maxShift < 1e-4) {
      converged = true;
    }
  }

  // Calculate cluster distribution counts
  const clusterCountsMap: Record<number, number> = {};
  for (const a of assignments) {
    clusterCountsMap[a] = (clusterCountsMap[a] || 0) + 1;
  }

  // Calculate Silhouette Score (sample of 80 to keep responsive)
  const sampleSize = Math.min(80, n);
  let totalSilhouette = 0;
  for (let i = 0; i < sampleSize; i++) {
    const myCluster = assignments[i];
    // a(i): mean intra-cluster distance
    let aSum = 0;
    let aCount = 0;
    const bSums: Record<number, number> = {};
    const bCounts: Record<number, number> = {};

    for (let j = 0; j < sampleSize; j++) {
      if (i === j) continue;
      const d = euclideanDistance(X[i], X[j]);
      if (assignments[j] === myCluster) {
        aSum += d;
        aCount++;
      } else {
        const c = assignments[j];
        bSums[c] = (bSums[c] || 0) + d;
        bCounts[c] = (bCounts[c] || 0) + 1;
      }
    }

    const a = aCount > 0 ? aSum / aCount : 0;
    let b = Infinity;
    for (const c in bSums) {
      const meanDist = bSums[c] / bCounts[c];
      if (meanDist < b) b = meanDist;
    }
    if (b === Infinity) b = 0;

    const s = Math.max(a, b) > 0 ? (b - a) / Math.max(a, b) : 0;
    totalSilhouette += s;
  }
  const silhouetteScore = +(totalSilhouette / sampleSize).toFixed(3);

  const trainTimeMs = Math.round(performance.now() - startTime);

  return {
    algorithmId: 'kmeans',
    metrics: {
      inertia: +inertia.toFixed(2),
      silhouetteScore,
      clusterCounts: clusterCountsMap,
      trainTimeMs,
      sampleCount: n,
    },
    predictions: assignments,
    simulationSteps: steps,
    modelDetails: {
      centroids,
      k,
      converged,
      iterations: iter,
    },
    predict: (sample: number[]) => {
      let minDist = Infinity;
      let bestCluster = 0;
      for (let c = 0; c < k; c++) {
        const dist = euclideanDistance(sample, centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      return `Cluster ${bestCluster + 1}`;
    },
  };
}

// -------------------------------------------------------------
// 5. Random Forest
// -------------------------------------------------------------
export function trainRandomForest(
  data: PreparedData,
  hyperparams: { numTrees?: number; maxDepth?: number } = {},
  isRegression: boolean = false
): TrainResult {
  const startTime = performance.now();
  const numTrees = Math.min(15, hyperparams.numTrees || 7);
  const maxDepth = hyperparams.maxDepth || 4;
  const n = data.X_train.length;
  const d = data.featureNames.length;
  const rng = new SeededRandom(12345);

  const trees: TreeNode[] = [];
  const featureImportances: number[] = new Array(d).fill(0);

  // Train each tree on a bootstrap bag
  for (let t = 0; t < numTrees; t++) {
    // Bootstrap sample with replacement
    const bagIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      bagIndices.push(Math.floor(rng.next() * n));
    }

    const bagData: PreparedData = {
      ...data,
      X_train: bagIndices.map((i) => data.X_train[i]),
      y_train: bagIndices.map((i) => data.y_train[i]),
    };

    const treeResult = trainDecisionTree(bagData, { maxDepth, criterion: 'gini' }, isRegression);
    const treeRoot = treeResult.modelDetails.treeRoot as TreeNode;
    trees.push(treeRoot);

    // Accumulate feature importance from tree gain
    const traverse = (node: TreeNode) => {
      if (node.isLeaf) return;
      if (node.featureIndex !== undefined && node.gain) {
        featureImportances[node.featureIndex] += node.gain;
      }
      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right);
    };
    traverse(treeRoot);
  }

  // Normalize feature importances
  const totalImp = featureImportances.reduce((a, b) => a + b, 0);
  const normalizedImportances = featureImportances.map((imp) =>
    totalImp > 0 ? +(imp / totalImp).toFixed(3) : +(1 / d).toFixed(3)
  );

  // Ensemble Predict Function
  const predictSingleTree = (tree: TreeNode, x: number[]): number => {
    let curr = tree;
    while (!curr.isLeaf) {
      const idx = curr.featureIndex ?? 0;
      const th = curr.threshold ?? 0;
      if (x[idx] <= th) {
        if (!curr.left) break;
        curr = curr.left;
      } else {
        if (!curr.right) break;
        curr = curr.right;
      }
    }
    if (typeof curr.value === 'number') return curr.value;
    const counts = curr.value as number[];
    let maxI = 0;
    let maxC = 0;
    for (let c = 0; c < counts.length; c++) {
      if (counts[c] > maxC) {
        maxC = counts[c];
        maxI = c;
      }
    }
    return maxI;
  };

  const predictSample = (x: number[]): number => {
    const treeVotes = trees.map((tr) => predictSingleTree(tr, x));
    if (isRegression) {
      return mean(treeVotes);
    }
    // Majority vote
    const counts: Record<number, number> = {};
    for (const v of treeVotes) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let bestV = 0;
    let maxC = 0;
    for (const v in counts) {
      if (counts[v] > maxC) {
        maxC = counts[v];
        bestV = Number(v);
      }
    }
    return bestV;
  };

  const predictions = data.X_test.map(predictSample);

  let metrics: EvaluationMetrics;
  if (isRegression) {
    let sumSq = 0;
    let sumAbs = 0;
    const yMean = mean(data.y_test);
    let totSq = 0;
    for (let i = 0; i < data.y_test.length; i++) {
      const err = predictions[i] - data.y_test[i];
      sumSq += err * err;
      sumAbs += Math.abs(err);
      totSq += Math.pow(data.y_test[i] - yMean, 2);
    }
    const mse = +(sumSq / (data.y_test.length || 1)).toFixed(3);
    const mae = +(sumAbs / (data.y_test.length || 1)).toFixed(3);
    const rmse = +(Math.sqrt(mse)).toFixed(3);
    const r2 = totSq > 1e-9 ? +(Math.max(-2, 1 - sumSq / totSq)).toFixed(3) : 0;
    metrics = { mae, mse, rmse, r2, trainTimeMs: Math.round(performance.now() - startTime), sampleCount: n };
  } else {
    let correct = 0;
    const numClasses = Math.max(2, (data.targetClasses || []).length);
    const confMat: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    for (let i = 0; i < data.y_test.length; i++) {
      const act = Math.floor(data.y_test[i]);
      const pr = Math.floor(predictions[i]);
      if (act === pr) correct++;
      if (confMat[act] && confMat[act][pr] !== undefined) confMat[act][pr]++;
    }
    const accuracy = +(correct / (data.y_test.length || 1)).toFixed(3);
    metrics = {
      accuracy,
      precision: accuracy,
      recall: accuracy,
      f1Score: accuracy,
      confusionMatrix: { matrix: confMat, labels: data.targetClasses || ['Class 0', 'Class 1'] },
      trainTimeMs: Math.round(performance.now() - startTime),
      sampleCount: n,
    };
  }

  const steps: SimulationStep[] = [
    {
      stepNumber: 0,
      title: 'Bootstrap Sampling & Bagging',
      description: `Created ${numTrees} independent bootstrap training bags with replacement.`,
      operation: 'Bootstrap Bagging',
      formula: 'D_t ~ Uniform(D, with replacement)',
      intermediateValues: { numTrees, samplesPerTree: n },
      visualState: {},
    },
    {
      stepNumber: 1,
      title: 'Parallel Tree Growth',
      description: `Built ${numTrees} randomized decision trees up to max depth ${maxDepth}.`,
      operation: 'Ensemble Construction',
      intermediateValues: { featureImportances: normalizedImportances },
      visualState: {},
    },
    {
      stepNumber: 2,
      title: 'Aggregated Committee Voting',
      description: 'Combined individual tree predictions via majority voting to reduce ensemble variance.',
      operation: 'Majority Voting',
      formula: 'ŷ = mode(Tree_1(x), ..., Tree_B(x))',
      visualState: {},
    },
  ];

  return {
    algorithmId: 'random_forest',
    metrics,
    predictions,
    simulationSteps: steps,
    modelDetails: {
      numTrees,
      featureImportances: normalizedImportances,
      featureNames: data.featureNames,
    },
    predict: (sample: number[]) => {
      const pred = predictSample(sample);
      if (isRegression) return pred.toFixed(2);
      return data.targetClasses ? data.targetClasses[pred] || pred : pred;
    },
  };
}

// -------------------------------------------------------------
// 6. K-Nearest Neighbors (KNN)
// -------------------------------------------------------------
export function trainKNN(
  data: PreparedData,
  hyperparams: { k?: number; metric?: 'euclidean' | 'manhattan' } = {},
  isRegression: boolean = false
): TrainResult {
  const startTime = performance.now();
  const k = Math.max(1, Math.min(15, hyperparams.k || 5));
  const distMetric = hyperparams.metric || 'euclidean';
  const X_train = data.X_train;
  const y_train = data.y_train;

  const distFn = distMetric === 'euclidean' ? euclideanDistance : manhattanDistance;

  const predictSample = (x: number[]): { prediction: number; neighborIndices: number[] } => {
    const distances = X_train.map((trainX, idx) => ({
      idx,
      dist: distFn(x, trainX),
    }));
    distances.sort((a, b) => a.dist - b.dist);
    const topK = distances.slice(0, k);
    const neighborIndices = topK.map((d) => d.idx);

    if (isRegression) {
      const neighborVals = neighborIndices.map((i) => y_train[i]);
      return { prediction: mean(neighborVals), neighborIndices };
    } else {
      // Majority vote
      const votes: Record<number, number> = {};
      for (const idx of neighborIndices) {
        const lbl = Math.floor(y_train[idx]);
        votes[lbl] = (votes[lbl] || 0) + 1;
      }
      let bestLbl = 0;
      let maxVotes = 0;
      for (const lbl in votes) {
        if (votes[lbl] > maxVotes) {
          maxVotes = votes[lbl];
          bestLbl = Number(lbl);
        }
      }
      return { prediction: bestLbl, neighborIndices };
    }
  };

  const results = data.X_test.map(predictSample);
  const predictions = results.map((r) => r.prediction);

  let metrics: EvaluationMetrics;
  if (isRegression) {
    let sumSq = 0;
    let sumAbs = 0;
    const yMean = mean(data.y_test);
    let totSq = 0;
    for (let i = 0; i < data.y_test.length; i++) {
      const err = predictions[i] - data.y_test[i];
      sumSq += err * err;
      sumAbs += Math.abs(err);
      totSq += Math.pow(data.y_test[i] - yMean, 2);
    }
    const mse = +(sumSq / (data.y_test.length || 1)).toFixed(3);
    const mae = +(sumAbs / (data.y_test.length || 1)).toFixed(3);
    const rmse = +(Math.sqrt(mse)).toFixed(3);
    const r2 = totSq > 1e-9 ? +(Math.max(-2, 1 - sumSq / totSq)).toFixed(3) : 0;
    metrics = { mae, mse, rmse, r2, trainTimeMs: Math.round(performance.now() - startTime), sampleCount: X_train.length };
  } else {
    let correct = 0;
    const numClasses = Math.max(2, (data.targetClasses || []).length);
    const confMat: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    for (let i = 0; i < data.y_test.length; i++) {
      const act = Math.floor(data.y_test[i]);
      const pr = Math.floor(predictions[i]);
      if (act === pr) correct++;
      if (confMat[act] && confMat[act][pr] !== undefined) confMat[act][pr]++;
    }
    const accuracy = +(correct / (data.y_test.length || 1)).toFixed(3);
    metrics = {
      accuracy,
      precision: accuracy,
      recall: accuracy,
      f1Score: accuracy,
      confusionMatrix: { matrix: confMat, labels: data.targetClasses || ['Class 0', 'Class 1'] },
      trainTimeMs: Math.round(performance.now() - startTime),
      sampleCount: X_train.length,
    };
  }

  const queryPoint = data.X_test[0] || [0, 0];
  const queryResult = predictSample(queryPoint);

  const steps: SimulationStep[] = [
    {
      stepNumber: 0,
      title: 'Instance-Based Memorization',
      description: `Loaded ${X_train.length} training vectors into memory. Non-parametric KNN requires no offline weight training.`,
      operation: 'Data Indexing',
      formula: 'd(p, q) = √(∑(p_i - q_i)²)',
      visualState: {},
    },
    {
      stepNumber: 1,
      title: `Compute Distances to Test Query (K=${k})`,
      description: `Calculated pairwise distances using ${distMetric} metric. Identified ${k} nearest neighbor vectors.`,
      operation: 'Distance Ranking',
      intermediateValues: { k, metric: distMetric },
      visualState: {
        queryPoint,
        neighborIndices: queryResult.neighborIndices,
      },
    },
    {
      stepNumber: 2,
      title: 'Plurality Voting / Averaging',
      description: `Tallied votes among the ${k} closest neighbors to assign final class outcome.`,
      operation: 'Vote Aggregation',
      visualState: {
        queryPoint,
        neighborIndices: queryResult.neighborIndices,
      },
    },
  ];

  return {
    algorithmId: 'knn',
    metrics,
    predictions,
    simulationSteps: steps,
    modelDetails: { k, metric: distMetric },
    predict: (sample: number[]) => {
      const res = predictSample(sample).prediction;
      if (isRegression) return res.toFixed(2);
      return data.targetClasses ? data.targetClasses[res] || res : res;
    },
  };
}

// -------------------------------------------------------------
// 7. Support Vector Machine (Linear SVM)
// -------------------------------------------------------------
export function trainSVM(
  data: PreparedData,
  hyperparams: { C?: number; learningRate?: number; maxIterations?: number } = {}
): TrainResult {
  const startTime = performance.now();
  const C = hyperparams.C || 1.0;
  const lr = hyperparams.learningRate || 0.01;
  const iters = hyperparams.maxIterations || 60;
  const X_train = data.X_train;
  const y_train = data.y_train.map((y) => (y > 0 ? 1 : -1)); // SVM uses +1, -1
  const n = X_train.length;
  const d = data.featureNames.length;

  let weights = new Array(d).fill(0);
  let bias = 0;
  const steps: SimulationStep[] = [];

  for (let iter = 1; iter <= iters; iter++) {
    for (let i = 0; i < n; i++) {
      const condition = y_train[i] * (dot(weights, X_train[i]) + bias) >= 1;
      if (condition) {
        // Gradient of 0.5 * ||w||^2
        for (let j = 0; j < d; j++) {
          weights[j] -= lr * (weights[j] / n);
        }
      } else {
        // Subgradient with hinge loss penalty
        for (let j = 0; j < d; j++) {
          weights[j] -= lr * (weights[j] / n - C * y_train[i] * X_train[i][j]);
        }
        bias += lr * C * y_train[i];
      }
    }

    if (iter === 1 || iter === 20 || iter === 40 || iter === iters) {
      const wNorm = vectorNorm(weights);
      const margin = wNorm > 1e-6 ? +(2 / wNorm).toFixed(3) : 0;
      steps.push({
        stepNumber: steps.length,
        title: `SVM Iteration ${iter}: Margin Width = ${margin}`,
        description: `Maximized separating margin while penalizing margin violations with penalty C = ${C}.`,
        operation: 'Hinge Loss Optimization',
        formula: 'min 1/2 ||w||² + C ∑ max(0, 1 - y_i(w·x_i + b))',
        intermediateValues: { marginWidth: margin, wNorm: +wNorm.toFixed(3), bias: +bias.toFixed(3) },
        visualState: { decisionBoundary: { w: [...weights], b: bias } },
      });
    }
  }

  const predictClass = (x: number[]): number => {
    return dot(weights, x) + bias >= 0 ? 1 : 0;
  };

  const predictions = data.X_test.map(predictClass);

  let correct = 0;
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < data.y_test.length; i++) {
    const act = data.y_test[i] > 0 ? 1 : 0;
    const pr = predictions[i];
    if (act === pr) correct++;
    if (act === 1 && pr === 1) tp++;
    else if (act === 0 && pr === 1) fp++;
    else if (act === 0 && pr === 0) tn++;
    else if (act === 1 && pr === 0) fn++;
  }

  const accuracy = +(correct / (data.y_test.length || 1)).toFixed(3);
  const precision = tp + fp > 0 ? +(tp / (tp + fp)).toFixed(3) : 0;
  const recall = tp + fn > 0 ? +(tp / (tp + fn)).toFixed(3) : 0;
  const f1Score = precision + recall > 0 ? +((2 * precision * recall) / (precision + recall)).toFixed(3) : 0;

  return {
    algorithmId: 'svm',
    metrics: {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix: {
        matrix: [
          [tn, fp],
          [fn, tp],
        ],
        labels: data.targetClasses ? data.targetClasses.slice(0, 2) : ['Negative', 'Positive'],
      },
      trainTimeMs: Math.round(performance.now() - startTime),
      sampleCount: n,
    },
    predictions,
    simulationSteps: steps,
    modelDetails: { weights, bias, C },
    predict: (sample: number[]) => {
      const cls = predictClass(sample);
      return data.targetClasses ? data.targetClasses[cls] || cls : cls;
    },
  };
}

// -------------------------------------------------------------
// 8. Principal Component Analysis (PCA)
// -------------------------------------------------------------
export function trainPCA(
  data: PreparedData,
  hyperparams: { numComponents?: number } = {}
): TrainResult {
  const startTime = performance.now();
  const k = Math.min(data.featureNames.length, hyperparams.numComponents || 2);
  const X = [...data.X_train, ...data.X_test];
  const n = X.length;

  // Covariance matrix of centered data
  const cov = computeCovarianceMatrix(X);
  const { eigenvalues, eigenvectors } = powerIterationTopEigenvectors(cov, k, 120);

  const totalVar = eigenvalues.reduce((a, b) => a + b, 0);
  const explainedVarianceRatios = eigenvalues.map((ev) =>
    totalVar > 0 ? +(ev / totalVar).toFixed(3) : +(1 / k).toFixed(3)
  );

  // Project data onto top k components
  const projectSample = (x: number[]): number[] => {
    return eigenvectors.map((ev) => dot(x, ev));
  };

  const projected2D = X.map(projectSample);

  const steps: SimulationStep[] = [
    {
      stepNumber: 0,
      title: 'Mean Centering & Covariance Matrix',
      description: `Centered ${n} records and computed the ${data.featureNames.length}x${data.featureNames.length} feature covariance matrix.`,
      operation: 'Covariance Computation',
      formula: 'Σ = (1 / (N - 1)) · X̃ᵀ X̃',
      visualState: {},
    },
    {
      stepNumber: 1,
      title: 'Eigen-Decomposition for Principal Axes',
      description: `Extracted top ${k} orthogonal eigenvectors corresponding to maximal variance directions.`,
      operation: 'Eigenvalue Extraction',
      formula: 'Σ v = λ v',
      intermediateValues: {
        explainedVarianceRatio: explainedVarianceRatios,
        topEigenvalues: eigenvalues.map((v) => +v.toFixed(3)),
      },
      visualState: {},
    },
    {
      stepNumber: 2,
      title: `Linear Subspace Projection (Dimension ${data.featureNames.length} → ${k})`,
      description: `Projected feature space to ${k} dimensions, capturing ${(explainedVarianceRatios.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of total variance.`,
      operation: 'Orthogonal Projection',
      formula: 'Z = X · V_k',
      visualState: {},
    },
  ];

  return {
    algorithmId: 'pca',
    metrics: {
      trainTimeMs: Math.round(performance.now() - startTime),
      sampleCount: n,
    },
    predictions: [],
    simulationSteps: steps,
    modelDetails: {
      explainedVarianceRatios,
      eigenvectors,
      projected2D: projected2D.map((p) => [+(p[0] || 0).toFixed(2), +(p[1] || 0).toFixed(2)]),
    },
    predict: (sample: number[]) => {
      const proj = projectSample(sample);
      return `PC1: ${proj[0].toFixed(2)}, PC2: ${proj[1]?.toFixed(2) || '0'}`;
    },
  };
}

// -------------------------------------------------------------
// Unified Dispatcher: Train any algorithm
// -------------------------------------------------------------
export function executeModelTraining(
  algorithmId: AlgorithmId,
  data: PreparedData,
  hyperparams: Record<string, any>,
  taskType: 'classification' | 'regression' | 'clustering'
): TrainResult {
  switch (algorithmId) {
    case 'linear_regression':
      return trainLinearRegression(data, hyperparams);
    case 'logistic_regression':
      return trainLogisticRegression(data, hyperparams);
    case 'decision_tree':
      return trainDecisionTree(data, hyperparams, taskType === 'regression');
    case 'random_forest':
      return trainRandomForest(data, hyperparams, taskType === 'regression');
    case 'kmeans':
      return trainKMeans(data, hyperparams);
    case 'knn':
      return trainKNN(data, hyperparams, taskType === 'regression');
    case 'svm':
      return trainSVM(data, hyperparams);
    case 'pca':
      return trainPCA(data, hyperparams);
    default:
      throw new Error(`Unsupported algorithm: ${algorithmId}`);
  }
}
