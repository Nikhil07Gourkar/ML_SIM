/**
 * Mathematical and Statistical Utilities for Machine Learning
 * Implements linear algebra, distance metrics, and statistical measures from scratch
 */

// Deterministic Pseudo-Random Number Generator (PRNG) using Mulberry32
export class SeededRandom {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed >>> 0;
  }

  // Returns float in [0, 1)
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Gaussian distributed random number using Box-Muller transform
  nextGaussian(mean: number = 0, std: number = 1): number {
    let u1 = this.next();
    let u2 = this.next();
    while (u1 === 0) u1 = this.next();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z * std;
  }

  // Shuffle an array in-place
  shuffle<T>(array: T[]): T[] {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// Vector operations
export function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function vectorNorm(v: number[]): number {
  return Math.sqrt(dot(v, v));
}

export function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export function manhattanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

// Matrix operations
export function transpose(matrix: number[][]): number[][] {
  const rows = matrix.length;
  if (rows === 0) return [];
  const cols = matrix[0].length;
  const result: number[][] = Array.from({ length: cols }, () => new Array(rows));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      result[c][r] = matrix[r][c];
    }
  }
  return result;
}

export function matrixMultiply(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const rowsB = B.length;
  const colsB = B[0].length;

  if (colsA !== rowsB) {
    throw new Error(`Matrix multiplication dimension mismatch: ${colsA} vs ${rowsB}`);
  }

  const result: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const a_ik = A[i][k];
      for (let j = 0; j < colsB; j++) {
        result[i][j] += a_ik * B[k][j];
      }
    }
  }
  return result;
}

// Matrix inverse using Gauss-Jordan elimination with partial pivoting and L2 ridge regularization
export function invertMatrix(matrix: number[][], ridge: number = 1e-6): number[][] {
  const n = matrix.length;
  // Create augmented matrix [A + ridge*I | I]
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    A[i] = [];
    for (let j = 0; j < n; j++) {
      A[i][j] = matrix[i][j] + (i === j ? ridge : 0);
    }
    for (let j = 0; j < n; j++) {
      A[i][n + j] = i === j ? 1 : 0;
    }
  }

  // Forward elimination & back-substitution
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    let maxVal = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > maxVal) {
        maxVal = Math.abs(A[r][i]);
        maxRow = r;
      }
    }

    if (maxVal < 1e-12) {
      // Near-singular, fallback identity matrix to avoid NaN
      return Array.from({ length: n }, (_, r) =>
        Array.from({ length: n }, (_, c) => (r === c ? 1 : 0))
      );
    }

    // Swap rows
    if (maxRow !== i) {
      [A[i], A[maxRow]] = [A[maxRow], A[i]];
    }

    // Scale pivot row
    const pivot = A[i][i];
    for (let c = 0; c < 2 * n; c++) {
      A[i][c] /= pivot;
    }

    // Eliminate other rows
    for (let r = 0; r < n; r++) {
      if (r !== i) {
        const factor = A[r][i];
        for (let c = 0; c < 2 * n; c++) {
          A[r][c] -= factor * A[i][c];
        }
      }
    }
  }

  // Extract right half
  const inverse: number[][] = [];
  for (let i = 0; i < n; i++) {
    inverse[i] = [];
    for (let j = 0; j < n; j++) {
      inverse[i][j] = A[i][n + j];
    }
  }
  return inverse;
}

// Statistical calculations
export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  const sum = values.reduce((acc, val) => acc + val, 0);
  return sum / values.length;
}

export function variance(values: number[], sample: boolean = true): number {
  if (values.length <= 1) return 0;
  const m = mean(values);
  const sumSq = values.reduce((acc, val) => acc + Math.pow(val - m, 2), 0);
  return sumSq / (sample ? values.length - 1 : values.length);
}

export function stdDev(values: number[], sample: boolean = true): number {
  return Math.sqrt(variance(values, sample));
}

export function quantile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0];
  const pos = (sortedValues.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sortedValues[base + 1] !== undefined) {
    return sortedValues[base] + rest * (sortedValues[base + 1] - sortedValues[base]);
  }
  return sortedValues[base];
}

// Pearson Correlation Coefficient
export function pearsonCorrelation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n <= 1) return 0;

  const meanX = mean(x.slice(0, n));
  const meanY = mean(y.slice(0, n));

  let num = 0;
  let denX = 0;
  let denY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }

  const den = Math.sqrt(denX * denY);
  if (den < 1e-12) return 0;
  return Math.max(-1, Math.min(1, num / den));
}

// Sigmoid Activation Function: sigma(z) = 1 / (1 + e^(-z))
export function sigmoid(z: number): number {
  // Clamp to prevent overflow
  if (z > 40) return 1;
  if (z < -40) return 0;
  return 1 / (1 + Math.exp(-z));
}

// Gini Impurity for class labels: 1 - sum(p_i^2)
export function giniImpurity(labels: number[]): number {
  const total = labels.length;
  if (total === 0) return 0;

  const counts: Record<number, number> = {};
  for (const label of labels) {
    counts[label] = (counts[label] || 0) + 1;
  }

  let sumSqProb = 0;
  for (const label in counts) {
    const p = counts[label] / total;
    sumSqProb += p * p;
  }

  return 1 - sumSqProb;
}

// Entropy for class labels: - sum(p_i * log2(p_i))
export function entropy(labels: number[]): number {
  const total = labels.length;
  if (total === 0) return 0;

  const counts: Record<number, number> = {};
  for (const label of labels) {
    counts[label] = (counts[label] || 0) + 1;
  }

  let ent = 0;
  for (const label in counts) {
    const p = counts[label] / total;
    if (p > 0) {
      ent -= p * Math.log2(p);
    }
  }

  return ent;
}

// Covariance matrix of X (n x d)
export function computeCovarianceMatrix(X: number[][]): number[][] {
  const n = X.length;
  if (n <= 1) return [];
  const d = X[0].length;

  // Calculate means of each column
  const means: number[] = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    means[j] /= n;
  }

  // Center the data
  const centered: number[][] = Array.from({ length: n }, () => new Array(d));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      centered[i][j] = X[i][j] - means[j];
    }
  }

  // Compute (X_centered^T * X_centered) / (n - 1)
  const cov: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let j1 = 0; j1 < d; j1++) {
    for (let j2 = j1; j2 < d; j2++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += centered[i][j1] * centered[i][j2];
      }
      const val = sum / (n - 1);
      cov[j1][j2] = val;
      cov[j2][j1] = val;
    }
  }

  return cov;
}

// Power Iteration for top k eigenvectors of a symmetric matrix
export function powerIterationTopEigenvectors(
  matrix: number[][],
  numComponents: number = 2,
  maxIterations: number = 100
): { eigenvalues: number[]; eigenvectors: number[][] } {
  const d = matrix.length;
  const k = Math.min(numComponents, d);
  let A = matrix.map((row) => [...row]);

  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];

  for (let comp = 0; comp < k; comp++) {
    // Initial random vector
    let b: number[] = Array.from({ length: d }, (_, i) => Math.sin(comp * 10 + i + 1));
    let norm = vectorNorm(b);
    b = b.map((x) => x / (norm || 1));

    let eigenvalue = 0;

    for (let iter = 0; iter < maxIterations; iter++) {
      // Multiply A * b
      const Ab: number[] = new Array(d).fill(0);
      for (let r = 0; r < d; r++) {
        for (let c = 0; c < d; c++) {
          Ab[r] += A[r][c] * b[c];
        }
      }

      const AbNorm = vectorNorm(Ab);
      if (AbNorm < 1e-12) break;

      // Rayleigh quotient
      eigenvalue = dot(b, Ab);
      b = Ab.map((x) => x / AbNorm);
    }

    eigenvalues.push(Math.max(0, eigenvalue));
    eigenvectors.push(b);

    // Deflate matrix: A_next = A - lambda * (v * v^T)
    for (let r = 0; r < d; r++) {
      for (let c = 0; c < d; c++) {
        A[r][c] -= eigenvalue * b[r] * b[c];
      }
    }
  }

  return { eigenvalues, eigenvectors };
}
