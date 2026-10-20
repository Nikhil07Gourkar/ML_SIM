import { DatasetRecord, PreparedData, PreprocessingConfig } from '../types/ml';
import { mean, quantile, SeededRandom, stdDev } from './mathUtils';

export function preprocessDataset(
  rawData: DatasetRecord[],
  config: PreprocessingConfig,
  taskType: 'classification' | 'regression' | 'clustering'
): PreparedData {
  const rng = new SeededRandom(config.randomSeed || 42);
  let workingData = rawData.map((row) => ({ ...row }));
  const initialRowCount = workingData.length;
  let duplicateRowsRemoved = 0;
  const outlierWarnings: string[] = [];

  // 1. Remove duplicate rows if requested
  if (config.removeDuplicates) {
    const seen = new Set<string>();
    const uniqueRows: DatasetRecord[] = [];
    for (const row of workingData) {
      const key = JSON.stringify(row);
      if (!seen.has(key)) {
        seen.add(key);
        uniqueRows.push(row);
      } else {
        duplicateRowsRemoved++;
      }
    }
    workingData = uniqueRows;
  }

  // 2. Identify target column and features
  const targetCol = taskType !== 'clustering' ? config.targetColumn : undefined;
  const featureCols = config.selectedFeatures.filter((f) => f !== targetCol);

  if (featureCols.length === 0) {
    throw new Error('At least one feature column must be selected for model training.');
  }

  // 3. Handle missing values
  let droppedRowsCount = 0;
  if (config.missingStrategy === 'drop') {
    const cleanRows: DatasetRecord[] = [];
    for (const row of workingData) {
      let hasMissing = false;
      // Check feature cols + target col
      const colsToCheck = targetCol ? [...featureCols, targetCol] : featureCols;
      for (const col of colsToCheck) {
        const val = row[col];
        if (val === '' || val === null || val === undefined || (typeof val === 'number' && isNaN(val))) {
          hasMissing = true;
          break;
        }
      }
      if (!hasMissing) {
        cleanRows.push(row);
      } else {
        droppedRowsCount++;
      }
    }
    workingData = cleanRows;
  }

  if (workingData.length < 5) {
    throw new Error(`Dataset has only ${workingData.length} valid rows after missing value filtering. At least 5 rows are required.`);
  }

  // 4. Train / Test Split (Strictly before fitting transformers to prevent leakage!)
  const testRatio = Math.max(0.1, Math.min(0.5, config.testSplitRatio || 0.2));
  const shuffledIndices = rng.shuffle(Array.from({ length: workingData.length }, (_, i) => i));
  const testCount = Math.max(1, Math.floor(workingData.length * testRatio));
  const trainCount = workingData.length - testCount;

  const trainIndices = shuffledIndices.slice(0, trainCount);
  const testIndices = shuffledIndices.slice(trainCount);

  const rawTrainRows = trainIndices.map((i) => workingData[i]);
  const rawTestRows = testIndices.map((i) => workingData[i]);

  // 5. Fit Imputers on Training Data Only
  const numericImputations: Record<string, number> = {};
  const categoricalImputations: Record<string, string> = {};

  if (config.missingStrategy !== 'drop' && config.missingStrategy !== 'none') {
    for (const col of featureCols) {
      const trainVals = rawTrainRows.map((r) => r[col]).filter((v) => v !== '' && v !== null && v !== undefined && !isNaN(v as number));
      if (trainVals.length > 0 && typeof trainVals[0] === 'number') {
        const numVals = trainVals as number[];
        if (config.missingStrategy === 'impute_mean') {
          numericImputations[col] = mean(numVals);
        } else if (config.missingStrategy === 'impute_median') {
          const sorted = [...numVals].sort((a, b) => a - b);
          numericImputations[col] = quantile(sorted, 0.5);
        }
      } else {
        // Mode for categorical
        const counts: Record<string, number> = {};
        for (const v of trainVals) {
          const s = String(v);
          counts[s] = (counts[s] || 0) + 1;
        }
        let bestKey = '';
        let maxCount = -1;
        for (const k in counts) {
          if (counts[k] > maxCount) {
            maxCount = counts[k];
            bestKey = k;
          }
        }
        categoricalImputations[col] = bestKey;
      }
    }
  }

  // Helper to get imputed value for a row
  const getImputedVal = (row: DatasetRecord, col: string): string | number => {
    const val = row[col];
    if (val !== '' && val !== null && val !== undefined && !isNaN(val as number)) {
      return val;
    }
    if (numericImputations[col] !== undefined) return numericImputations[col];
    if (categoricalImputations[col] !== undefined) return categoricalImputations[col];
    return 0;
  };

  // 6. Build Target Classes and Maps for Classification / Numeric for Regression
  let targetClasses: string[] = [];
  const targetToNumeric = (val: string | number): number => {
    if (taskType === 'regression') {
      const num = Number(val);
      return isNaN(num) ? 0 : num;
    }
    const str = String(val);
    let idx = targetClasses.indexOf(str);
    if (idx === -1) {
      targetClasses.push(str);
      idx = targetClasses.length - 1;
    }
    return idx;
  };

  // Collect target classes from train set first
  if (targetCol && taskType === 'classification') {
    const uniqueTargets = Array.from(new Set(rawTrainRows.map((r) => String(r[targetCol]))));
    targetClasses = uniqueTargets.sort();
  }

  // 7. Categorical Encoding on Features
  interface FeatureEncoder {
    name: string;
    isCategorical: boolean;
    categories?: string[];
  }

  const encoders: FeatureEncoder[] = featureCols.map((col) => {
    const firstVal = rawTrainRows.find((r) => r[col] !== '' && r[col] !== null)?.[col];
    const isCat = typeof firstVal === 'string' && isNaN(Number(firstVal));
    if (isCat) {
      const uniqueCats = Array.from(new Set(rawTrainRows.map((r) => String(getImputedVal(r, col))))).sort();
      return { name: col, isCategorical: true, categories: uniqueCats };
    }
    return { name: col, isCategorical: false };
  });

  // Expand final numeric column names
  const finalFeatureNames: string[] = [];
  for (const enc of encoders) {
    if (!enc.isCategorical) {
      finalFeatureNames.push(enc.name);
    } else if (config.encodeCategorical === 'one_hot' && enc.categories) {
      for (const cat of enc.categories) {
        finalFeatureNames.push(`${enc.name}_${cat}`);
      }
    } else {
      finalFeatureNames.push(enc.name);
    }
  }

  // Vector transform function
  const transformRow = (row: DatasetRecord): number[] => {
    const vector: number[] = [];
    for (const enc of encoders) {
      const rawVal = getImputedVal(row, enc.name);
      if (!enc.isCategorical) {
        const num = Number(rawVal);
        vector.push(isNaN(num) ? 0 : num);
      } else if (config.encodeCategorical === 'one_hot' && enc.categories) {
        const strVal = String(rawVal);
        for (const cat of enc.categories) {
          vector.push(strVal === cat ? 1 : 0);
        }
      } else if (enc.categories) {
        // Label encode
        const idx = enc.categories.indexOf(String(rawVal));
        vector.push(idx === -1 ? 0 : idx);
      } else {
        vector.push(0);
      }
    }
    return vector;
  };

  let X_train = rawTrainRows.map(transformRow);
  let X_test = rawTestRows.map(transformRow);

  const y_train = targetCol ? rawTrainRows.map((r) => targetToNumeric(r[targetCol])) : [];
  const y_test = targetCol ? rawTestRows.map((r) => targetToNumeric(r[targetCol])) : [];

  // 8. Detect Constant Columns & Outliers on Training set
  const d = finalFeatureNames.length;
  for (let colIdx = 0; colIdx < d; colIdx++) {
    const colVals = X_train.map((r) => r[colIdx]);
    const m = mean(colVals);
    const s = stdDev(colVals);

    if (s < 1e-9) {
      outlierWarnings.push(`Feature "${finalFeatureNames[colIdx]}" is constant (all values = ${m.toFixed(2)}). It will not contribute variance to the model.`);
    }

    // Check IQR for outliers
    const sorted = [...colVals].sort((a, b) => a - b);
    const q1 = quantile(sorted, 0.25);
    const q3 = quantile(sorted, 0.75);
    const iqr = q3 - q1;
    if (iqr > 0) {
      const lower = q1 - 1.5 * iqr;
      const upper = q3 + 1.5 * iqr;
      const outlierCount = colVals.filter((v) => v < lower || v > upper).length;
      if (outlierCount > 0) {
        outlierWarnings.push(`Feature "${finalFeatureNames[colIdx]}" has ${outlierCount} statistical outlier(s) outside [${lower.toFixed(2)}, ${upper.toFixed(2)}].`);
      }
    }
  }

  // 9. Feature Scaling (Fitted ON X_train ONLY, transformed on both)
  let scalerParams: PreparedData['scalerParams'] = undefined;
  if (config.scaleMethod !== 'none') {
    const means: number[] = new Array(d).fill(0);
    const stds: number[] = new Array(d).fill(1);
    const mins: number[] = new Array(d).fill(0);
    const maxs: number[] = new Array(d).fill(1);

    for (let j = 0; j < d; j++) {
      const colVals = X_train.map((r) => r[j]);
      means[j] = mean(colVals);
      const s = stdDev(colVals);
      stds[j] = s > 1e-8 ? s : 1;

      mins[j] = Math.min(...colVals);
      maxs[j] = Math.max(...colVals);
      if (Math.abs(maxs[j] - mins[j]) < 1e-8) {
        maxs[j] = mins[j] + 1;
      }
    }

    scalerParams = { means, stds, mins, maxs };

    const scaleVector = (vec: number[]): number[] => {
      return vec.map((val, idx) => {
        if (config.scaleMethod === 'standard') {
          return (val - means[idx]) / stds[idx];
        } else {
          // minmax
          return (val - mins[idx]) / (maxs[idx] - mins[idx]);
        }
      });
    };

    X_train = X_train.map(scaleVector);
    X_test = X_test.map(scaleVector);
  }

  return {
    featureNames: finalFeatureNames,
    targetName: targetCol,
    targetClasses: targetClasses.length > 0 ? targetClasses : undefined,
    X_train,
    y_train,
    X_test,
    y_test,
    rawTrainRows,
    rawTestRows,
    scalerParams,
    droppedRowsCount,
    duplicateRowsRemoved,
    outlierWarnings,
  };
}
