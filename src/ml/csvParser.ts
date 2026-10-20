import { DatasetFeature, DatasetRecord } from '../types/ml';

export interface CSVParseResult {
  records: DatasetRecord[];
  features: DatasetFeature[];
  rows: number;
  columns: number;
  detectedTarget?: string;
  duplicateCount: number;
  errors: string[];
}

export function parseCSVString(csvContent: string): CSVParseResult {
  const errors: string[] = [];
  if (!csvContent || csvContent.trim().length === 0) {
    return {
      records: [],
      features: [],
      rows: 0,
      columns: 0,
      duplicateCount: 0,
      errors: ['The uploaded file is empty. Please upload a valid CSV file with data rows.'],
    };
  }

  // Split lines accounting for \r\n and \n
  const rawLines = csvContent.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (rawLines.length < 2) {
    return {
      records: [],
      features: [],
      rows: 0,
      columns: 0,
      duplicateCount: 0,
      errors: ['CSV must have at least a header row and at least one data row.'],
    };
  }

  // Detect delimiter: comma, semicolon, tab
  const headerLine = rawLines[0];
  const commaCount = (headerLine.match(/,/g) || []).length;
  const semiCount = (headerLine.match(/;/g) || []).length;
  const tabCount = (headerLine.match(/\t/g) || []).length;

  let delimiter = ',';
  if (semiCount > commaCount && semiCount > tabCount) delimiter = ';';
  else if (tabCount > commaCount && tabCount > semiCount) delimiter = '\t';

  // Helper to split a line safely handling quotes
  const parseLine = (line: string): string[] => {
    const values: string[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++; // skip escaped quote
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === delimiter && !inQuotes) {
        values.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    values.push(current.trim());
    return values;
  };

  const rawHeaders = parseLine(headerLine);
  // Sanitize headers: trim quotes and whitespace
  const headers = rawHeaders.map((h, idx) => {
    const cleaned = h.replace(/^["']|["']$/g, '').trim();
    return cleaned.length > 0 ? cleaned : `column_${idx + 1}`;
  });

  const records: DatasetRecord[] = [];
  const seenRowStrings = new Set<string>();
  let duplicateCount = 0;

  for (let lineIdx = 1; lineIdx < rawLines.length; lineIdx++) {
    const rowTokens = parseLine(rawLines[lineIdx]);
    if (rowTokens.length === 0 || (rowTokens.length === 1 && rowTokens[0] === '')) {
      continue;
    }

    // Check for row duplicates
    const rowKey = rowTokens.join('|');
    if (seenRowStrings.has(rowKey)) {
      duplicateCount++;
    } else {
      seenRowStrings.add(rowKey);
    }

    const record: DatasetRecord = {};
    for (let c = 0; c < headers.length; c++) {
      const colName = headers[c];
      const valStr = rowTokens[c] !== undefined ? rowTokens[c].replace(/^["']|["']$/g, '').trim() : '';

      // Check missing markers
      if (valStr === '' || valStr.toLowerCase() === 'na' || valStr.toLowerCase() === 'nan' || valStr.toLowerCase() === 'null' || valStr === '?') {
        record[colName] = '';
      } else {
        const num = Number(valStr);
        if (!isNaN(num) && valStr !== '') {
          record[colName] = num;
        } else {
          record[colName] = valStr;
        }
      }
    }
    records.push(record);
  }

  if (records.length === 0) {
    return {
      records: [],
      features: [],
      rows: 0,
      columns: headers.length,
      duplicateCount: 0,
      errors: ['No valid data rows found in the CSV file.'],
    };
  }

  // Analyze feature column data types, missing counts, min/max/mean
  const features: DatasetFeature[] = headers.map((colName) => {
    let numericCount = 0;
    let missingCount = 0;
    const distinctValues = new Set<string>();
    const numbers: number[] = [];

    for (const row of records) {
      const val = row[colName];
      if (val === '' || val === null || val === undefined) {
        missingCount++;
      } else if (typeof val === 'number') {
        numericCount++;
        numbers.push(val);
      } else {
        distinctValues.add(String(val));
      }
    }

    const validCount = records.length - missingCount;
    const isNumeric = validCount > 0 && numericCount / validCount >= 0.8;

    if (isNumeric) {
      // Coerce any string numbers to actual numbers
      for (const row of records) {
        if (typeof row[colName] === 'string' && row[colName] !== '') {
          const parsed = Number(row[colName]);
          if (!isNaN(parsed)) row[colName] = parsed;
        }
      }

      numbers.sort((a, b) => a - b);
      const min = numbers.length > 0 ? numbers[0] : 0;
      const max = numbers.length > 0 ? numbers[numbers.length - 1] : 0;
      const sum = numbers.reduce((a, b) => a + b, 0);
      const mean = numbers.length > 0 ? +(sum / numbers.length).toFixed(3) : 0;

      return {
        name: colName,
        type: 'numeric',
        description: `Numeric feature (${min} to ${max})`,
        min,
        max,
        mean,
        missingCount,
      };
    } else {
      const categories = Array.from(distinctValues).slice(0, 20);
      return {
        name: colName,
        type: 'categorical',
        description: `Categorical feature with ${distinctValues.size} unique values`,
        categories,
        missingCount,
      };
    }
  });

  // Suggest default target: last column or column with 'target', 'label', 'class', 'price'
  let detectedTarget: string | undefined = undefined;
  for (const f of features) {
    const l = f.name.toLowerCase();
    if (l === 'target' || l === 'label' || l === 'class' || l === 'y' || l === 'output' || l.includes('price')) {
      detectedTarget = f.name;
      break;
    }
  }
  if (!detectedTarget && features.length > 1) {
    detectedTarget = features[features.length - 1].name;
  }

  return {
    records,
    features,
    rows: records.length,
    columns: headers.length,
    detectedTarget,
    duplicateCount,
    errors,
  };
}
