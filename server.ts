import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: '10mb' }));

// Initialize GoogleGenAI SDK on server side with mandatory User-Agent
let ai: GoogleGenAI | null = null;
if (process.env.GEMINI_API_KEY) {
  try {
    ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  } catch (err) {
    console.warn('Failed to initialize GoogleGenAI client:', err);
  }
}

// 1. AI Tutor Endpoint
app.post('/api/tutor', async (req: Request, res: Response) => {
  const { question, context, history } = req.body;

  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'Question is required' });
  }

  const systemInstruction = `You are a brilliant, encouraging, and clear Machine Learning Professor and AI Tutor for the "ML Algorithm Simulator" platform.
Your goal is to help students learn machine learning from scratch through intuition, clear analogies, visual geometry, and exact mathematics.

Context provided:
- Current Lesson: ${context?.lessonTitle || 'General ML'}
- Current Algorithm: ${context?.algorithmId || 'None'}
- Selected Dataset: ${context?.datasetName || 'None'} (${context?.taskType || 'N/A'})
- Recent Model Performance: Accuracy: ${context?.metrics?.accuracy ?? 'N/A'}, MSE: ${context?.metrics?.mse ?? 'N/A'}, R2: ${context?.metrics?.r2 ?? 'N/A'}, Silhouette: ${context?.metrics?.silhouetteScore ?? 'N/A'}
- Model Hyperparameters: ${JSON.stringify(context?.hyperparameters || {})}

Rules:
1. Always distinguish actual calculated statistics from theoretical derivations. Never invent numbers that contradict the provided context.
2. Explain concepts in intuitive everyday language before providing formal mathematical equations.
3. If the user asks for code, provide clean, idiomatic Python with scikit-learn.
4. Keep answers focused, engaging, and under 3-4 paragraphs. Use markdown formatting with bullet points.
5. If the user asks about a poor model score (e.g. low R2 or low accuracy), analyze the hyperparameters and dataset to give actionable diagnosis.`;

  // If Gemini API is available, call Gemini 3.8 Flash
  if (ai && process.env.GEMINI_API_KEY) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: question,
        config: {
          systemInstruction,
        },
      });

      const reply = response.text || 'I analyzed your query. How else can I assist your machine learning journey?';
      return res.json({ reply, source: 'gemini-3.8-flash' });
    } catch (apiError: any) {
      console.warn('Gemini API call failed, using intelligent educational fallback:', apiError?.message);
    }
  }

  // Educational Fallback Engine when Gemini Key is missing or rate-limited
  const lowerQ = question.toLowerCase();
  let fallbackReply = '';

  if (lowerQ.includes('overfit') || lowerQ.includes('variance')) {
    fallbackReply = `### Overfitting & High Variance
Overfitting happens when your model memorizes random noise and anomalies in the training set instead of learning the true general relationship.
- **Symptom**: Extremely high training accuracy (e.g. 99%) but poor test accuracy (e.g. 60%).
- **Remedy**:
  1. For Decision Trees: Reduce \`max_depth\` or increase \`min_samples_split\`.
  2. For Linear Models: Use L2 regularization (Ridge) or collect more training rows.
  3. Use ensemble methods like **Random Forest** to average out individual tree variance!`;
  } else if (lowerQ.includes('slope') || lowerQ.includes('intercept') || lowerQ.includes('linear')) {
    fallbackReply = `### Linear Regression Mechanics
In Linear Regression, the formula is:
$$\\hat{y} = w \\cdot x + b$$
- **Slope ($w$)**: The rate of change. It tells you how much target $y$ changes for every 1-unit increase in feature $x$.
- **Intercept ($b$)**: The baseline value of $y$ when all feature inputs $x$ are zero.
- **Least Squares**: We find optimal $w$ and $b$ by minimizing the sum of squared vertical distances (residuals) between each point and the line.`;
  } else if (lowerQ.includes('cluster') || lowerQ.includes('kmeans') || lowerQ.includes('elbow')) {
    fallbackReply = `### K-Means Clustering & The Elbow Method
K-Means groups unlabeled points into $K$ distinct geometric clusters:
1. **Centroids**: The mathematical average (mean) coordinate of all points currently assigned to that cluster.
2. **Inertia (WCSS)**: The sum of squared distances from every point to its closest centroid.
3. **Elbow Method**: As you increase $K$, inertia always drops. The "elbow" is the point where the curve bends and adding more clusters gives diminishing returns!`;
  } else if (lowerQ.includes('metric') || lowerQ.includes('precision') || lowerQ.includes('recall') || lowerQ.includes('f1')) {
    fallbackReply = `### Choosing the Right Evaluation Metric
- **Accuracy**: $\\frac{TP + TN}{\\text{Total}}$. Great when classes are balanced, misleading when one class is rare.
- **Precision**: $\\frac{TP}{TP + FP}$. Purity of alarms. Crucial when false alarms are expensive (e.g. spam filter).
- **Recall**: $\\frac{TP}{TP + FN}$. Sensitivity. Crucial when missing a positive is disastrous (e.g. cancer detection, fraud).
- **F1-Score**: Harmonic mean of Precision and Recall. Balances both metrics!`;
  } else {
    fallbackReply = `### Machine Learning Tutor Insight
Based on your current workspace:
- **Algorithm**: ${context?.algorithmId || 'Algorithm Simulation'}
- **Dataset**: ${context?.datasetName || 'Active Dataset'}

Machine learning algorithms learn functional mappings $f(X) \\to y$ by minimizing a loss function (like Mean Squared Error for regression or Cross-Entropy for classification).
Try stepping through the simulation in the **Algorithm Simulator** to see how weights or centroids adjust at each step!`;
  }

  return res.json({ reply: fallbackReply, source: 'educational-knowledge-base' });
});

// 2. Python Backend Health Check Proxy
app.get('/api/python-status', async (_req: Request, res: Response) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1200);

    const response = await fetch('http://localhost:8000/api/health', {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return res.json({ online: true, data });
    }
  } catch (err) {
    // Offline, perfectly normal when using the built-in mathematical engine
  }

  return res.json({
    online: false,
    message: 'External Python FastAPI service is offline. Using built-in high-performance TypeScript ML engine.',
  });
});

// Vite Middleware for Dev or Static Serving for Prod
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`ML Algorithm Simulator running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
