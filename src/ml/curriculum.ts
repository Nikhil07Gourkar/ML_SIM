import { CurriculumLesson } from '../types/ml';

export const CURRICULUM_MODULES = [
  { id: 1, title: 'Foundations', description: 'Core principles of Artificial Intelligence and Machine Learning' },
  { id: 2, title: 'Data Fundamentals', description: 'Tabular structure, data cleaning, scaling, and leakage prevention' },
  { id: 3, title: 'Supervised Learning', description: 'Regression, Classification, Decision Trees, and Ensembles' },
  { id: 4, title: 'Unsupervised Learning', description: 'Clustering, Centroids, and Dimensionality Reduction' },
  { id: 5, title: 'Model Evaluation', description: 'Bias-Variance tradeoff, Overfitting, and Diagnostic Metrics' },
  { id: 6, title: 'Practical ML Workflow', description: 'End-to-end industrial modeling pipeline from zero to deployment' },
];

export const CURRICULUM_LESSONS: CurriculumLesson[] = [
  {
    id: 'm1_intro_ml',
    moduleId: 1,
    moduleTitle: 'Foundations',
    order: 1,
    title: 'What is Machine Learning?',
    summary: 'Discover how machine learning differs from traditional rules-based programming.',
    simpleExplanation:
      'In traditional programming, a human programmer writes exact rules by hand (like "if temperature > 100, turn on cooling"). In Machine Learning, we do the reverse: we feed the computer lots of data (inputs and answers) and let the computer discover the hidden rules by itself.',
    technicalDefinition:
      'Machine Learning (ML) is a subfield of computer science where statistical algorithms learn functional mappings f: X → y directly from training data without being explicitly programmed with predetermined heuristic rules.',
    realWorldExample:
      'Spam Email Filters: Instead of writing a million "if email contains words like WINNER or FREE MONEY" rules (which spammers easily evade), an ML model inspects 500,000 spam and non-spam emails and calculates which word patterns reliably correlate with spam.',
    stepByStep: [
      '1. Data Collection: Gather past records of inputs and known outcomes.',
      '2. Model Selection: Choose an algorithm family (e.g., Linear Regression, Decision Tree).',
      '3. Training Phase: The algorithm adjusts its internal mathematical weights to minimize prediction errors.',
      '4. Evaluation: Test the model on unseen data to ensure it generalized rather than memorized.',
      '5. Deployment: Feed brand-new real-world inputs to make automated predictions.',
    ],
    codeSnippet: `# Scikit-Learn Example
from sklearn.linear_model import LinearRegression

# 1. Provide data
X = [[1], [2], [3], [4]]  # Study hours
y = [2, 4, 6, 8]          # Exam scores

# 2. Train model
model = LinearRegression()
model.fit(X, y)

# 3. Predict on unseen data
print(model.predict([[5]]))  # Output: [10.]`,
    keyTakeaways: [
      'Traditional programming: Rules + Data = Answers.',
      'Machine Learning: Data + Answers = Rules.',
      'ML excels at complex patterns that are too difficult or variable for humans to hand-code.',
    ],
    practicalTask: 'Go to the Simulator, select the "Study Hours vs Exam Score" dataset, and run Linear Regression to see how the computer finds the slope by itself.',
    quizQuestions: [
      {
        id: 'q1_1',
        question: 'How does Machine Learning primarily differ from traditional software programming?',
        options: [
          'ML requires faster processors to run normal if-else statements',
          'In ML, the computer learns rules from data rather than humans hand-coding all rules',
          'ML only works with text data, while programming works with numbers',
          'ML models never make any prediction errors',
        ],
        correctIndex: 1,
        explanation: 'In traditional programming, humans write explicit rules. In ML, the algorithm infers patterns and mathematical rules from data.',
        misconception: 'Students often believe ML is simply a very large collection of if-else statements programmed by engineers.',
      },
      {
        id: 'q1_2',
        question: 'Which of the following is the best candidate for Machine Learning rather than rule-based code?',
        options: [
          'Calculating 8% sales tax on an e-commerce receipt',
          'Converting temperatures from Fahrenheit to Celsius',
          'Recognizing handwritten digits on postal envelopes',
          'Checking if a user password has at least 8 characters',
        ],
        correctIndex: 2,
        explanation: 'Recognizing diverse handwriting variations involves non-linear visual patterns that are impossible to hand-craft with simple rules, making it an ideal ML problem.',
        misconception: 'Believing that everything should use ML; simple mathematical conversions should always use exact deterministic formulas.',
      },
    ],
  },
  {
    id: 'm1_ml_types',
    moduleId: 1,
    moduleTitle: 'Foundations',
    order: 2,
    title: 'Supervised vs Unsupervised Learning',
    summary: 'Master the core dichotomy of ML: learning with a teacher (labels) vs discovering hidden structure.',
    simpleExplanation:
      'Supervised Learning is like studying flashcards that have the question on the front and the answer on the back—the teacher tells you if you are right. Unsupervised Learning is like being handed a bucket of mixed foreign coins with no labels and sorting them into groups by their color, size, and weight.',
    technicalDefinition:
      'Supervised learning optimizes a parameterized hypothesis on a labeled training set D = {(x_i, y_i)}. Unsupervised learning models the underlying probability density or geometric clustering of unlabeled data D = {x_i}.',
    realWorldExample:
      'Supervised: Predicting house sale prices based on square footage and location (labeled with actual sale price). Unsupervised: Segmenting online shoppers into distinct personality clusters based on clickstream behavior without predefined categories.',
    stepByStep: [
      '1. Check your data: Does each row have a target "ground truth" answer column?',
      '2. If YES: You are doing Supervised Learning (predicting continuous numbers = Regression, predicting categories = Classification).',
      '3. If NO: You are doing Unsupervised Learning (grouping similar items = Clustering, compressing features = PCA).',
    ],
    codeSnippet: `# Supervised: Has labels (y)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

# Unsupervised: No labels, only features (X)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X_features)`,
    keyTakeaways: [
      'Supervised learning requires ground-truth labels (targets).',
      'Unsupervised learning discovers natural geometric clusters or latent structures without targets.',
      'Regression predicts numbers; Classification predicts categories.',
    ],
    practicalTask: 'Navigate to Dataset Explorer. Inspect the "Customer Segmentation" dataset and note that it has no target class—it is made for unsupervised clustering!',
    quizQuestions: [
      {
        id: 'q1_3',
        question: 'If you want to group hospital patients into 4 health cohorts without any pre-existing diagnosis labels, what type of ML is this?',
        options: ['Supervised Classification', 'Supervised Regression', 'Unsupervised Clustering', 'Reinforcement Gaming'],
        correctIndex: 2,
        explanation: 'Because there are no ground-truth target labels provided, grouping records based on feature similarity is Unsupervised Clustering.',
        misconception: 'Assuming that any grouping task is classification; without labels, it is clustering.',
      },
    ],
  },
  {
    id: 'm2_scaling_leakage',
    moduleId: 2,
    moduleTitle: 'Data Fundamentals',
    order: 3,
    title: 'Feature Scaling & Data Leakage',
    summary: 'Why feature magnitudes distort distance-based models and how to prevent test-set contamination.',
    simpleExplanation:
      'Imagine measuring human height in millimeters (e.g., 1800 mm) and age in years (e.g., 25 years). If an algorithm calculates distance, the 1800 mm height completely overpowers the 25 years, making age practically invisible! Feature scaling puts all features on an equal playing field.',
    technicalDefinition:
      'Feature scaling normalizes numeric ranges using StandardScaler (z = (x - μ) / σ) or MinMaxScaler (x_norm = (x - min) / (max - min)). Data leakage occurs when statistics from the test set contaminate the training transformation.',
    realWorldExample:
      'In KNN or SVM, if Annual Income is $80,000 and Number of Children is 2, the Euclidean distance will be 99.99% driven by income unless both are scaled to mean 0 and unit variance.',
    stepByStep: [
      '1. Split raw data into Train and Test sets FIRST.',
      '2. Calculate mean (μ) and standard deviation (σ) on the Training set ONLY.',
      '3. Apply that exact transformation to both the Training and Test sets.',
      '4. Never calculate mean or min/max using the combined whole dataset before splitting!',
    ],
    codeSnippet: `# Correct Leakage-Free Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit ONLY on X_train!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform X_test using parameters learned from X_train
X_test_scaled = scaler.transform(X_test)`,
    keyTakeaways: [
      'Distance-based models (KNN, K-Means, SVM, PCA) are highly sensitive to feature scales.',
      'Tree-based models (Decision Trees, Random Forests) are invariant to monotonic scale transformations.',
      'Data leakage artificially inflates validation scores but fails catastrophically in production.',
    ],
    practicalTask: 'In the Simulator, train KNN on the Diabetes dataset with and without StandardScaler to observe the difference in prediction accuracy.',
    quizQuestions: [
      {
        id: 'q2_1',
        question: 'Why must feature scalers (like StandardScaler) be fitted only on the training set?',
        options: [
          'Because the test set does not contain numeric features',
          'To prevent data leakage by keeping test set distribution information hidden during training',
          'Because scikit-learn will crash if you fit on test data',
          'Scaling test data is completely unnecessary',
        ],
        correctIndex: 1,
        explanation: 'Fitting transformers on test data exposes future distribution statistics (mean, std) to the training pipeline, causing optimistic bias (data leakage).',
        misconception: 'Many beginners scale the entire CSV before splitting, invalidating their test evaluation.',
      },
    ],
  },
  {
    id: 'm3_linear_regression',
    moduleId: 3,
    moduleTitle: 'Supervised Learning',
    order: 4,
    title: 'Linear Regression & Least Squares',
    summary: 'The foundational regression algorithm: fitting the line of best fit by minimizing squared residuals.',
    simpleExplanation:
      'Linear regression draws a straight line through a scatter plot of dots. It searches for the slope (angle) and intercept (starting height) that makes the total distance from the line to all points as small as possible.',
    technicalDefinition:
      'Linear Regression models a linear relationship between input vector x and scalar target y = w · x + b by minimizing the Ordinary Least Squares (OLS) objective: J(w, b) = (1/2m) ∑ (ŷ_i - y_i)², solved analytically via β = (XᵀX)⁻¹ Xᵀy or iteratively via gradient descent.',
    realWorldExample:
      'Predicting taxi fare: Base fare is $3.00 (intercept) plus $2.50 per mile traveled (slope). Total Fare = 2.50 * Miles + 3.00.',
    stepByStep: [
      '1. Plot each sample as a point (x, y) on a Cartesian grid.',
      '2. Start with an initial guess for slope m and intercept b.',
      '3. For each point, measure the vertical distance (residual) between actual y and line prediction ŷ.',
      '4. Square all residuals to make negative errors positive and penalize large errors heavily.',
      '5. Sum all squared residuals (Sum of Squared Errors).',
      '6. Use calculus (or the Normal Equation) to find the exact m and b that minimize this sum.',
    ],
    codeSnippet: `from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Slope: {model.coef_[0]:.3f}, Intercept: {model.intercept_:.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")`,
    keyTakeaways: [
      'Equation: y = mx + b (or y = w · x + b in multiple dimensions).',
      'Residual = Actual y - Predicted y.',
      'OLS minimizes the sum of squared residuals.',
      'R² measures the proportion of variance in the target explained by the model (1.0 is perfect, 0.0 means no better than predicting the mean).',
    ],
    practicalTask: 'Open the Simulator with Linear Regression on the Study Hours dataset. Step through the gradient descent iterations to watch the line rotate into place.',
    quizQuestions: [
      {
        id: 'q3_1',
        question: 'In the linear regression formula y = 3.5x + 12, what does the number 3.5 represent?',
        options: [
          'The value of y when x is 0',
          'The slope: for every 1-unit increase in x, y increases by 3.5 units',
          'The total error percentage of the model',
          'The number of samples in the dataset',
        ],
        correctIndex: 1,
        explanation: 'The slope (coefficient) represents the rate of change in the target y for each one-unit increase in feature x.',
        misconception: 'Confusing slope (rate of change) with intercept (starting value when x=0).',
      },
      {
        id: 'q3_2',
        question: 'Why do we square the residuals (errors) in Ordinary Least Squares instead of just adding them up?',
        options: [
          'Because negative errors and positive errors would cancel each other out, and squaring heavily penalizes large errors',
          'Because computers cannot add negative numbers',
          'Squaring makes the dataset larger',
          'It is required by the Python language syntax',
        ],
        correctIndex: 0,
        explanation: 'If errors are -5 and +5, simple addition gives 0 error. Squaring makes all errors positive (25 + 25 = 50) and penalizes bigger mistakes more severely.',
        misconception: 'Thinking that sum of raw errors works for measuring fit.',
      },
    ],
  },
  {
    id: 'm3_decision_trees',
    moduleId: 3,
    moduleTitle: 'Supervised Learning',
    order: 5,
    title: 'Decision Trees & Impurity',
    summary: 'Hierarchical if-then rules built using Gini Impurity and Entropy.',
    simpleExplanation:
      'A Decision Tree is like playing a game of 20 Questions. At each turn, it asks the single question that divides the possibilities into the cleanest, most uniform groups (e.g., "Is petal length ≤ 2.45 cm?").',
    technicalDefinition:
      'Decision Trees partition feature space recursively via greedy orthogonal splits that maximize Information Gain: Gain = I(parent) - ∑ (N_child / N_parent) · I(child), where impurity I is Gini (1 - ∑ p_i²) or Entropy (-∑ p_i log₂ p_i).',
    realWorldExample:
      'Loan Approval: 1. Is credit score > 700? (If yes, approve). 2. If no, is annual income > $60,000? (If yes, approve; if no, reject).',
    stepByStep: [
      '1. Start at root node containing all training samples.',
      '2. Calculate initial impurity (Gini or Entropy) of the target labels.',
      '3. Test every possible split threshold across all features.',
      '4. Pick the feature and threshold that yields the highest Information Gain.',
      '5. Split the data into Left and Right children and repeat recursively.',
      '6. Stop when max depth is reached or all samples in a leaf belong to the same class.',
    ],
    codeSnippet: `from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=3, criterion='gini')
tree.fit(X_train, y_train)

# Inspect feature splits
print(f"Feature importances: {tree.feature_importances_}")`,
    keyTakeaways: [
      'Gini = 0 means a node is pure (100% of samples belong to one class).',
      'Unconstrained trees quickly overfit by memorizing individual training noise.',
      'Hyperparameters like max_depth and min_samples_split prevent trees from becoming too complex.',
    ],
    practicalTask: 'Simulate Decision Tree on the Iris dataset. Notice how a single split on petal_length perfectly isolates Setosa with Gini = 0.0!',
    quizQuestions: [
      {
        id: 'q3_3',
        question: 'What does a Gini Impurity of 0.0 indicate for a decision tree node?',
        options: [
          'The node contains an equal mix of all classes',
          'The node is completely pure: all samples belong to exactly one class',
          'The model has failed to train',
          'The tree depth has reached infinity',
        ],
        correctIndex: 1,
        explanation: 'Gini = 1 - ∑ p_i². If all samples are Class A, p_A = 1.0, so Gini = 1 - (1.0)² = 0.0 (pure node).',
        misconception: 'Confusing 0 impurity with 0 accuracy; 0 impurity is the most desirable, cleanest state for a leaf.',
      },
    ],
  },
  {
    id: 'm4_kmeans_clustering',
    moduleId: 4,
    moduleTitle: 'Unsupervised Learning',
    order: 6,
    title: 'K-Means Clustering & The Elbow Method',
    summary: 'Partitioning unlabeled points into K clusters using iterative centroid updates.',
    simpleExplanation:
      'Imagine K party hosts standing in a room. Every guest walks over to stand next to whichever host is closest. Then, each host moves to the center of their new group of guests. Guests switch hosts if another is now closer. This repeats until nobody needs to move!',
    technicalDefinition:
      'K-Means partitions n observations into k clusters by minimizing Within-Cluster Sum of Squares (Inertia): WCSS = ∑_k ∑_{x ∈ C_k} ||x - μ_k||² via alternating expectation (point assignment) and maximization (centroid re-estimation).',
    realWorldExample:
      'Uber dispatching: Identifying where riders congregate in a city to position drivers efficiently at optimal centroid hubs.',
    stepByStep: [
      '1. Choose K (number of clusters).',
      '2. Initialize K centroids (using K-Means++ to spread them out).',
      '3. Assignment Step: Measure distance from each data point to all K centroids; assign point to the nearest one.',
      '4. Update Step: Recalculate each centroid as the geometric mean of all points assigned to it.',
      '5. Repeat steps 3 and 4 until centroids stop moving (convergence).',
      '6. Use the Elbow Method (plotting Inertia vs K) to find where adding more clusters gives diminishing returns.',
    ],
    codeSnippet: `from sklearn.cluster import KMeans

# Train K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X)

print(f"Centroids:\\n{kmeans.cluster_centers_}")
print(f"Final Inertia (WCSS): {kmeans.inertia_:.2f}")`,
    keyTakeaways: [
      'K-Means requires you to specify K in advance.',
      'Inertia always decreases as K increases; the "elbow" indicates optimal balance of simplicity and tightness.',
      'K-Means assumes spherical, equally-sized clusters and is sensitive to outliers and scaling.',
    ],
    practicalTask: 'Simulate K-Means on "Synthetic Blobs". Watch the centroids move frame by frame and see Inertia drop to a minimum.',
    quizQuestions: [
      {
        id: 'q4_1',
        question: 'In the Elbow Method for K-Means, how do you identify the recommended number of clusters (K)?',
        options: [
          'Choose the K value where inertia is highest',
          'Look for the point where inertia drops sharply and then levels off into an "elbow" shape',
          'Always choose K equal to the number of rows in the dataset',
          'Select the K where centroids never move',
        ],
        correctIndex: 1,
        explanation: 'The elbow point marks the sweet spot where adding further clusters yields diminishing returns in reducing within-cluster variance.',
        misconception: 'Thinking that K with the absolute lowest inertia is best (setting K=N gives 0 inertia but no useful clustering).',
      },
    ],
  },
  {
    id: 'm5_evaluation_metrics',
    moduleId: 5,
    moduleTitle: 'Model Evaluation',
    order: 7,
    title: 'Classification Metrics & Confusion Matrix',
    summary: 'Why Accuracy is deceptive on imbalanced data and how to use Precision, Recall, and F1-Score.',
    simpleExplanation:
      'Imagine a rare disease affects 1 in 1,000 people. A lazy doctor who diagnoses "Healthy" for EVERY single patient is 99.9% accurate! But they are a terrible doctor because they caught 0 sick patients. That is why we need Precision and Recall, not just Accuracy.',
    technicalDefinition:
      'Accuracy = (TP + TN) / Total. Precision = TP / (TP + FP) (purity of positive predictions). Recall = TP / (TP + FN) (sensitivity to actual positives). F1-Score = 2 · (Precision · Recall) / (Precision + Recall) (harmonic mean).',
    realWorldExample:
      'Credit card fraud detection: You care far more about High Recall (catching every fraudulent charge, even if a few legitimate ones are flagged for review) than pure accuracy.',
    stepByStep: [
      '1. Construct a 2x2 Confusion Matrix:',
      '   - True Positive (TP): Predicted Positive, Actually Positive.',
      '   - False Positive (FP): Predicted Positive, Actually Negative (Type I error).',
      '   - True Negative (TN): Predicted Negative, Actually Negative.',
      '   - False Negative (FN): Predicted Negative, Actually Positive (Type II error).',
      '2. Calculate Precision: Out of all alarms raised, how many were real fires?',
      '3. Calculate Recall: Out of all real fires, how many alarms were raised?',
      '4. Combine them into F1-score when both matter equally.',
    ],
    codeSnippet: `from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Precision, Recall, F1 for each class
print(classification_report(y_true, y_pred))`,
    keyTakeaways: [
      'Never rely solely on Accuracy when classes are imbalanced.',
      'High Precision means low false alarms (good for spam filters).',
      'High Recall means low missed positives (vital for medical screening and fraud detection).',
      'F1-score penalizes models with extreme tradeoffs.',
    ],
    practicalTask: 'In the Simulator with Logistic Regression on Exam Result Predictor, drag the decision threshold slider from 0.5 to 0.2 and observe how Recall increases while Precision drops!',
    quizQuestions: [
      {
        id: 'q5_1',
        question: 'In a medical test for a deadly contagious illness, which metric is most critical to maximize to ensure no infected patients are sent home?',
        options: ['Precision', 'Recall', 'Accuracy', 'Training Speed'],
        correctIndex: 1,
        explanation: 'Recall = TP / (TP + FN). Maximizing recall minimizes False Negatives (infected patients incorrectly diagnosed as healthy).',
        misconception: 'Assuming high accuracy automatically guarantees safety in critical diagnosis.',
      },
    ],
  },
];
