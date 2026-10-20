# Python FastAPI Machine Learning Backend Service

This directory provides the production Scikit-Learn / FastAPI microservice for the **ML Algorithm Simulator**.

## Architecture

The ML Algorithm Simulator features a dual-engine architecture:
1. **Primary Built-in Mathematical Engine**: Runs 100% locally in TypeScript using closed-form linear algebra, analytical least-squares, recursive Gini decision trees, and Lloyd's K-Means clustering. This ensures zero external dependencies and immediate execution anywhere.
2. **External Scikit-Learn Python Backend**: Uses FastAPI, Pandas, NumPy, and Scikit-learn to run native Python estimators.

## Running Locally

1. Create a Python virtual environment:
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Launch FastAPI server on port 8000:
\`\`\`bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

4. Verify health status in browser:
\`\`\`
http://localhost:8000/api/health
\`\`\`

## Docker Deployment

Build and run via Docker:
\`\`\`bash
docker build -t ml-simulator-backend .
docker run -p 8000:8000 ml-simulator-backend
\`\`\`
