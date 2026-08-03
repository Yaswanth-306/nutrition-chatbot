AI-Powered Nutrition Chatbot
 
Machine learning-based nutrition chatbot that predicts post-prandial glycemic risk and provides personalized dietary recommendations.
 
 
ARCHITECTURE
 
ML Model: XGBoost (91% accuracy)
API: Flask REST API
Workflow: n8n orchestration
Frontend: HTML/JavaScript chatbot
Proxy: Node.js CORS proxy
 
 
FEATURES
 
Predicts 2-hour post-prandial glucose risk
40 engineered features from 12 base measurements
Explainable AI with feature importance
Personalized meal recommendations
Support for dietary restrictions and allergies
 
 
SETUP
 
Step 1: Install Dependencies
 
Python dependencies
pip install -r requirements.txt
 
Node.js dependencies (for proxy)
npm install
 
n8n (global installation)
npm install -g n8n
 
Step 2: Start Services
 
Terminal 1, Flask API
python api/ml_api_updated.py
 
Terminal 2, n8n
n8n start
 
Terminal 3, Proxy Server
node api/proxy_new.js
 
Terminal 4, HTML Server
python -m http.server 8000
 
Step 3: Import n8n Workflow
 
Open n8n at http://localhost:5678
Go to Workflows, then Import from File
Select the file n8n/nutrition-chatbot-workflow.json
Activate the workflow
 
Step 4: Access Chatbot
 
Open http://localhost:8000/frontend/nutrition-chatbot-final.html
 
 
MODEL PERFORMANCE
 
Cross-validation AUC: 0.8395 (plus or minus 0.0613)
Test AUC: 0.7809
Accuracy: 91%
Dataset: 1,050 samples
Features: 40, engineered from 12 base features
 
 
FILES
 
api/ml_api_updated.py: Flask ML API
api/proxy_new.js: Node.js CORS proxy
models/trained_model_1.pkl: Trained XGBoost model
n8n/nutrition-chatbot-workflow.json: n8n workflow
frontend/nutrition-chatbot-final.html: Chatbot UI
training/setup_project_new.py: Model training script
 
 

 
 
LICENSE
 
MIT
