\# AI-Powered Nutrition Chatbot



Machine learning-based nutrition chatbot that predicts post-prandial glycemic risk and provides personalized dietary recommendations.



\## Architecture



\- \*\*ML Model:\*\* XGBoost (91% accuracy)

\- \*\*API:\*\* Flask REST API

\- \*\*Workflow:\*\* n8n orchestration

\- \*\*Frontend:\*\* HTML/JavaScript chatbot

\- \*\*Proxy:\*\* Node.js CORS proxy



\## Features



\- Predicts 2-hour post-prandial glucose risk

\- 40 engineered features from 12 base measurements

\- Explainable AI with feature importance

\- Personalized meal recommendations

\- Support for dietary restrictions and allergies



\## Setup



\### 1. Install Dependencies

```bash

\# Python dependencies

pip install -r requirements.txt



\# Node.js dependencies (for proxy)

npm install



\# n8n (global installation)

npm install -g n8n

```



\### 2. Start Services

```bash

\# Terminal 1: Flask API

python api/ml\_api\_updated.py



\# Terminal 2: n8n

n8n start



\# Terminal 3: Proxy Server

node api/proxy\_new.js



\# Terminal 4: HTML Server

python -m http.server 8000

```



\### 3. Import n8n Workflow



1\. Open n8n: http://localhost:5678

2\. Go to Workflows → Import from File

3\. Select `n8n/nutrition-chatbot-workflow.json`

4\. Activate the workflow



\### 4. Access Chatbot



Open: http://localhost:8000/frontend/nutrition-chatbot-final.html



\## Model Performance



\- \*\*Cross-validation AUC:\*\* 0.8395 (±0.0613)

\- \*\*Test AUC:\*\* 0.7809

\- \*\*Accuracy:\*\* 91%

\- \*\*Dataset:\*\* 1,050 samples

\- \*\*Features:\*\* 40 (engineered from 12 base features)



\## Files



\- `api/ml\_api\_updated.py` - Flask ML API

\- `api/proxy\_new.js` - Node.js CORS proxy

\- `models/trained\_model\_1.pkl` - Trained XGBoost model

\- `n8n/nutrition-chatbot-workflow.json` - n8n workflow

\- `frontend/nutrition-chatbot-final.html` - Chatbot UI

\- `training/setup\_project\_new.py` - Model training script



\## Deployment



See deployment documentation for AWS EC2, Render, or Railway deployment instructions.



\## License



MIT

