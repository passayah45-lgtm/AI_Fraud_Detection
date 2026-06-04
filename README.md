# AI-Powered Financial Fraud Detection System

## Project Overview
A complete end-to-end machine learning system that detects fraudulent credit card transactions using a trained Random Forest classifier, with a Streamlit web application, SQLite database, and analytics dashboard.

## Dataset
- Source: Kaggle Credit Card Fraud Detection dataset
- 284,807 transactions (2013, European cardholders)
- 30 features: Time, V1-V28 (PCA), Amount
- Fraud rate: 0.17% (highly imbalanced)

## Model Performance
| Model | F1 % | ROC AUC % | Recall % |
|---|---|---|---|
| Random Forest | 68.78 | 97.95 | 80.00 |
| XGBoost | 32.26 | 97.26 | 84.21 |
| Logistic Regression | 9.95 | 96.31 | 87.37 |

**Best model: Random Forest**

## Features
- Real-time fraud prediction with confidence gauge
- Batch CSV upload and prediction
- Analytics dashboard with Plotly charts
- Prediction history stored in SQLite
- Model comparison table
- Retraining workflow interface

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, SMOTE
- Streamlit, Plotly, Seaborn
- SQLite, Joblib

## Run Locally
```bash
pip install -r requirements.txt
python train_models2.py   # builds models and database
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Set main file to app.py
5. Deploy

## Project Structure
```
AI_Fraud_Detection/
  app.py               # Streamlit application
  train_models2.py     # Full ML training pipeline
  requirements.txt     # Dependencies
  data/                # Dataset files
  models/              # Trained model artifacts
  database/            # SQLite database
  reports/             # Charts and evaluation plots
```

## Author
Mister T | KIIT University | Master's in Data Analysis
