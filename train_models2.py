import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns, sqlite3, random
from datetime import datetime, timedelta
import os

df = pd.read_csv('data/cleaned_creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42, k_neighbors=3)
X_tr, y_tr = smote.fit_resample(X_train, y_train)
print(f"SMOTE done. Samples: {len(X_tr):,}")

# Train RF and XGB only (LR already logged)
results = []
models = {}

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
yp = rf.predict(X_test)
yprob = rf.predict_proba(X_test)[:,1]
r = dict(Model='Random Forest',
         Accuracy =round(accuracy_score(y_test,yp)*100,2),
         Precision=round(precision_score(y_test,yp)*100,2),
         Recall   =round(recall_score(y_test,yp)*100,2),
         F1       =round(f1_score(y_test,yp)*100,2),
         ROC_AUC  =round(roc_auc_score(y_test,yprob)*100,2))
results.append(r)
models['Random Forest'] = rf
print(f"  RF: F1={r['F1']}% | AUC={r['ROC_AUC']}%")

print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                    eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_tr, y_tr)
yp = xgb.predict(X_test)
yprob = xgb.predict_proba(X_test)[:,1]
r = dict(Model='XGBoost',
         Accuracy =round(accuracy_score(y_test,yp)*100,2),
         Precision=round(precision_score(y_test,yp)*100,2),
         Recall   =round(recall_score(y_test,yp)*100,2),
         F1       =round(f1_score(y_test,yp)*100,2),
         ROC_AUC  =round(roc_auc_score(y_test,yprob)*100,2))
results.append(r)
models['XGBoost'] = xgb
print(f"  XGB: F1={r['F1']}% | AUC={r['ROC_AUC']}%")

# Add LR result from previous run (known)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_tr, y_tr)
yp = lr.predict(X_test)
yprob = lr.predict_proba(X_test)[:,1]
r_lr = dict(Model='Logistic Regression',
         Accuracy =round(accuracy_score(y_test,yp)*100,2),
         Precision=round(precision_score(y_test,yp)*100,2),
         Recall   =round(recall_score(y_test,yp)*100,2),
         F1       =round(f1_score(y_test,yp)*100,2),
         ROC_AUC  =round(roc_auc_score(y_test,yprob)*100,2))
models['Logistic Regression'] = lr
results.append(r_lr)

rdf = pd.DataFrame(results)
rdf.to_csv('models/model_metrics.csv', index=False)
print("\nAll results:")
print(rdf.to_string(index=False))

best_name = rdf.loc[rdf['F1'].idxmax(), 'Model']
print(f"\nBest model: {best_name}")
final_model = models[best_name]

with open('models/best_model_name.txt','w') as f:
    f.write(best_name)
joblib.dump(final_model, 'models/fraud_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
print("Saved fraud_model.pkl")

# Evaluation plots
y_pred_f  = final_model.predict(X_test)
y_prob_f  = final_model.predict_proba(X_test)[:,1]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f'Model Evaluation: {best_name}', fontsize=13, fontweight='bold')
cm = confusion_matrix(y_test, y_pred_f)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
axes[0].set_title('Confusion Matrix'); axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
fpr, tpr, _ = roc_curve(y_test, y_prob_f)
auc_val = roc_auc_score(y_test, y_prob_f)
axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC={auc_val:.4f}')
axes[1].plot([0,1],[0,1],'--', color='gray')
axes[1].set_title('ROC Curve'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend()
prec, rec, _ = precision_recall_curve(y_test, y_prob_f)
ap = average_precision_score(y_test, y_prob_f)
axes[2].plot(rec, prec, color='#3498db', lw=2, label=f'AP={ap:.4f}')
axes[2].set_title('Precision-Recall Curve'); axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].legend()
plt.tight_layout()
plt.savefig('reports/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved model_evaluation.png")

# Database
conn = sqlite3.connect('database/fraud.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    amount REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
scaler = joblib.load('models/scaler.pkl')
random.seed(42)
sample_idx = random.sample(range(len(X_test)), min(300, len(X_test)))
X_samp = X_test.iloc[sample_idx]
p_samp = final_model.predict_proba(X_samp)[:,1]
pred_samp = (p_samp >= 0.5).astype(int)
amt_approx = X_samp['scaled_amount'].values * scaler['amount'].scale_[0] + scaler['amount'].mean_[0]
base_time = datetime.now() - timedelta(days=30)
for i, (pred, prob, amt) in enumerate(zip(pred_samp, p_samp, amt_approx)):
    ts = (base_time + timedelta(hours=i*2.4)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO predictions (prediction,probability,amount,timestamp) VALUES (?,?,?,?)',
              (int(pred), float(prob), float(amt), ts))
conn.commit()
conn.close()
print(f"Database seeded with {len(sample_idx)} records")
print("\n=== COMPLETE ===")
