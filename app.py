import streamlit as st
import pandas as pd
import numpy as np
import joblib, sqlite3, os, shutil
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# CONFIG 
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DB_PATH    = os.path.join(BASE_DIR, 'database', 'fraud.db')
RPT_DIR    = os.path.join(BASE_DIR, 'reports')
ORIG_MODEL = os.path.join(MODEL_DIR, 'fraud_model_original.pkl')

# RISK PROFILE THRESHOLDS 
RISK_PROFILES = {
    "🟢  Low Risk":    {"threshold": 0.3, "desc": "Catches more fraud. Expect more false alerts.", "color": "#2ecc71"},
    "🟡  Medium Risk": {"threshold": 0.5, "desc": "Balanced. Recommended for most cases.",          "color": "#f39c12"},
    "🔴  High Risk":   {"threshold": 0.7, "desc": "Only flags very likely fraud. Fewer alerts.",     "color": "#e74c3c"},
}

# LOAD RESOURCES 
@st.cache_resource
def load_model():
    model   = joblib.load(os.path.join(MODEL_DIR, 'fraud_model.pkl'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feats   = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    metrics = pd.read_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'))
    with open(os.path.join(MODEL_DIR, 'best_model_name.txt')) as f:
        best_name = f.read().strip()
    return model, scaler, feats, metrics, best_name

@st.cache_data
def get_db_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def save_prediction(prediction, probability, amount):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (prediction, probability, amount) VALUES (?,?,?)',
              (int(prediction), float(probability), float(amount)))
    conn.commit()
    conn.close()

def reset_to_original():
    # Reset database to first 300 seeded records
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE id > 300')
    conn.commit()
    conn.close()
    # Restore original model if backup exists
    if os.path.exists(ORIG_MODEL):
        shutil.copy(ORIG_MODEL, os.path.join(MODEL_DIR, 'fraud_model.pkl'))
        # Restore original metrics
        orig_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'model_metrics_original.csv'))
        orig_metrics.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
        with open(os.path.join(MODEL_DIR, 'best_model_name_original.txt')) as f:
            orig_name = f.read().strip()
        with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'w') as f:
            f.write(orig_name)
        load_model.clear()

def detect_fraud_column(df):
    """Auto-detect the fraud/label column regardless of its name."""
    candidates = ['Class', 'class', 'is_fraud', 'fraud', 'Fraud', 'label',
                  'Label', 'target', 'Target', 'isFraud', 'is_Fraud']
    for c in candidates:
        if c in df.columns:
            return c
    # Try any binary column with 0/1 values
    for c in df.columns:
        try:
            unique_vals = df[c].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                return c
        except:
            pass
    return None

try:
    model, scaler, feature_names, metrics_df, best_model_name = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Model loading failed: {e}")

# SIDEBAR 
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lock.png", width=60)
    st.title("Fraud Detection")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "🔍  Fraud Detection",
        "📊  Dashboard",
        "📋  History",
        "🔄  Retraining"
    ])
    st.markdown("---")
    if MODEL_LOADED:
        st.success(f"Model: **{best_model_name}**")
        best_row = metrics_df.loc[metrics_df['F1'].idxmax()]
        st.metric("F1 Score", f"{best_row['F1']:.2f}%")
        st.metric("ROC AUC",  f"{best_row['ROC_AUC']:.2f}%")
        st.metric("Recall",   f"{best_row['Recall']:.2f}%")
    st.markdown("---")
    st.subheader("Reset Data")
    st.caption("Restores original model and 300 seeded records.")
    if st.button("Reset to Original Data", use_container_width=True, type="primary"):
        reset_to_original()
        get_db_predictions.clear()
        st.success("Reset complete.")
        st.rerun()

# PAGE: HOME 
if "Home" in page:
    st.title("🔒 AI-Powered Financial Fraud Detection System")
    st.markdown("### Protecting transactions with machine learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    preds = get_db_predictions()
    total  = len(preds)
    frauds = preds['prediction'].sum()
    legits = total - frauds

    col1.metric("Total Predictions", f"{total:,}")
    col2.metric("Fraud Detected",    f"{frauds:,}", delta=f"{frauds/max(total,1)*100:.1f}%", delta_color="inverse")
    col3.metric("Legitimate",        f"{legits:,}")
    col4.metric("Model AUC",         f"{metrics_df.loc[metrics_df['F1'].idxmax(),'ROC_AUC']:.2f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("What This System Does")
        st.markdown("""
- **Detects fraud** in real-time using a trained Random Forest classifier
- **Stores** every prediction in SQLite for audit trails
- **Visualizes** transaction trends and fraud patterns
- **Supports retraining** with any new labeled dataset
- **Dataset**: 284,807 European credit card transactions (2013)
- **Fraud rate**: 0.17% (highly imbalanced, handled with SMOTE)
        """)
    with col_b:
        st.subheader("Technology Stack")
        tech = {
            "Machine Learning": "Scikit-learn, XGBoost, SMOTE",
            "Data Processing":  "Pandas, NumPy",
            "Visualization":    "Plotly, Seaborn, Matplotlib",
            "Application":      "Streamlit",
            "Database":         "SQLite",
            "Deployment":       "Streamlit Community Cloud",
        }
        for k, v in tech.items():
            st.markdown(f"**{k}**: {v}")

    st.markdown("---")
    st.subheader("Model Comparison")
    metrics_display = metrics_df[['Model','Accuracy','Precision','Recall','F1','ROC_AUC']].copy()
    metrics_display.columns = ['Model','Accuracy %','Precision %','Recall %','F1 %','ROC AUC %']
    st.dataframe(metrics_display.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

    st.markdown("---")
    if os.path.exists(f'{RPT_DIR}/eda_dashboard.png'):
        st.subheader("Exploratory Data Analysis")
        st.image(f'{RPT_DIR}/eda_dashboard.png', use_column_width=True)
    if os.path.exists(f'{RPT_DIR}/model_evaluation.png'):
        st.subheader("Model Evaluation Charts")
        st.image(f'{RPT_DIR}/model_evaluation.png', use_column_width=True)

#  PAGE: FRAUD DETECTION 
elif "Detection" in page:
    st.title("🔍 Fraud Detection")
    st.markdown("### Predict whether a transaction is fraudulent")
    st.markdown("---")

    # RISK PROFILE SELECTOR 
    st.subheader("Step 1: Choose Your Risk Profile")
    risk_choice = st.radio(
        "Risk Profile",
        list(RISK_PROFILES.keys()),
        horizontal=True,
        help="This controls how sensitive the fraud detector is."
    )
    rp        = RISK_PROFILES[risk_choice]
    threshold = rp["threshold"]

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Risk Profile",        risk_choice.split("  ")[1])
    rc2.metric("Decision Threshold",  f"{threshold:.0%}")
    rc3.metric("Mode",                rp["desc"].split(".")[0])
    st.markdown(f"> {rp['desc']}")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

    with tab1:
        st.markdown("#### Step 2: Enter Transaction Features")
        st.info("V1-V28 are PCA-transformed features from the original dataset.")

        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
            time_s = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)
        with col2:
            st.markdown(f"**Active threshold:** `{threshold}` ({risk_choice.split('  ')[1]})")
            st.markdown(f"**Meaning:** A transaction is flagged as fraud if its probability exceeds `{threshold:.0%}`")

        fraud_signature = {'V14': -10.0, 'V4': 4.5, 'V11': -3.5, 'V12': -5.0, 'V10': -4.0}
        use_demo = st.checkbox("Load a fraud demo signature", value=False)
        v_cols = [c for c in feature_names if c.startswith('V')]
        rows   = [v_cols[i:i+7] for i in range(0, len(v_cols), 7)]
        v_vals = {}
        for row in rows:
            cols_r = st.columns(len(row))
            for i, feat in enumerate(row):
                default = fraud_signature.get(feat, 0.0) if use_demo else 0.0
                v_vals[feat] = cols_r[i].number_input(feat, value=default, format="%.4f", key=feat)

        if st.button("Predict Transaction", type="primary", use_container_width=True):
            sc_amount  = scaler['amount'].transform([[amount]])[0][0]
            sc_time    = scaler['time'].transform([[time_s]])[0][0]
            input_data = {}
            for f in feature_names:
                if f == 'scaled_amount': input_data[f] = sc_amount
                elif f == 'scaled_time': input_data[f] = sc_time
                else:                    input_data[f] = v_vals.get(f, 0.0)
            input_df = pd.DataFrame([input_data])[feature_names]
            proba    = model.predict_proba(input_df)[0][1]
            pred     = 1 if proba >= threshold else 0

            # Risk badge
            if proba < 0.3:
                risk_label = "🟢 LOW RISK"
                risk_color = "#2ecc71"
            elif proba < 0.6:
                risk_label = "🟡 MEDIUM RISK"
                risk_color = "#f39c12"
            else:
                risk_label = "🔴 HIGH RISK"
                risk_color = "#e74c3c"

            save_prediction(pred, proba, amount)
            get_db_predictions.clear()

            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Verdict",          "FRAUD" if pred == 1 else "LEGITIMATE")
            col_r2.metric("Fraud Probability", f"{proba*100:.2f}%")
            col_r3.metric("Risk Level",        risk_label.split(" ",1)[1])

            if pred == 1:
                st.error(f"### ⚠️ FRAUD DETECTED    {risk_label}")
            else:
                st.success(f"### ✅ LEGITIMATE TRANSACTION    {risk_label}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': risk_color},
                    'steps': [
                        {'range': [0,  30],  'color': '#d4edda'},
                        {'range': [30, 60],  'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 4},
                                  'thickness': 0.75, 'value': threshold * 100}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Step 2: Upload any CSV file")
        st.info(f"Using **{risk_choice.split('  ')[1]}** profile  threshold {threshold:.0%}. Missing columns filled automatically.")
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"Uploaded: {df_up.shape[0]:,} rows x {df_up.shape[1]} columns")
            st.write(f"Columns found: {', '.join(df_up.columns.tolist())}")

            X_up = pd.DataFrame(0.0, index=range(len(df_up)), columns=feature_names)
            if 'Amount' in df_up.columns:
                X_up['scaled_amount'] = scaler['amount'].transform(df_up[['Amount']])
            if 'Time' in df_up.columns:
                X_up['scaled_time'] = scaler['time'].transform(df_up[['Time']])
            for col in feature_names:
                if col in df_up.columns:
                    X_up[col] = df_up[col].fillna(0).values

            probas = model.predict_proba(X_up)[:, 1]
            preds  = (probas >= threshold).astype(int)

            # Risk labels
            def risk_label(p):
                if p < 0.3:   return "🟢 Low Risk"
                elif p < 0.6: return "🟡 Medium Risk"
                else:         return "🔴 High Risk"

            df_up['Fraud_Probability_%'] = np.round(probas * 100, 2)
            df_up['Prediction']          = preds
            df_up['Verdict']             = df_up['Prediction'].map({0: 'Legitimate', 1: 'FRAUD'})
            df_up['Risk_Level']          = [risk_label(p) for p in probas]

            st.dataframe(df_up[['Fraud_Probability_%', 'Verdict', 'Risk_Level']].head(50), use_container_width=True)

            fraud_count = preds.sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", f"{len(preds):,}")
            c2.metric("Fraud Flagged",      f"{fraud_count:,}")
            c3.metric("Legitimate",         f"{len(preds)-fraud_count:,}")

            if fraud_count == 0:
                st.success("No fraud detected with this risk profile.")
            else:
                st.warning(f"{fraud_count} suspicious transactions flagged at **{risk_choice.split('  ')[1]}** threshold.")

            # Save to database
            amt_col = df_up['Amount'].values if 'Amount' in df_up.columns else [0.0] * len(preds)
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            for pred_val, prob, amt in zip(preds, probas, amt_col):
                c.execute('INSERT INTO predictions (prediction, probability, amount) VALUES (?,?,?)',
                          (int(pred_val), float(prob), float(amt)))
            conn.commit()
            conn.close()
            get_db_predictions.clear()
            st.success(f"Saved {len(preds):,} predictions to database. Dashboard and History are now updated.")

            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv_out, "fraud_predictions.csv", "text/csv")

# PAGE: DASHBOARD 
elif "Dashboard" in page:
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    preds = get_db_predictions()
    if preds.empty:
        st.warning("No prediction data yet.")
    else:
        total  = len(preds)
        frauds = preds['prediction'].sum()
        legits = total - frauds
        avg_prob  = preds['probability'].mean() * 100
        fraud_pct = frauds / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", f"{total:,}")
        c2.metric("Fraud",  f"{frauds:,}",  f"{fraud_pct:.1f}%",       delta_color="inverse")
        c3.metric("Legit",  f"{legits:,}",  f"{100-fraud_pct:.1f}%")
        c4.metric("Avg Fraud Prob",         f"{avg_prob:.1f}%")

        st.markdown("---")
        r1l, r1r = st.columns(2)
        with r1l:
            fig_pie = px.pie(values=[legits, frauds], names=['Legitimate', 'Fraud'],
                             color_discrete_sequence=['#2ecc71', '#e74c3c'],
                             title="Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        with r1r:
            fig_hist = px.histogram(preds, x='probability', nbins=30, color='prediction',
                                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                                    title="Fraud Probability Distribution",
                                    labels={'probability': 'Probability', 'prediction': 'Class'})
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        r2l, r2r = st.columns(2)
        with r2l:
            pm = preds.copy()
            pm['day'] = pm['timestamp'].dt.to_period('D').astype(str)
            monthly   = pm.groupby(['day', 'prediction']).size().reset_index(name='count')
            monthly['label'] = monthly['prediction'].map({0: 'Legit', 1: 'Fraud'})
            fig_trend = px.bar(monthly, x='day', y='count', color='label',
                               color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                               title="Daily Prediction Trend", barmode='stack')
            fig_trend.update_xaxes(tickangle=45)
            st.plotly_chart(fig_trend, use_container_width=True)
        with r2r:
            preds_amt = preds.dropna(subset=['amount'])
            fig_amt   = px.scatter(preds_amt, x='amount', y='probability', color='prediction',
                                   color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                                   title="Amount vs Fraud Probability",
                                   labels={'amount': 'Amount ($)', 'probability': 'Fraud Prob'})
            st.plotly_chart(fig_amt, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Performance Summary")
        fig_bar = px.bar(metrics_df, x='Model', y=['F1', 'Recall', 'Precision', 'ROC_AUC'],
                         barmode='group', title="Model Comparison",
                         color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        st.plotly_chart(fig_bar, use_container_width=True)

#  PAGE: HISTORY 
elif "History" in page:
    st.title("📋 Prediction History")
    st.markdown("---")

    preds = get_db_predictions()
    if preds.empty:
        st.warning("No prediction records found.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_class = st.selectbox("Filter by class", ["All", "Fraud Only", "Legit Only"])
            min_prob     = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)

        df_show = preds.copy()
        if filter_class == "Fraud Only":  df_show = df_show[df_show['prediction'] == 1]
        elif filter_class == "Legit Only": df_show = df_show[df_show['prediction'] == 0]
        df_show = df_show[df_show['probability'] >= min_prob]

        df_show['Label']       = df_show['prediction'].map({0: 'Legitimate', 1: 'FRAUD'})
        df_show['Probability'] = (df_show['probability'] * 100).round(2).astype(str) + '%'
        df_show['Amount']      = df_show['amount'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else 'N/A')

        st.dataframe(
            df_show[['id', 'Label', 'Probability', 'Amount', 'timestamp']].rename(
                columns={'id': 'ID', 'timestamp': 'Timestamp'}),
            use_container_width=True
        )
        st.markdown(f"Showing **{len(df_show):,}** of **{len(preds):,}** records")
        csv_export = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Export History CSV", csv_export, "prediction_history.csv", "text/csv")

#  PAGE: RETRAINING 
elif "Retraining" in page:
    st.title("🔄 Model Retraining")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Model")
        best_row = metrics_df.loc[metrics_df['F1'].idxmax()]
        m1, m2 = st.columns(2)
        m1.metric("Model Name", best_model_name)
        m2.metric("F1 Score",   f"{best_row['F1']:.2f}%")
        m3, m4 = st.columns(2)
        m3.metric("ROC AUC",    f"{best_row['ROC_AUC']:.2f}%")
        m4.metric("Recall",     f"{best_row['Recall']:.2f}%")
        m5, m6 = st.columns(2)
        m5.metric("Precision",  f"{best_row['Precision']:.2f}%")
        m6.metric("Accuracy",   f"{best_row['Accuracy']:.2f}%")

    with col2:
        st.subheader("Retraining Workflow")
        st.markdown("""
```
Any CSV file uploaded
       ↓
Auto-detect fraud column
       ↓
Feature Engineering + SMOTE
       ↓
Train Random Forest
       ↓
Evaluate F1 + AUC
       ↓
Replace fraud_model.pkl
       ↓
Metrics updated live
```
        """)

    st.markdown("---")
    st.subheader("Upload New Training Data")
    st.info("Upload any CSV file. The system auto-detects the fraud column. No exact column name required.")
    new_data = st.file_uploader("Upload CSV file", type=['csv'])

    if new_data:
        df_new = pd.read_csv(new_data)
        st.write(f"Uploaded: {df_new.shape[0]:,} rows x {df_new.shape[1]} columns")

        # Show all columns found
        st.markdown("**Columns found in your file:**")
        st.write(df_new.columns.tolist())

        # Auto-detect fraud column
        fraud_col = detect_fraud_column(df_new)

        if fraud_col:
            df_new = df_new.rename(columns={fraud_col: 'Class'})
            df_new['Class'] = df_new['Class'].astype(int)
            fraud_cnt = int(df_new['Class'].sum())
            legit_cnt = len(df_new) - fraud_cnt

            st.success(f"Fraud column detected: **'{fraud_col}'** → mapped to Class automatically.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows",     f"{len(df_new):,}")
            c2.metric("Fraud Samples",  f"{fraud_cnt:,}")
            c3.metric("Legit Samples",  f"{legit_cnt:,}")
            c4.metric("Fraud Rate",     f"{fraud_cnt/len(df_new)*100:.3f}%")

            if fraud_cnt < 10:
                st.warning("Very few fraud samples found. Retraining may not improve the model.")

            if st.button("Start Retraining", type="primary", use_container_width=True):
                with st.spinner("Retraining model... this may take 1-2 minutes."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
                        from sklearn.preprocessing import StandardScaler

                        # Backup original model if not already backed up
                        orig_path = os.path.join(MODEL_DIR, 'fraud_model_original.pkl')
                        if not os.path.exists(orig_path):
                            shutil.copy(os.path.join(MODEL_DIR, 'fraud_model.pkl'), orig_path)
                            shutil.copy(os.path.join(MODEL_DIR, 'model_metrics.csv'),
                                        os.path.join(MODEL_DIR, 'model_metrics_original.csv'))
                            shutil.copy(os.path.join(MODEL_DIR, 'best_model_name.txt'),
                                        os.path.join(MODEL_DIR, 'best_model_name_original.txt'))

                        # Feature engineering
                        num_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
                        num_cols = [c for c in num_cols if c != 'Class']
                        if len(num_cols) == 0:
                            st.error("No numeric feature columns found for training.")
                            st.stop()

                        X = df_new[num_cols].fillna(0)
                        y = df_new['Class']

                        # Scale if Amount/Time present
                        new_scaler = StandardScaler()
                        X_scaled   = pd.DataFrame(new_scaler.fit_transform(X), columns=X.columns)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

                        # Manual oversampling (no external dependency)
                        if y_train.sum() >= 5:
                            minority = X_train[y_train == 1]
                            min_labels = y_train[y_train == 1]
                            majority_count = (y_train == 0).sum()
                            minority_count = (y_train == 1).sum()
                            repeats = max(1, majority_count // minority_count)
                            X_train = pd.concat([X_train] + [minority] * repeats, ignore_index=True)
                            y_train = pd.concat([y_train] + [min_labels] * repeats, ignore_index=True)

                        # Train
                        new_model = RandomForestClassifier(
                            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
                        new_model.fit(X_train, y_train)

                        # Evaluate
                        yp    = new_model.predict(X_test)
                        yprob = new_model.predict_proba(X_test)[:, 1]
                        new_metrics = {
                            'Model':     'Random Forest (Retrained)',
                            'Accuracy':  round(accuracy_score(y_test, yp) * 100, 2),
                            'Precision': round(precision_score(y_test, yp, zero_division=0) * 100, 2),
                            'Recall':    round(recall_score(y_test, yp, zero_division=0) * 100, 2),
                            'F1':        round(f1_score(y_test, yp, zero_division=0) * 100, 2),
                            'ROC_AUC':   round(roc_auc_score(y_test, yprob) * 100, 2),
                        }

                        # Save new model and metrics
                        joblib.dump(new_model, os.path.join(MODEL_DIR, 'fraud_model.pkl'))
                        joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'feature_names.pkl'))
                        new_metrics_df = pd.DataFrame([new_metrics])
                        new_metrics_df.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
                        with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'w') as f:
                            f.write('Random Forest (Retrained)')

                        load_model.clear()
                        model, scaler, feature_names, metrics_df, best_model_name = load_model()

                        st.success("Retraining complete! Model updated successfully.")
                        st.markdown("### New Model Performance")
                        r1, r2, r3, r4, r5 = st.columns(5)
                        r1.metric("F1 Score",   f"{new_metrics['F1']:.2f}%")
                        r2.metric("ROC AUC",    f"{new_metrics['ROC_AUC']:.2f}%")
                        r3.metric("Recall",     f"{new_metrics['Recall']:.2f}%")
                        r4.metric("Precision",  f"{new_metrics['Precision']:.2f}%")
                        r5.metric("Accuracy",   f"{new_metrics['Accuracy']:.2f}%")
                        st.info("Use the Reset button in the sidebar to restore the original model anytime.")

                    except Exception as e:
                        st.error(f"Retraining failed: {e}")

        else:
            st.warning("No fraud/label column detected automatically in this file.")
            st.markdown("**Your file has these columns:**")
            col_df = pd.DataFrame({'Column Name': df_new.columns.tolist(),
                                   'Data Type':   [str(df_new[c].dtype) for c in df_new.columns],
                                   'Sample Value':[str(df_new[c].iloc[0]) if len(df_new) > 0 else 'N/A'
                                                   for c in df_new.columns]})
            st.dataframe(col_df, use_container_width=True)
            st.info("For retraining, your CSV needs a column with values 0 (Legit) and 1 (Fraud). "
                    "Common names: Class, is_fraud, fraud, label, target. "
                    "For prediction on new data without labels, use the Fraud Detection page instead.")

    st.markdown("---")
    st.subheader("Retraining Triggers")
    st.markdown("""
| Trigger | Threshold | Action |
|---|---|---|
| Recall drop | Below 75% | Immediate retrain |
| Data drift | Feature shift >10% | Scheduled retrain |
| Volume | 10,000 new labeled records | Weekly retrain |
| Performance decay | F1 drops 5+ points | Alert + retrain |
    """)
