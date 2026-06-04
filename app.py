import streamlit as st
import pandas as pd
import numpy as np
import joblib, sqlite3, os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

#  CONFIG 
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH   = os.path.join(BASE_DIR, 'database', 'fraud.db')
RPT_DIR   = os.path.join(BASE_DIR, 'reports')

#LOAD RESOURCES 
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
        st.metric("F1 Score",  f"{best_row['F1']:.2f}%")
        st.metric("ROC AUC",   f"{best_row['ROC_AUC']:.2f}%")
        st.metric("Recall",    f"{best_row['Recall']:.2f}%")

#  PAGE: HOME 
if "Home" in page:
    st.title("🔒 AI-Powered Financial Fraud Detection System")
    st.markdown("### Protecting transactions with machine learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    preds = get_db_predictions()
    total = len(preds)
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
- **Supports retraining** when new data is available
- **Dataset**: 284,807 European credit card transactions (2013)
- **Fraud rate**: 0.17% (highly imbalanced, handled with SMOTE)
        """)

    with col_b:
        st.subheader("Technology Stack")
        tech = {
            "Machine Learning": "Scikit-learn, XGBoost, SMOTE",
            "Data Processing": "Pandas, NumPy",
            "Visualization": "Plotly, Seaborn, Matplotlib",
            "Application": "Streamlit",
            "Database": "SQLite",
            "Deployment": "Streamlit Community Cloud",
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

#PAGE: FRAUD DETECTION 
elif "Detection" in page:
    st.title("🔍 Fraud Detection")
    st.markdown("### Predict whether a transaction is fraudulent")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

    with tab1:
        st.markdown("#### Enter Transaction Features")
        st.info("V1-V28 are PCA-transformed features from the original dataset. Enter Amount and Time manually.")

        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
            time_s = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)
        with col2:
            threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                                  help="Lower = catch more fraud (more false positives). Higher = more precise.")

        st.markdown("#### V-Features (PCA Components)")
        v_cols = [c for c in feature_names if c.startswith('V')]
        v_defaults = {c: 0.0 for c in v_cols}
        # Typical fraud signature for demo
        fraud_signature = {'V14': -10.0, 'V4': 4.5, 'V11': -3.5, 'V12': -5.0, 'V10': -4.0}

        use_demo = st.checkbox("Load a fraud demo signature", value=False)
        rows = [v_cols[i:i+7] for i in range(0, len(v_cols), 7)]
        v_vals = {}
        for row in rows:
            cols_r = st.columns(len(row))
            for i, feat in enumerate(row):
                default = fraud_signature.get(feat, 0.0) if use_demo else 0.0
                v_vals[feat] = cols_r[i].number_input(feat, value=default, format="%.4f", key=feat)

        if st.button("Predict Transaction", type="primary", use_container_width=True):
            # Build input vector
            sc_amount = scaler['amount'].transform([[amount]])[0][0]
            sc_time   = scaler['time'].transform([[time_s]])[0][0]
            input_data = {}
            for f in feature_names:
                if f == 'scaled_amount': input_data[f] = sc_amount
                elif f == 'scaled_time': input_data[f] = sc_time
                else:                    input_data[f] = v_vals.get(f, 0.0)
            input_df = pd.DataFrame([input_data])[feature_names]
            proba  = model.predict_proba(input_df)[0][1]
            pred   = 1 if proba >= threshold else 0

            save_prediction(pred, proba, amount)
            get_db_predictions.clear()

            st.markdown("---")
            if pred == 1:
                st.error(f"### FRAUD DETECTED")
                st.metric("Fraud Probability", f"{proba*100:.2f}%")
            else:
                st.success(f"### LEGITIMATE TRANSACTION")
                st.metric("Fraud Probability", f"{proba*100:.2f}%")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range':[0,100]},
                    'bar': {'color': '#e74c3c' if pred==1 else '#2ecc71'},
                    'steps': [
                        {'range':[0,30], 'color':'#d4edda'},
                        {'range':[30,60],'color':'#fff3cd'},
                        {'range':[60,100],'color':'#f8d7da'},
                    ],
                    'threshold': {'line':{'color':'black','width':4}, 'thickness':0.75, 'value':threshold*100}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Upload any CSV file to run fraud prediction")
        st.info("You can upload any CSV file. Missing columns will be filled automatically.")
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"Uploaded: {df_up.shape[0]:,} rows x {df_up.shape[1]} columns")
            st.write(f"Columns found: {', '.join(df_up.columns.tolist())}")

            # Build input using matching columns, fill missing ones with 0
            X_up = pd.DataFrame(0.0, index=range(len(df_up)), columns=feature_names)
            if 'Amount' in df_up.columns:
                X_up['scaled_amount'] = scaler['amount'].transform(df_up[['Amount']])
            if 'Time' in df_up.columns:
                X_up['scaled_time'] = scaler['time'].transform(df_up[['Time']])
            for col in feature_names:
                if col in df_up.columns:
                    X_up[col] = df_up[col].fillna(0).values

            probas = model.predict_proba(X_up)[:,1]
            preds  = (probas >= 0.5).astype(int)
            df_up['Fraud_Probability_%'] = np.round(probas*100, 2)
            df_up['Prediction']          = preds
            df_up['Label']               = df_up['Prediction'].map({0:'Legitimate', 1:'FRAUD'})

            st.dataframe(df_up[['Fraud_Probability_%', 'Label']].head(50), use_container_width=True)
            fraud_count = preds.sum()
            st.markdown(f"**Fraud detected:** {fraud_count:,} / {len(preds):,} transactions")
            if fraud_count == 0:
                st.success("No fraud detected in this file.")
            else:
                st.warning(f"{fraud_count} suspicious transactions found. Download results for details.")
            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv_out, "fraud_predictions.csv", "text/csv")

#  PAGE: DASHBOARD 
elif "Dashboard" in page:
    st.title("📊 Analytics Dashboard")
    st.markdown("---")

    preds = get_db_predictions()
    if preds.empty:
        st.warning("No prediction data yet. Make some predictions first.")
    else:
        total  = len(preds)
        frauds = preds['prediction'].sum()
        legits = total - frauds
        avg_prob = preds['probability'].mean()*100
        fraud_pct = frauds/total*100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Predictions", f"{total:,}")
        c2.metric("Fraud",  f"{frauds:,}",  f"{fraud_pct:.1f}%", delta_color="inverse")
        c3.metric("Legit",  f"{legits:,}",  f"{100-fraud_pct:.1f}%")
        c4.metric("Avg Fraud Prob", f"{avg_prob:.1f}%")

        st.markdown("---")
        row1_l, row1_r = st.columns(2)

        with row1_l:
            fig_pie = px.pie(values=[legits, frauds], names=['Legitimate','Fraud'],
                             color_discrete_sequence=['#2ecc71','#e74c3c'],
                             title="Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with row1_r:
            fig_hist = px.histogram(preds, x='probability', nbins=30, color='prediction',
                                    color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                    title="Fraud Probability Distribution",
                                    labels={'probability':'Probability','prediction':'Class'})
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        row2_l, row2_r = st.columns(2)

        with row2_l:
            preds_monthly = preds.copy()
            preds_monthly['month'] = preds_monthly['timestamp'].dt.to_period('D').astype(str)
            monthly = preds_monthly.groupby(['month','prediction']).size().reset_index(name='count')
            monthly['label'] = monthly['prediction'].map({0:'Legit',1:'Fraud'})
            fig_trend = px.bar(monthly, x='month', y='count', color='label',
                               color_discrete_map={'Legit':'#2ecc71','Fraud':'#e74c3c'},
                               title="Daily Prediction Trend", barmode='stack')
            fig_trend.update_xaxes(tickangle=45)
            st.plotly_chart(fig_trend, use_container_width=True)

        with row2_r:
            preds_amt = preds.dropna(subset=['amount'])
            fig_amt = px.scatter(preds_amt, x='amount', y='probability',
                                 color='prediction',
                                 color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                 title="Amount vs Fraud Probability",
                                 labels={'amount':'Amount ($)','probability':'Fraud Prob'})
            st.plotly_chart(fig_amt, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Performance Summary")
        fig_bar = px.bar(metrics_df, x='Model', y=['F1','Recall','Precision','ROC_AUC'],
                         barmode='group', title="Model Comparison",
                         color_discrete_sequence=['#3498db','#e74c3c','#2ecc71','#f39c12'])
        st.plotly_chart(fig_bar, use_container_width=True)

# PAGE: HISTORY 
elif "History" in page:
    st.title("📋 Prediction History")
    st.markdown("---")

    preds = get_db_predictions()
    if preds.empty:
        st.warning("No prediction records found.")
    else:
        col1, col2 = st.columns([1,3])
        with col1:
            filter_class = st.selectbox("Filter by class", ["All","Fraud Only","Legit Only"])
            min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)

        df_show = preds.copy()
        if filter_class == "Fraud Only": df_show = df_show[df_show['prediction']==1]
        elif filter_class == "Legit Only": df_show = df_show[df_show['prediction']==0]
        df_show = df_show[df_show['probability'] >= min_prob]

        df_show['Label']       = df_show['prediction'].map({0:'Legitimate',1:'FRAUD'})
        df_show['Probability'] = (df_show['probability']*100).round(2).astype(str) + '%'
        df_show['Amount']      = df_show['amount'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else 'N/A')

        st.dataframe(
            df_show[['id','Label','Probability','Amount','timestamp']].rename(
                columns={'id':'ID','timestamp':'Timestamp'}),
            use_container_width=True
        )
        st.markdown(f"Showing **{len(df_show):,}** of **{len(preds):,}** records")

        csv_export = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Export History CSV", csv_export, "prediction_history.csv", "text/csv")

#PAGE: RETRAINING 
elif "Retraining" in page:
    st.title("🔄 Model Retraining")
    st.markdown("---")
    st.info("This page outlines the retraining workflow. Automated retraining will be triggered when new labeled data is available.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Model")
        best_row = metrics_df.loc[metrics_df['F1'].idxmax()]
        st.write(f"**Name:** {best_model_name}")
        st.write(f"**F1 Score:** {best_row['F1']:.2f}%")
        st.write(f"**ROC AUC:** {best_row['ROC_AUC']:.2f}%")
        st.write(f"**Recall:** {best_row['Recall']:.2f}%")
        st.write(f"**Precision:** {best_row['Precision']:.2f}%")

    with col2:
        st.subheader("Retraining Workflow")
        st.markdown("""
```
New Labeled Data
       ↓
Data Cleaning + Feature Engineering
       ↓
SMOTE (handle imbalance)
       ↓
Train/Test Split (80/20)
       ↓
Train RF + XGBoost + LR
       ↓
Compare F1 + AUC
       ↓
Save Best Model
       ↓
Replace fraud_model.pkl
```
        """)

    st.markdown("---")
    st.subheader("Upload New Training Data")
    st.info("Upload any CSV file. The system will inspect it and show you what was found.")
    new_data = st.file_uploader("Upload CSV file", type=['csv'])
    if new_data:
        df_new = pd.read_csv(new_data)
        st.write(f"Uploaded: {df_new.shape[0]:,} rows x {df_new.shape[1]} columns")
        st.write(f"Columns found: {', '.join(df_new.columns.tolist())}")

        if 'Class' in df_new.columns:
            fraud_cnt = int(df_new['Class'].sum())
            legit_cnt = len(df_new) - fraud_cnt
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(df_new):,}")
            col2.metric("Fraud Samples", f"{fraud_cnt:,}")
            col3.metric("Legit Samples", f"{legit_cnt:,}")
            st.success(f"Class column detected. Fraud rate: {fraud_cnt/len(df_new)*100:.3f}%")
            st.button("Start Retraining (Future Feature)", disabled=True)
            st.caption("Full automated retraining pipeline coming in v2.0")
        else:
            st.warning("No Class column found in this file.")
            st.write("The system inspected your file and found these columns:")
            st.write(df_new.columns.tolist())
            st.info("For retraining, a Class column (0=Legit, 1=Fraud) is needed. For fraud prediction on new data, use the Fraud Detection page instead.")

    st.markdown("---")
    st.subheader("Retraining Triggers")
    st.markdown("""
| Trigger | Threshold | Action |
|---|---|---|
| New fraud patterns detected | Recall drops below 75% | Immediate retrain |
| Data drift | Feature distribution shifts > 10% | Scheduled retrain |
| Volume-based | 10,000 new labeled records | Weekly retrain |
| Performance decay | F1 drops 5+ points | Alert + retrain |
    """)
