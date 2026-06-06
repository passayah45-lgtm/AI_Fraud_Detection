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
    # Add analyst_status column if it does not exist yet
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN analyst_status TEXT DEFAULT 'Pending'")
        conn.commit()
    except:
        pass
    c.execute("INSERT INTO predictions (prediction, probability, amount, analyst_status) VALUES (?,?,?,'Pending')",
              (int(prediction), float(probability), float(amount)))
    conn.commit()
    conn.close()

def update_analyst_status(record_id, status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE predictions SET analyst_status = ? WHERE id = ?", (status, record_id))
    conn.commit()
    conn.close()

def reset_to_original():
    # Reset database to first 300 seeded records
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE id > 300')
    conn.commit()
    conn.close()
    # Restore original model files if backup exists
    if os.path.exists(ORIG_MODEL):
        shutil.copy(ORIG_MODEL, os.path.join(MODEL_DIR, 'fraud_model.pkl'))
        orig_metrics = pd.read_csv(os.path.join(MODEL_DIR, 'model_metrics_original.csv'))
        orig_metrics.to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
        with open(os.path.join(MODEL_DIR, 'best_model_name_original.txt')) as f:
            orig_name = f.read().strip()
        with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'w') as f:
            f.write(orig_name)
    # Clear both cache and session_state so app reloads fresh model
    load_model.clear()
    for key in ['model','scaler','feature_names','metrics_df','best_model_name']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.force_reload = True

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
    _model, _scaler, _feats, _metrics, _best_name = load_model()
    if 'model' not in st.session_state or st.session_state.get('force_reload'):
        st.session_state.model          = _model
        st.session_state.scaler         = _scaler
        st.session_state.feature_names  = _feats
        st.session_state.metrics_df     = _metrics
        st.session_state.best_model_name= _best_name
        st.session_state.force_reload   = False
    model          = st.session_state.model
    scaler         = st.session_state.scaler
    feature_names  = st.session_state.feature_names
    metrics_df     = st.session_state.metrics_df
    best_model_name= st.session_state.best_model_name
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

    # Always fetch fresh data  reflects latest uploads and predictions
    get_db_predictions.clear()
    preds  = get_db_predictions()
    total  = len(preds)
    frauds = int(preds['prediction'].sum())
    legits = total - frauds
    fraud_pct = frauds / max(total, 1) * 100
    best_row  = metrics_df.loc[metrics_df['F1'].idxmax()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions",  f"{total:,}")
    col2.metric("Fraud Detected",     f"{frauds:,}",
                delta=f"{fraud_pct:.1f}%", delta_color="inverse")
    col3.metric("Legitimate",         f"{legits:,}")
    col4.metric("Model AUC",          f"{best_row['ROC_AUC']:.2f}%")

    st.markdown("---")

    # About section
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

    # About the author
    st.subheader("About the Developer")
    st.markdown("""
**Mister T** is a data analyst and full-stack developer currently pursuing a Master of Science
in Data Analysis at KIIT University, Bhubaneswar, India. This project was built as part of a
summer internship program to demonstrate end-to-end machine learning deployment in a real-world
financial fraud context.

The system covers the full data science lifecycle: raw data ingestion, exploratory analysis,
feature engineering, model training with class imbalance handling, evaluation, deployment,
and live monitoring. The application was built entirely in Python using open-source tools
and is hosted publicly on Streamlit Community Cloud.

**Areas of expertise:** Data Analytics, Python, Machine Learning, Web Development, SQL.
    """)

    st.markdown("---")

    # Model comparison - reflects current model_metrics.csv
    st.subheader("Model Comparison")
    st.caption("Updates automatically after retraining.")
    metrics_display = metrics_df[['Model','Accuracy','Precision','Recall','F1','ROC_AUC']].copy()
    metrics_display.columns = ['Model','Accuracy %','Precision %','Recall %','F1 %','ROC AUC %']
    st.dataframe(
        metrics_display.style.highlight_max(axis=0, color='#d4edda'),
        use_container_width=True
    )

    st.markdown("---")

    # Live EDA - built from current prediction database
    st.subheader("Exploratory Data Analysis")
    st.caption("Charts reflect all predictions currently in the database.")

    if total == 0:
        st.info("No predictions yet. Make some predictions to see charts.")
    else:
        eda1, eda2 = st.columns(2)

        with eda1:
            # Class distribution
            fig_class = px.bar(
                x=['Legitimate', 'Fraud'], y=[legits, frauds],
                color=['Legitimate', 'Fraud'],
                color_discrete_map={'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'},
                title=f"Class Distribution — {total:,} total predictions",
                labels={'x': 'Class', 'y': 'Count'}
            )
            fig_class.update_layout(showlegend=False)
            st.plotly_chart(fig_class, use_container_width=True)

        with eda2:
            # Amount distribution by class
            preds_amt = preds.dropna(subset=['amount'])
            if len(preds_amt) > 0:
                fig_amt = px.histogram(
                    preds_amt, x='amount', color='prediction',
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                    title="Transaction Amount Distribution",
                    labels={'amount': 'Amount ($)', 'prediction': 'Class'},
                    nbins=40, barmode='overlay', opacity=0.7
                )
                fig_amt.update_layout(legend_title="0=Legit  1=Fraud")
                st.plotly_chart(fig_amt, use_container_width=True)

        eda3, eda4 = st.columns(2)

        with eda3:
            # Fraud probability distribution
            fig_prob = px.histogram(
                preds, x='probability', nbins=30,
                color='prediction',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                title="Fraud Probability Distribution",
                labels={'probability': 'Probability', 'prediction': 'Class'},
                barmode='overlay', opacity=0.7
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with eda4:
            # Daily trend
            pm = preds.copy()
            pm['day'] = pm['timestamp'].dt.to_period('D').astype(str)
            daily = pm.groupby(['day', 'prediction']).size().reset_index(name='count')
            daily['label'] = daily['prediction'].map({0: 'Legit', 1: 'Fraud'})
            fig_trend = px.bar(
                daily, x='day', y='count', color='label',
                color_discrete_map={'Legit': '#2ecc71', 'Fraud': '#e74c3c'},
                title="Daily Prediction Trend", barmode='stack'
            )
            fig_trend.update_xaxes(tickangle=45)
            st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    # Live Model Evaluation - ROC and Precision-Recall from DB
    st.subheader("Model Evaluation")
    st.caption("Computed from current predictions in the database.")

    if total < 10 or frauds == 0:
        st.info("Not enough labeled predictions yet to render evaluation charts. Make predictions first.")
    else:
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

            y_true  = preds['prediction'].values
            y_score = preds['probability'].values

            ev1, ev2, ev3 = st.columns(3)

            with ev1:
                # Confusion matrix from 0.5 threshold
                y_pred = (y_score >= 0.5).astype(int)
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                tn = int(((y_pred == 0) & (y_true == 0)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                cm_df = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=['Actual Legit', 'Actual Fraud'],
                    columns=['Predicted Legit', 'Predicted Fraud']
                )
                fig_cm = px.imshow(
                    cm_df, text_auto=True, color_continuous_scale='Blues',
                    title="Confusion Matrix (threshold 0.5)"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with ev2:
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_val = roc_auc_score(y_true, y_score)
                fig_roc = px.line(
                    x=fpr, y=tpr,
                    title=f"ROC Curve (AUC = {auc_val:.4f})",
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                )
                fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                                   line=dict(dash='dash', color='gray'))
                fig_roc.update_traces(line_color='#e74c3c', line_width=2)
                st.plotly_chart(fig_roc, use_container_width=True)

            with ev3:
                # Precision-Recall Curve
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                fig_pr = px.line(
                    x=rec, y=prec,
                    title=f"Precision-Recall (AP = {ap:.4f})",
                    labels={'x': 'Recall', 'y': 'Precision'}
                )
                fig_pr.update_traces(line_color='#3498db', line_width=2)
                st.plotly_chart(fig_pr, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not render evaluation charts: {e}")

    st.markdown("---")

    # Chatbot assistant
    st.subheader("App Assistant")
    st.caption("Ask anything in any language: English, French, Arabic, Spanish and more.")

    SYSTEM_PROMPT = (
        "You are a multilingual assistant for an AI-Powered Financial Fraud Detection System "
        "built by Issa TOURE, a data analyst at KIIT University, India.\n\n"
        "LANGUAGE RULE: Always respond in the same language the user writes in. "
        "French input gets French response. Arabic gets Arabic. Never switch unless the user does.\n\n"
        "APP PAGES:\n"
        "- Home: live KPIs, system description, tech stack, model comparison, EDA charts, evaluation charts, chatbot, about developer.\n"
        "- Fraud Detection: choose Risk Profile (Low/Medium/High), enter transaction manually or upload CSV.\n"
        "- Dashboard: live Plotly charts - distribution, daily trends, amount vs probability.\n"
        "- History: editable table, mark records as Pending / Confirmed Fraud / False Positive. Export CSV.\n"
        "- Retraining: upload any CSV, auto-detects fraud column, retrains Random Forest, updates metrics.\n\n"
        "MODEL: Random Forest, 284,807 transactions, F1=68.78%, AUC=97.95%, Recall=80%, Precision=60.32%.\n"
        "RISK PROFILES: Low(0.3)=more alerts, Medium(0.5)=balanced, High(0.7)=fewer alerts.\n"
        "RESET BUTTON: in sidebar, restores original model and 300 seeded predictions.\n"
        "Answer clearly and help users understand the app and fraud predictions."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "voice_transcript" not in st.session_state:
        st.session_state.voice_transcript = ""
    if "send_voice" not in st.session_state:
        st.session_state.send_voice = False

    import streamlit.components.v1 as components

    voice_html = (
        '<div style="margin-bottom:8px;display:flex;align-items:center;gap:12px;">'
        '<button id="micBtn" onclick="startVoice()" style="'
        'background:#e74c3c;color:white;border:none;border-radius:8px;'
        'padding:8px 18px;font-size:14px;cursor:pointer;font-weight:500;">'
        '&#127908; Speak</button>'
        '<span id="statusTxt" style="font-size:13px;color:#888;">'
        'Click to speak in any language</span>'
        '</div>'
        '<script>'
        'function startVoice(){'
        'var btn=document.getElementById("micBtn");'
        'var st=document.getElementById("statusTxt");'
        'if(!("webkitSpeechRecognition" in window)&&!("SpeechRecognition" in window)){'
        'st.textContent="Voice not supported. Use Chrome or Edge.";'
        'st.style.color="#e74c3c";return;}'
        'var SR=window.SpeechRecognition||window.webkitSpeechRecognition;'
        'var rec=new SR();rec.continuous=false;rec.interimResults=false;rec.lang="";'
        'btn.textContent="Listening...";btn.style.background="#c0392b";'
        'st.textContent="Speak now...";st.style.color="#e74c3c";'
        'rec.start();'
        'rec.onresult=function(e){'
        'var t=e.results[0][0].transcript;'
        'btn.textContent="&#127908; Speak";btn.style.background="#e74c3c";'
        'st.textContent="Heard: "+t;st.style.color="#2ecc71";'
        'window.parent.postMessage({type:"streamlit:setComponentValue",value:t},"*");};'
        'rec.onerror=function(e){'
        'btn.textContent="&#127908; Speak";btn.style.background="#e74c3c";'
        'st.textContent="Error: "+e.error+". Try again.";st.style.color="#e74c3c";};'
        'rec.onend=function(){'
        'if(btn.textContent==="Listening..."){'
        'btn.textContent="&#127908; Speak";btn.style.background="#e74c3c";}};}'
        '</script>'
    )

    voice_result = components.html(voice_html, height=55)

    # Store voice transcript when received from browser
    if voice_result and isinstance(voice_result, str) and voice_result.strip():
        st.session_state.voice_transcript = voice_result.strip()
        st.session_state.send_voice = False

    # Show captured voice with Send button
    if st.session_state.voice_transcript and not st.session_state.send_voice:
        st.info(f"Voice captured: **{st.session_state.voice_transcript}**")
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            pass
        with col_v2:
            if st.button("Send Voice", type="primary", use_container_width=True):
                st.session_state.send_voice = True
                st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Determine input source
    user_input = st.chat_input("Ask me anything in any language...")

    # Use voice if Send Voice was clicked
    if st.session_state.send_voice and st.session_state.voice_transcript:
        final_input = st.session_state.voice_transcript
        st.session_state.voice_transcript = ""
        st.session_state.send_voice = False
    elif user_input:
        final_input = user_input
    else:
        final_input = None

    if final_input:
        st.session_state.chat_history.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                    messages = [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history]
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=512,
                        system=SYSTEM_PROMPT,
                        messages=messages
                    )
                    reply = response.content[0].text
                except Exception as e:
                    reply = f"Assistant unavailable: {e}. Please check ANTHROPIC_API_KEY in Streamlit secrets."
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        if st.button("Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.voice_transcript = ""
            st.session_state.send_voice = False
            st.rerun()

    st.markdown("---")

    # About the developer - very bottom
    st.subheader("About the Developer")
    st.markdown(
        "**Issa TOURE** is a data analyst and full-stack developer currently pursuing a Master of Science "
        "in Computer Science at KIIT University, Bhubaneswar, India. This project was built as part of a "
        "summer internship program to demonstrate end-to-end machine learning deployment in a real-world "
        "financial fraud context.\n\n"
        "The system covers the full data science lifecycle: raw data ingestion, exploratory analysis, "
        "feature engineering, model training with class imbalance handling, evaluation, deployment, "
        "and live monitoring. The application was built entirely in Python using open-source tools "
        "and is hosted publicly on Streamlit Community Cloud.\n\n"
        "**Areas of expertise:** Data Analytics, Python, Machine Learning, Web Development, SQL."
    )

# PAGE: FRAUD DETECTION
elif "Detection" in page:
    st.title("🔍 Fraud Detection")
    st.markdown("### Predict whether a transaction is fraudulent")
    st.markdown("---")

    # Risk profile selector
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
            # Pass as numpy array to bypass sklearn feature name validation
            input_arr = np.array([[input_data[f] for f in feature_names]])
            proba     = model.predict_proba(input_arr)[0][1]
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
                st.error(f"### ⚠️ FRAUD DETECTED   {risk_label}")
            else:
                st.success(f"### ✅ LEGITIMATE TRANSACTION   {risk_label}")

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
        st.info(f"Using **{risk_choice.split('  ')[1]}** profile — threshold {threshold:.0%}. Missing columns filled automatically.")
        uploaded = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"Uploaded: {df_up.shape[0]:,} rows x {df_up.shape[1]} columns")
            st.write(f"Columns found: {', '.join(df_up.columns.tolist())}")

            # Build numpy array matching exact training feature order
            X_arr = np.zeros((len(df_up), len(feature_names)), dtype=np.float64)
            feat_idx = {f: i for i, f in enumerate(feature_names)}
            if 'Amount' in df_up.columns:
                scaled_amt = scaler['amount'].transform(df_up[['Amount']]).flatten()
                X_arr[:, feat_idx['scaled_amount']] = scaled_amt
            if 'Time' in df_up.columns:
                scaled_time = scaler['time'].transform(df_up[['Time']]).flatten()
                X_arr[:, feat_idx['scaled_time']] = scaled_time
            for col in df_up.columns:
                if col in feat_idx:
                    X_arr[:, feat_idx[col]] = df_up[col].fillna(0).values

            # Pass as numpy array - skips sklearn feature name validation
            probas = model.predict_proba(X_arr)[:, 1]
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

# PAGE: HISTORY
elif "History" in page:
    st.title("📋 Prediction History")
    st.markdown("---")

    preds = get_db_predictions()
    if 'analyst_status' not in preds.columns:
        preds['analyst_status'] = 'Pending'

    if preds.empty:
        st.warning("No prediction records found.")
    else:
        # Status summary
        s1, s2, s3 = st.columns(3)
        s1.metric("Pending Review",  (preds['analyst_status'] == 'Pending').sum())
        s2.metric("Confirmed Fraud", (preds['analyst_status'] == 'Confirmed Fraud').sum())
        s3.metric("False Positives", (preds['analyst_status'] == 'False Positive').sum())
        st.markdown("---")

        col1, col2 = st.columns([1, 3])
        with col1:
            filter_class  = st.selectbox("Filter by class",  ["All", "Fraud Only", "Legit Only"])
            min_prob      = st.slider("Min probability", 0.0, 1.0, 0.0, 0.01)
            filter_status = st.selectbox("Filter by status", ["All", "Pending", "Confirmed Fraud", "False Positive"])

        df_show = preds.copy()
        if filter_class == "Fraud Only":   df_show = df_show[df_show['prediction'] == 1]
        elif filter_class == "Legit Only":  df_show = df_show[df_show['prediction'] == 0]
        df_show = df_show[df_show['probability'] >= min_prob]
        if filter_status != "All":
            df_show = df_show[df_show['analyst_status'] == filter_status]

        df_show['Label']       = df_show['prediction'].map({0: 'Legitimate', 1: 'FRAUD'})
        df_show['Probability'] = (df_show['probability'] * 100).round(2).astype(str) + '%'
        df_show['Amount']      = df_show['amount'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else 'N/A')

        st.info("Change the Status column to mark predictions as Confirmed Fraud or False Positive, then click Save Reviews.")

        edit_df = df_show[['id', 'Label', 'Probability', 'Amount', 'timestamp', 'analyst_status']].rename(
            columns={'id': 'ID', 'timestamp': 'Timestamp', 'analyst_status': 'Status'})

        updated = st.data_editor(
            edit_df,
            column_config={
                "ID":          st.column_config.Column(disabled=True),
                "Label":       st.column_config.Column(disabled=True),
                "Probability": st.column_config.Column(disabled=True),
                "Amount":      st.column_config.Column(disabled=True),
                "Timestamp":   st.column_config.Column(disabled=True),
                "Status": st.column_config.SelectboxColumn(
                    "Analyst Review",
                    options=["Pending", "Confirmed Fraud", "False Positive"],
                    required=True
                )
            },
            use_container_width=True,
            hide_index=True,
            key="analyst_review_table"
        )

        if st.button("Save Reviews", type="primary", use_container_width=True):
            changes = 0
            for i, row in updated.iterrows():
                original_status = df_show.iloc[i]['analyst_status'] if i < len(df_show) else 'Pending'
                if row['Status'] != original_status:
                    update_analyst_status(int(row['ID']), row['Status'])
                    changes += 1
            get_db_predictions.clear()
            if changes > 0:
                st.success(f"{changes} review(s) saved successfully.")
            else:
                st.info("No changes detected.")
            st.rerun()

        st.markdown(f"Showing **{len(df_show):,}** of **{len(preds):,}** records")
        csv_export = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Export History CSV", csv_export, "prediction_history.csv", "text/csv")

# PAGE: RETRAINING
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
                        from sklearn.pipeline import Pipeline
                        from sklearn.compose import ColumnTransformer

                        # Backup original model if not already backed up
                        orig_path = os.path.join(MODEL_DIR, 'fraud_model_original.pkl')
                        if not os.path.exists(orig_path):
                            shutil.copy(os.path.join(MODEL_DIR, 'fraud_model.pkl'), orig_path)
                            shutil.copy(os.path.join(MODEL_DIR, 'model_metrics.csv'),
                                        os.path.join(MODEL_DIR, 'model_metrics_original.csv'))
                            shutil.copy(os.path.join(MODEL_DIR, 'best_model_name.txt'),
                                        os.path.join(MODEL_DIR, 'best_model_name_original.txt'))

                        # Select numeric features only
                        num_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
                        num_cols = [c for c in num_cols if c != 'Class']
                        if len(num_cols) == 0:
                            st.error("No numeric feature columns found for training.")
                            st.stop()

                        X = df_new[num_cols].fillna(0)
                        y = df_new['Class']

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y)

                        # Manual oversampling (no external dependency)
                        if y_train.sum() >= 5:
                            minority   = X_train[y_train == 1]
                            min_labels = y_train[y_train == 1]
                            repeats    = max(1, (y_train == 0).sum() // (y_train == 1).sum())
                            X_train = pd.concat([X_train] + [minority] * repeats, ignore_index=True)
                            y_train = pd.concat([y_train] + [min_labels] * repeats, ignore_index=True)

                        # Build Pipeline: scaler + classifier in one object
                        # This means the saved model handles its own preprocessing
                        scale_cols = [c for c in X_train.columns if c.lower() in ['amount', 'time']]
                        if scale_cols:
                            transformer = ColumnTransformer(
                                transformers=[('scaler', StandardScaler(), scale_cols)],
                                remainder='passthrough'
                            )
                            new_pipeline = Pipeline([
                                ('preprocessing', transformer),
                                ('classifier', RandomForestClassifier(
                                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
                            ])
                        else:
                            new_pipeline = Pipeline([
                                ('preprocessing', StandardScaler()),
                                ('classifier', RandomForestClassifier(
                                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
                            ])

                        new_pipeline.fit(X_train, y_train)

                        # Evaluate
                        yp    = new_pipeline.predict(X_test)
                        yprob = new_pipeline.predict_proba(X_test)[:, 1]
                        new_metrics = {
                            'Model':     'Random Forest + Pipeline (Retrained)',
                            'Accuracy':  round(accuracy_score(y_test, yp) * 100, 2),
                            'Precision': round(precision_score(y_test, yp, zero_division=0) * 100, 2),
                            'Recall':    round(recall_score(y_test, yp, zero_division=0) * 100, 2),
                            'F1':        round(f1_score(y_test, yp, zero_division=0) * 100, 2),
                            'ROC_AUC':   round(roc_auc_score(y_test, yprob) * 100, 2),
                        }

                        # Save pipeline and metadata
                        joblib.dump(new_pipeline, os.path.join(MODEL_DIR, 'fraud_model.pkl'))
                        joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'feature_names.pkl'))
                        pd.DataFrame([new_metrics]).to_csv(os.path.join(MODEL_DIR, 'model_metrics.csv'), index=False)
                        with open(os.path.join(MODEL_DIR, 'best_model_name.txt'), 'w') as f:
                            f.write('Random Forest + Pipeline (Retrained)')

                        load_model.clear()
                        for key in ['model','scaler','feature_names','metrics_df','best_model_name']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.force_reload = True

                        st.success("Retraining complete. Pipeline model saved and active.")
                        st.markdown("### New Model Performance")
                        r1, r2, r3, r4, r5 = st.columns(5)
                        r1.metric("F1 Score",  f"{new_metrics['F1']:.2f}%")
                        r2.metric("ROC AUC",   f"{new_metrics['ROC_AUC']:.2f}%")
                        r3.metric("Recall",    f"{new_metrics['Recall']:.2f}%")
                        r4.metric("Precision", f"{new_metrics['Precision']:.2f}%")
                        r5.metric("Accuracy",  f"{new_metrics['Accuracy']:.2f}%")
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
