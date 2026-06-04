const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Mister T';
pres.title = 'AI-Powered Financial Fraud Detection System';

// COLOR PALETTE: Midnight Executive + Red Threat 
const C = {
  navy:    '1E2761',
  iceBlue: 'CADCFC',
  red:     'E74C3C',
  green:   '2ECC71',
  white:   'FFFFFF',
  lightBg: 'F4F6FB',
  darkGray:'2C3E50',
  mid:     '8899CC',
  accent:  'E74C3C',
};

//SLIDE 1: TITLE 
{
  let s = pres.addSlide();
  s.background = { color: C.navy };
  // Top accent band
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.red }, line: { color: C.red } });
  // Big title text
  s.addText('AI-Powered Financial', {
    x: 0.6, y: 0.7, w: 8.8, h: 0.85, fontSize: 38, bold: true,
    color: C.white, fontFace: 'Calibri', align: 'left'
  });
  s.addText('Fraud Detection System', {
    x: 0.6, y: 1.45, w: 8.8, h: 0.85, fontSize: 38, bold: true,
    color: C.iceBlue, fontFace: 'Calibri', align: 'left'
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.42, w: 2.5, h: 0.04, fill: { color: C.red }, line: { color: C.red } });
  s.addText('Machine Learning Project | KIIT University', {
    x: 0.6, y: 2.55, w: 8.8, h: 0.4, fontSize: 14, color: C.mid, fontFace: 'Calibri', align: 'left'
  });
  s.addText('Issa TOURE  |  Master of Science in Computer Science  |  2025', {
    x: 0.6, y: 3.0, w: 8.8, h: 0.36, fontSize: 12, color: C.iceBlue, fontFace: 'Calibri', align: 'left', italic: true
  });
  // Stats boxes bottom
  const boxes = [
    { label: '284,807', sub: 'Transactions' },
    { label: '0.17%', sub: 'Fraud Rate' },
    { label: '97.95%', sub: 'ROC AUC' },
    { label: '80%', sub: 'Recall' },
  ];
  boxes.forEach((b, i) => {
    const x = 0.6 + i * 2.3;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.8, w: 2.1, h: 1.3, fill: { color: '162255' }, line: { color: C.red, pt: 1 } });
    s.addText(b.label, { x, y: 3.85, w: 2.1, h: 0.65, fontSize: 22, bold: true, color: C.white, align: 'center' });
    s.addText(b.sub,   { x, y: 4.5,  w: 2.1, h: 0.5,  fontSize: 11, color: C.iceBlue, align: 'center' });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.525, w: 10, h: 0.1, fill: { color: C.red }, line: { color: C.red } });
}

// SLIDE 2: PROBLEM STATEMENT 
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('The Problem', { x: 0.35, y: 0.25, w: 9.3, h: 0.6, fontSize: 28, bold: true, color: C.navy });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.88, w: 9.3, h: 0.04, fill: { color: C.red }, line: { color: C.red } });

  const cards = [
    { top: '$32B', body: 'Lost to credit card fraud globally each year', col: C.red },
    { top: '0.17%', body: 'Extreme class imbalance: fraud is rare but costly', col: C.navy },
    { top: '48 hrs', body: 'Dataset covers 284K transactions in 48 hours', col: '028090' },
  ];
  cards.forEach((c, i) => {
    const x = 0.4 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.1, w: 2.9, h: 2.9,
      fill: { color: C.white },
      shadow: { type: 'outer', blur: 8, offset: 3, angle: 135, color: '000000', opacity: 0.10 },
      line: { color: 'DDDDDD', pt: 0.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.1, w: 2.9, h: 0.1, fill: { color: c.col }, line: { color: c.col } });
    s.addText(c.top,  { x, y: 1.3,  w: 2.9, h: 0.85, fontSize: 30, bold: true, color: c.col, align: 'center' });
    s.addText(c.body, { x: x+0.15, y: 2.2, w: 2.6, h: 1.65, fontSize: 13, color: C.darkGray, align: 'center', valign: 'middle' });
  });

  s.addText('Goal: Build a system that catches fraud without overwhelming legitimate customers.', {
    x: 0.35, y: 5.1, w: 9.3, h: 0.4, fontSize: 12, color: C.navy, italic: true
  });
}

//SLIDE 3: DATASET 
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.05, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Dataset Overview', { x: 0.5, y: 0.22, w: 9, h: 0.6, fontSize: 26, bold: true, color: C.white, align: 'left' });

  // Left: description
  const points = [
    'European cardholders, September 2013',
    '284,807 total transactions over 48 hours',
    '492 fraudulent (0.172%) vs 284,315 legitimate',
    '30 features: Time, V1-V28 (PCA), Amount',
    'V-features are PCA-transformed (confidential)',
    '1,081 duplicate rows removed during cleaning',
  ];
  points.forEach((p, i) => {
    s.addText([{ text: '●  ', options: { color: C.red, bold: true } }, { text: p }], {
      x: 0.5, y: 1.25 + i * 0.52, w: 5.0, h: 0.45, fontSize: 13, color: C.darkGray
    });
  });

  // Right: visual bar chart (native)
  s.addChart(pres.charts.BAR, [{
    name: 'Count',
    labels: ['Legitimate', 'Fraud'],
    values: [284315, 492]
  }], {
    x: 5.8, y: 1.15, w: 3.9, h: 4.0,
    barDir: 'col',
    chartColors: [C.navy, C.red],
    chartArea: { fill: { color: 'F8FAFF' }, roundedCorners: false },
    catAxisLabelColor: '64748B',
    valAxisLabelColor: '64748B',
    valGridLine: { color: 'E2E8F0', size: 0.5 },
    catGridLine: { style: 'none' },
    showValue: true,
    dataLabelColor: '1E293B',
    showLegend: false,
    showTitle: true,
    title: 'Class Distribution',
    titleColor: C.navy,
    titleFontSize: 13,
  });
}

//  SLIDE 4: METHODOLOGY 
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Methodology Pipeline', { x: 0.35, y: 0.2, w: 9.3, h: 0.6, fontSize: 26, bold: true, color: C.navy });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 9.3, h: 0.04, fill: { color: C.red }, line: { color: C.red } });

  const steps = [
    { num: '01', title: 'Data Cleaning',         desc: 'Remove 1,081 duplicates. Check nulls. Validate schema.' },
    { num: '02', title: 'Feature Engineering',   desc: 'StandardScaler on Amount + Time. Drop originals.' },
    { num: '03', title: 'SMOTE Oversampling',     desc: 'Balance 0.17% fraud to 50% on training set only.' },
    { num: '04', title: '80/20 Stratified Split', desc: 'Preserve fraud ratio in both train and test sets.' },
    { num: '05', title: 'Train 3 Models',         desc: 'Logistic Regression, Random Forest, XGBoost.' },
    { num: '06', title: 'Evaluate + Select',      desc: 'Compare F1, AUC, Recall. Save best model.' },
  ];

  steps.forEach((step, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.35 + col * 3.15;
    const y = 1.0 + row * 2.0;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.0, h: 1.75,
      fill: { color: C.white },
      shadow: { type: 'outer', blur: 6, offset: 2, angle: 135, color: '000000', opacity: 0.09 },
      line: { color: 'DDDDDD', pt: 0.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.55, h: 0.55, fill: { color: C.navy }, line: { color: C.navy } });
    s.addText(step.num, { x, y, w: 0.55, h: 0.55, fontSize: 14, bold: true, color: C.white, align: 'center', valign: 'middle' });
    s.addText(step.title, { x: x+0.6, y: y+0.05, w: 2.3, h: 0.45, fontSize: 12.5, bold: true, color: C.navy });
    s.addText(step.desc,  { x: x+0.1, y: y+0.55, w: 2.8, h: 1.1, fontSize: 11, color: C.darkGray, valign: 'top' });
  });
}

// SLIDE 5: MODEL COMPARISON 
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.05, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Model Comparison', { x: 0.5, y: 0.22, w: 9, h: 0.6, fontSize: 26, bold: true, color: C.white });

  // Table
  const headers = [['Model', 'Accuracy %', 'Precision %', 'Recall %', 'F1 %', 'ROC AUC %']];
  const rows = [
    ['Logistic Regression', '97.35', '5.27', '87.37', '9.95', '96.31'],
    ['XGBoost',             '99.41', '19.95', '84.21', '32.26', '97.26'],
    ['Random Forest ✓',     '99.88', '60.32', '80.00', '68.78', '97.95'],
  ];
  const tableData = [
    headers[0].map(h => ({ text: h, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 12 } })),
    ...rows.map((r, ri) => r.map(c => ({
      text: c,
      options: {
        fill: { color: ri === 2 ? 'd4edda' : (ri % 2 === 0 ? 'F8FAFF' : C.white) },
        bold: ri === 2,
        color: ri === 2 ? '155724' : C.darkGray,
        fontSize: 12
      }
    })))
  ];
  s.addTable(tableData, {
    x: 0.5, y: 1.2, w: 9, h: 2.5,
    border: { pt: 0.5, color: 'DDDDDD' },
    colW: [2.8, 1.4, 1.4, 1.4, 1.2, 1.8],
  });

  s.addText('Random Forest selected as best model (highest F1 Score at 68.78%)', {
    x: 0.5, y: 3.85, w: 9, h: 0.45, fontSize: 13, bold: true, color: C.navy
  });

  s.addText([
    { text: 'Why F1 matters: ', options: { bold: true } },
    { text: 'In fraud detection, false negatives (missed fraud) are expensive. F1 balances precision and recall better than accuracy on imbalanced data.' }
  ], { x: 0.5, y: 4.35, w: 9, h: 0.7, fontSize: 12, color: C.darkGray });

  // Bar chart F1 comparison
  s.addChart(pres.charts.BAR, [{
    name: 'F1 Score',
    labels: ['Logistic\nRegression', 'XGBoost', 'Random\nForest'],
    values: [9.95, 32.26, 68.78]
  }], {
    x: 6.2, y: 1.15, w: 3.5, h: 2.4,
    barDir: 'col',
    chartColors: ['95A5A6', '3498DB', '2ECC71'],
    chartArea: { fill: { color: 'F8FAFF' } },
    catAxisLabelColor: '64748B',
    valAxisLabelColor: '64748B',
    valGridLine: { color: 'E2E8F0', size: 0.5 },
    catGridLine: { style: 'none' },
    showValue: true,
    dataLabelColor: '1E293B',
    showLegend: false,
    showTitle: true, title: 'F1 Score (%)',
    titleFontSize: 12, titleColor: C.navy,
  });
}

//  SLIDE 6: RESULTS 
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.red }, line: { color: C.red } });
  s.addText('Final Model Results', { x: 0.35, y: 0.2, w: 9.3, h: 0.6, fontSize: 26, bold: true, color: C.navy });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 9.3, h: 0.04, fill: { color: C.red }, line: { color: C.red } });
  s.addText('Random Forest Classifier (SMOTE-balanced)', {
    x: 0.35, y: 0.9, w: 9.3, h: 0.38, fontSize: 13, color: C.darkGray, italic: true
  });

  const kpis = [
    { val: '97.95%', lab: 'ROC AUC',   col: C.navy },
    { val: '68.78%', lab: 'F1 Score',  col: C.red },
    { val: '80.00%', lab: 'Recall',    col: '028090' },
    { val: '60.32%', lab: 'Precision', col: '8E44AD' },
  ];
  kpis.forEach((k, i) => {
    const x = 0.4 + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.4, w: 2.2, h: 1.7,
      fill: { color: C.white },
      shadow: { type: 'outer', blur: 8, offset: 3, angle: 135, color: '000000', opacity: 0.10 },
      line: { color: 'DDDDDD', pt: 0.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.4, w: 2.2, h: 0.1, fill: { color: k.col }, line: { color: k.col } });
    s.addText(k.val, { x, y: 1.6,  w: 2.2, h: 0.75, fontSize: 26, bold: true, color: k.col, align: 'center' });
    s.addText(k.lab, { x, y: 2.38, w: 2.2, h: 0.5,  fontSize: 12, color: C.darkGray, align: 'center' });
  });

  // Confusion matrix text representation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.28, w: 9.2, h: 2.1,
    fill: { color: C.white }, line: { color: 'DDDDDD', pt: 0.5 }
  });
  s.addText('Confusion Matrix Results', {
    x: 0.6, y: 3.35, w: 9, h: 0.4, fontSize: 13, bold: true, color: C.navy
  });
  const cmData = [
    [{ text: '', options: { bold: true } }, { text: 'Predicted Legit', options: { bold: true, color: C.white, fill: { color: C.navy } } }, { text: 'Predicted Fraud', options: { bold: true, color: C.white, fill: { color: C.red } } }],
    [{ text: 'Actual Legit',  options: { bold: true } }, { text: '56,580 (TN)', options: { fill: { color: 'd4edda' } } }, { text: '241 (FP)', options: { fill: { color: 'fff3cd' } } }],
    [{ text: 'Actual Fraud',  options: { bold: true } }, { text: '19 (FN)',    options: { fill: { color: 'fff3cd' } } }, { text: '76 (TP)', options: { fill: { color: 'd4edda' } } }],
  ];
  s.addTable(cmData, {
    x: 0.6, y: 3.75, w: 9, h: 1.5,
    border: { pt: 0.5, color: 'DDDDDD' }, fontSize: 12,
    colW: [2.5, 3.25, 3.25],
  });
}

//  SLIDE 7: APPLICATION 
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.05, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Streamlit Application', { x: 0.5, y: 0.22, w: 9, h: 0.6, fontSize: 26, bold: true, color: C.white });

  const pages = [
    { tag: 'HOME',      tagCol: C.navy,   name: 'Home',            desc: 'Project overview, model metrics, EDA charts, evaluation plots' },
    { tag: 'DETECT',    tagCol: C.red,    name: 'Fraud Detection',  desc: 'Manual form + CSV batch upload with confidence gauge' },
    { tag: 'DASH',      tagCol: '028090', name: 'Dashboard',        desc: 'Plotly charts: pie, histogram, daily trends, scatter' },
    { tag: 'HISTORY',   tagCol: '8E44AD', name: 'History',          desc: 'SQLite prediction log with filter, sort, CSV export' },
    { tag: 'RETRAIN',   tagCol: 'F39C12', name: 'Retraining',       desc: 'Workflow documentation + new data upload interface' },
  ];
  pages.forEach((p, i) => {
    const x = 0.5;
    const y = 1.2 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 9, h: 0.72,
      fill: { color: i % 2 === 0 ? 'F4F6FB' : C.white },
      line: { color: 'DDDDDD', pt: 0.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x: x+0.1, y: y+0.16, w: 0.85, h: 0.4, fill: { color: p.tagCol }, line: { color: p.tagCol } });
    s.addText(p.tag, { x: x+0.1, y: y+0.16, w: 0.85, h: 0.4, fontSize: 7.5, bold: true, color: C.white, align: 'center', valign: 'middle' });
    s.addText(p.name, { x: x+1.1, y: y+0.05, w: 2.0, h: 0.35, fontSize: 13, bold: true, color: C.navy });
    s.addText(p.desc, { x: x+1.1, y: y+0.35, w: 7.8, h: 0.35, fontSize: 11, color: C.darkGray });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.4, w: 10, h: 0.225, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Tech: Streamlit  |  Plotly  |  SQLite  |  Joblib', {
    x: 0, y: 5.4, w: 10, h: 0.225, fontSize: 11, color: C.iceBlue, align: 'center'
  });
}

//  SLIDE 8: DEPLOYMENT 
{
  let s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Deployment & Architecture', { x: 0.35, y: 0.2, w: 9.3, h: 0.6, fontSize: 26, bold: true, color: C.navy });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 9.3, h: 0.04, fill: { color: C.red }, line: { color: C.red } });

  // Architecture flow
  const nodes = ['creditcard.csv', 'Data Cleaning', 'SMOTE + Split', 'Model Training', 'fraud_model.pkl', 'Streamlit App', 'Streamlit Cloud'];
  const nodeColors = [C.navy, '028090', '8E44AD', C.red, 'F39C12', '2ECC71', C.navy];
  nodes.forEach((n, i) => {
    const x = 0.35 + i * 1.34;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.1, w: 1.25, h: 0.65,
      fill: { color: nodeColors[i] }, line: { color: nodeColors[i] }, rectRadius: 0.08
    });
    s.addText(n, { x, y: 1.1, w: 1.25, h: 0.65, fontSize: 9.5, bold: true, color: C.white, align: 'center', valign: 'middle' });
    if (i < nodes.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + 1.25, y: 1.425, w: 0.09, h: 0,
        line: { color: C.darkGray, width: 1.5 }
      });
    }
  });

  // File structure
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 2.05, w: 4.4, h: 3.2,
    fill: { color: C.white }, line: { color: 'DDDDDD', pt: 0.5 }
  });
  s.addText('Project Structure', { x: 0.55, y: 2.1, w: 4.0, h: 0.38, fontSize: 13, bold: true, color: C.navy });
  const files = [
    'app.py             — Streamlit application',
    'train_models2.py  — ML pipeline',
    'requirements.txt  — Dependencies',
    'models/           — fraud_model.pkl, scaler.pkl',
    'database/         — fraud.db (SQLite)',
    'reports/          — EDA + evaluation charts',
    'data/             — cleaned_creditcard.csv',
  ];
  files.forEach((f, i) => {
    s.addText(f, { x: 0.55, y: 2.55 + i * 0.36, w: 4.1, h: 0.34, fontSize: 10.5, color: C.darkGray, fontFace: 'Consolas' });
  });

  // Deployment steps
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.0, y: 2.05, w: 4.65, h: 3.2,
    fill: { color: C.white }, line: { color: 'DDDDDD', pt: 0.5 }
  });
  s.addText('Deploy to Streamlit Cloud', { x: 5.2, y: 2.1, w: 4.2, h: 0.38, fontSize: 13, bold: true, color: C.navy });
  const dsteps = [
    '1. Push project to GitHub repository',
    '2. Visit share.streamlit.io',
    '3. Connect GitHub account',
    '4. Select repo + set main: app.py',
    '5. Click Deploy',
    '6. Access at: https://your-app.streamlit.app',
  ];
  dsteps.forEach((d, i) => {
    s.addText(d, { x: 5.2, y: 2.55 + i * 0.38, w: 4.3, h: 0.35, fontSize: 11, color: C.darkGray });
  });
}

// SLIDE 9: MONITORING 
{
  let s = pres.addSlide();
  s.background = { color: C.white };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.05, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText('Monitoring & Retraining', { x: 0.5, y: 0.22, w: 9, h: 0.6, fontSize: 26, bold: true, color: C.white });

  // Left: Dashboard metrics
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.15, w: 4.4, h: 4.1,
    fill: { color: C.lightBg }, line: { color: 'DDDDDD', pt: 0.5 }
  });
  s.addText('Live Dashboard Metrics', { x: 0.7, y: 1.22, w: 4.0, h: 0.42, fontSize: 13, bold: true, color: C.navy });
  const metrics = [
    '✅  Total predictions count',
    '🔴  Fraud predictions + %',
    '🟢  Legitimate predictions + %',
    '📈  Daily prediction trend',
    '💰  Amount vs fraud probability',
    '📊  Model comparison bar chart',
  ];
  metrics.forEach((m, i) => {
    s.addText(m, { x: 0.7, y: 1.72 + i * 0.54, w: 4.0, h: 0.46, fontSize: 12, color: C.darkGray });
  });

  // Right: Retraining trigger table
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.15, w: 4.4, h: 4.1,
    fill: { color: C.lightBg }, line: { color: 'DDDDDD', pt: 0.5 }
  });
  s.addText('Retraining Triggers', { x: 5.4, y: 1.22, w: 4.0, h: 0.42, fontSize: 13, bold: true, color: C.navy });
  const triggers = [
    ['Recall < 75%',          'Immediate retrain'],
    ['Feature drift > 10%',   'Scheduled retrain'],
    ['10,000 new records',    'Weekly retrain'],
    ['F1 drops 5+ points',    'Alert + retrain'],
  ];
  triggers.forEach((t, i) => {
    const y = 1.72 + i * 0.68;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.4, y, w: 3.9, h: 0.6, fill: { color: C.white }, line: { color: 'DDDDDD', pt: 0.5 } });
    s.addText(t[0], { x: 5.55, y: y+0.04, w: 2.0, h: 0.28, fontSize: 11, bold: true, color: C.red });
    s.addText(t[1], { x: 5.55, y: y+0.3,  w: 3.5, h: 0.25, fontSize: 11, color: C.darkGray });
  });
  s.addText('New Data → Clean → SMOTE → Train → Compare → Deploy', {
    x: 5.4, y: 4.5, w: 3.9, h: 0.6, fontSize: 11, color: C.navy, italic: true, align: 'center'
  });
}

// ─── SLIDE 10: CONCLUSION 
{
  let s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.red }, line: { color: C.red } });

  s.addText('Key Takeaways', { x: 0.6, y: 0.3, w: 8.8, h: 0.65, fontSize: 30, bold: true, color: C.white });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.96, w: 3, h: 0.04, fill: { color: C.red }, line: { color: C.red } });

  const takeaways = [
    { icon: '🎯', text: 'SMOTE is critical for imbalanced fraud detection. Accuracy is misleading.' },
    { icon: '🏆', text: 'Random Forest delivers best F1 (68.78%) and AUC (97.95%) on this dataset.' },
    { icon: '⚡', text: 'Streamlit enables rapid deployment of ML models as web applications.' },
    { icon: '📦', text: 'SQLite provides audit trails essential for production financial systems.' },
    { icon: '🔄', text: 'A defined retraining strategy keeps the model current as fraud patterns evolve.' },
  ];
  takeaways.forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.15 + i * 0.76, w: 8.8, h: 0.68,
      fill: { color: '162255' }, line: { color: '2a3a7a', pt: 0.5 }
    });
    s.addText(t.icon, { x: 0.65, y: 1.15 + i * 0.76, w: 0.6, h: 0.68, fontSize: 20, align: 'center', valign: 'middle' });
    s.addText(t.text, { x: 1.35, y: 1.15 + i * 0.76, w: 7.9, h: 0.68, fontSize: 13, color: C.iceBlue, valign: 'middle' });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.42, w: 10, h: 0.2, fill: { color: C.red }, line: { color: C.red } });
  s.addText('Mister T  |  KIIT University  |  Master of Science in Data Analysis', {
    x: 0, y: 5.42, w: 10, h: 0.2, fontSize: 10, color: C.white, align: 'center', valign: 'middle'
  });
}

pres.writeFile({ fileName: '/home/claude/AI_Fraud_Detection/reports/presentation.pptx' })
  .then(() => console.log('Presentation saved'))
  .catch(e => { console.error(e); process.exit(1); });
