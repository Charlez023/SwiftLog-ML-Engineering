"""Regenerate SwiftLog Week 5 documentation and notebook from saved model outputs."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'Models'
DATA = ROOT / 'Data'
PROCESSED = DATA / 'Processed_Datasets'
REPORTS = ROOT / 'Reports'
PRESENTATION = ROOT / 'Presentation'
NOTEBOOK = ROOT / 'Notebook'
for d in [REPORTS, PRESENTATION, NOTEBOOK]: d.mkdir(parents=True, exist_ok=True)
comparison = pd.read_excel(MODELS / 'Model_Comparison.xlsx')
quality_path = PROCESSED / 'data_quality_summary.csv'
quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
bundles = joblib.load(MODELS / 'Best_Model.pkl')

# Correctly encoded, task-specific Streamlit app text.
app_path = ROOT / 'App' / 'app.py'
if app_path.exists():
    app = app_path.read_text(encoding='utf-8')
    app = app.replace('page_icon="Ã°Å¸Å¡Åš"', 'page_icon="📦"')
    app = app.replace('Predicted daily product demand (units)', 'Predicted weekly product demand (units)')
    app = app.replace('st.title("SwiftLog Logistics: ML Decision Support")', 'st.title("SwiftLog Logistics: ML Decision Support")')
    app = app.replace('st.write("Held-out test metrics:", b["metrics"])', '''st.write("Held-out test metrics:", b["metrics"])
if b["kind"] == "classification" and b["metrics"].get("ROC_AUC", 1) < 0.60:
    st.warning("The ROC-AUC indicates weak separation between classes. Treat this output as an experimental signal, not a reliable operational decision.")
if b["kind"] == "regression" and b["metrics"].get("R2", 0) < 0:
    st.warning("The held-out R-squared is below zero, so this model performed worse than a simple test-set mean baseline. Use the estimate for demonstration only.")''')
    app_path.write_text(app, encoding='utf-8')

# Model comparison in a readable, stable order.
problem_order = [
    'Delivery Delay Classification', 'Product Return Classification',
    'Product Demand Forecasting', 'Customer Value Prediction (Next Month)'
]
comparison['Problem'] = pd.Categorical(comparison['Problem'], categories=problem_order, ordered=True)
comparison = comparison.sort_values(['Problem', 'Selected', 'Algorithm'], ascending=[True, False, True])
comparison['Problem'] = comparison['Problem'].astype(str)
comparison.to_excel(MODELS / 'Model_Comparison.xlsx', index=False)

# Notebook: readable analysis and audit trail of the saved modelling workflow.
try:
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# SwiftLog Week 5: Machine Learning Engineering

**Mission:** Intelligent Logistics Prediction & Supply Chain Optimization.

This notebook reviews source data, engineered features, model comparisons, evaluation artefacts, and limitations. The full model-training pipeline is `../run_swiftlog_project.py`. Run that script from the project root to regenerate the saved models and outputs before rerunning this notebook.

The supplied data is used as provided for this educational prototype. Results are reported as measured; weak performance is not hidden or described as production-ready."""))
    cells.append(nbf.v4.new_markdown_cell("## 1. Imports and paths"))
    cells.append(nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nROOT = Path('..').resolve()\nDATA = ROOT / 'Data'\nPROCESSED = DATA / 'Processed_Datasets'\nMODELS = ROOT / 'Models'\n"))
    cells.append(nbf.v4.new_markdown_cell("## 2. Source data and quality checks\nThe source tables cover customers, orders, order items, products, shipments, delivery performance, returns, inventory, warehouses, suppliers, and employees. The original file-level data dictionary is brief; calculated model features are documented in `Feature_Data_Dictionary.csv`."))
    cells.append(nbf.v4.new_code_cell("quality = pd.read_csv(PROCESSED / 'data_quality_summary.csv')\ndisplay(quality)"))
    cells.append(nbf.v4.new_markdown_cell("## 3. Inspect the original target distributions"))
    cells.append(nbf.v4.new_code_cell("shipments = pd.read_csv(DATA / 'Original_Datasets' / 'SwiftLog_Shipments.csv')\nreturns = pd.read_csv(DATA / 'Original_Datasets' / 'SwiftLog_Returns.csv')\norders = pd.read_csv(DATA / 'Original_Datasets' / 'SwiftLog_Orders.csv')\nprint('Shipments:', len(shipments))\nprint('Delayed shipments (DelayDays > 0):', int((pd.to_numeric(shipments['DelayDays'], errors='coerce') > 0).sum()))\nprint('Orders associated with recorded returns:', returns['OrderID'].nunique())\nprint('Orders:', orders['OrderID'].nunique())"))
    cells.append(nbf.v4.new_markdown_cell("## 4. Engineered feature samples\nThe delivery-delay target is `DelayDays > 0`. The return target is whether an order appears in the Returns table. Demand features use weekly product quantities and lagged values from prior weeks. Customer value is framed as next-month revenue, using the preceding three months of activity. For the demand and customer-value tasks, evaluation is chronological rather than a random row split."))
    for title, fn in [
        ('Delivery delay features','delivery_delay_features.csv'),
        ('Product return features','product_return_features.csv'),
        ('Weekly demand forecasting features','demand_forecasting_features.csv'),
        ('Customer next-month value features','customer_value_features.csv')]:
        cells.append(nbf.v4.new_markdown_cell('### ' + title))
        cells.append(nbf.v4.new_code_cell(f"features = pd.read_csv(PROCESSED / '{fn}')\nprint('Shape:', features.shape)\ndisplay(features.head())"))
    cells.append(nbf.v4.new_markdown_cell("## 5. Compare the candidate algorithms\nThe table contains three algorithms per task. `Selected` marks the model stored in `Best_Model.pkl`; `Tuned` indicates that a tuning search was accepted. The selected model metrics are updated in the comparison table to match the saved model."))
    cells.append(nbf.v4.new_code_cell("comparison = pd.read_excel(MODELS / 'Model_Comparison.xlsx')\ndisplay(comparison)"))
    cells.append(nbf.v4.new_markdown_cell("## 6. Review selected models and held-out metrics"))
    cells.append(nbf.v4.new_code_cell("selected = comparison[comparison['Selected'] == True]\ndisplay(selected)"))
    cells.append(nbf.v4.new_markdown_cell("## 7. Evaluation visuals"))
    cells.append(nbf.v4.new_code_cell("from IPython.display import Image, display\nfor name in ['delivery_delay_classification_confusion_matrix.png', 'product_return_classification_confusion_matrix.png']:\n    path = MODELS / name\n    if path.exists():\n        print(name)\n        display(Image(filename=str(path)))\nimportance = MODELS / 'Feature_Importance.png'\nif importance.exists():\n    display(Image(filename=str(importance)))"))
    cells.append(nbf.v4.new_markdown_cell("""## 8. Interpretation and limitations

- Delivery delay and return classification ROC-AUC values near 0.5 indicate weak class discrimination, even when F1 or accuracy appears high. This is especially important for the imbalanced delay target.
- Negative held-out R-squared for demand or next-month customer value means the selected model performed worse than a simple mean-prediction baseline on the test period.
- The dataset spans a limited period and has limited operational context. Forecast quality may improve with richer timestamps, planned delivery dates, inventory snapshots, and more history.
- Model outputs should support human review only. They are not validated for live logistics decisions.

## 9. Reproduce the outputs

From the project root, run `python run_swiftlog_project.py`. The script preprocesses the source data, builds the features, compares three algorithms for each task, evaluates candidates, saves model bundles and visualizations, and regenerates reports and the presentation. Then rerun this notebook to inspect the refreshed outputs."""))
    nb.cells = cells
    nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.x'}}
    nbf.write(nb, NOTEBOOK / 'Week5_MachineLearning.ipynb')
except Exception as e:
    print('Notebook update warning:', e)

# PDF reports: technical report and executive summary are intentionally different.
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='SmallSwift', parent=styles['BodyText'], fontSize=8, leading=10))
styles.add(ParagraphStyle(name='TinySwift', parent=styles['BodyText'], fontSize=7, leading=8))
def para(text, style='BodyText'):
    import xml.sax.saxutils as sax
    return Paragraph(sax.escape(str(text)), styles[style])
def table_for(frame, columns, widths=None):
    rows = [[para(c, 'TinySwift') for c in columns]]
    for _, row in frame.iterrows():
        vals=[]
        for c in columns:
            val=row.get(c, '')
            if pd.isna(val): val=''
            elif isinstance(val, (float, np.floating)): val=f'{val:.3f}'
            vals.append(para(val, 'TinySwift'))
        rows.append(vals)
    t=Table(rows, repeatRows=1, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17324D')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#B7C2CC')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F4F8')]),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

selected = comparison[comparison['Selected'] == True].copy()

def make_technical_report():
    path=REPORTS/'Technical_Report.pdf'
    story=[para('SwiftLog Technical Report','Title'), para('Machine Learning Track | Mission ML05: Intelligent Logistics Prediction & Supply Chain Optimization','Heading2'), Spacer(1,8)]
    sections=[
        ('1. Purpose', 'This report documents the machine-learning workflow for four logistics tasks using the supplied SwiftLog educational datasets. The workflow includes data quality review, feature engineering, three candidate algorithms per task, held-out evaluation, limited hyperparameter tuning, permutation importance, saved model bundles, and a Streamlit prototype.'),
        ('2. Data preparation', 'Source tables were loaded from Data/Original_Datasets, text fields were standardized, exact duplicate rows were removed from processed copies, and row counts, duplicate counts, and missing cells were recorded. Related order, item, product, customer, shipment, warehouse, and return tables were joined where required.'),
        ('3. Targets and feature design', 'Delivery delay: positive when DelayDays > 0; DelayDays and post-event delivery status are excluded from predictors. Product return: positive when an OrderID appears in Returns; order timing, city, and order-item aggregates are used. Demand forecasting: weekly product quantity is the target; lagged prior-week demand, a four-week lag, and a rolling mean of the four preceding weeks are predictors. Customer value: next-month revenue is the target; features summarize customer activity in the preceding three months.'),
        ('4. Evaluation design', 'Classification tasks use a stratified random 80/20 split and report accuracy, precision, recall, F1, and ROC-AUC. The demand and customer-value tasks use chronological train/test splits so later periods are held out. Regression reports MAE, RMSE, and R-squared. Tuning is limited and results should be treated as experimental.'),
        ('5. Model comparison', 'The following table reports all candidate algorithms. Selected marks the saved model; the selected row is updated to match the saved model metrics.')]
    for h,txt in sections: story += [para(h,'Heading2'),para(txt),Spacer(1,6)]
    cols=['Problem','Algorithm','Selected','Tuned','F1','ROC_AUC','MAE','RMSE','R2','Accuracy']
    story.append(table_for(comparison, cols))
    story += [Spacer(1,8),para('6. Interpretation','Heading2'),
      para('The delivery-delay target is imbalanced, and its ROC-AUC is close to 0.5. Its high F1 should therefore not be interpreted as strong ability to distinguish delayed from on-time shipments. The return classifier also has weak discrimination. The demand and next-month customer-value models have negative held-out R-squared, indicating worse performance than a simple mean baseline on their test periods. These results are reported transparently rather than presented as successful forecasts.'),
      para('7. Feature importance and artefacts','Heading2'),
      para('Permutation importance is saved in Models/Permutation_Importance.csv and visualized in Models/Feature_Importance.png. Classification confusion matrices are saved under Models/. The selected pipelines and their feature definitions are stored in Models/Best_Model.pkl. The Streamlit app loads this bundle for interactive demonstrations.'),
      para('8. Assumptions and limitations','Heading2'),
      para('The delay target is defined from DelayDays because the supplied status field may disagree with the numeric delay field. A recorded return is treated as the return label. Supplier reliability is not modelled because the available tables do not directly link supplier IDs to shipments. Customer value is defined as next-month item revenue rather than lifetime value. The data covers a limited historical period and does not establish production-grade predictive performance.'),
      para('9. Recommendations','Heading2'),
      para('Improve data capture for planned and actual delivery times; reconcile delivery status with delay days; capture supplier IDs on purchase and shipment records; maintain time-stamped inventory snapshots; define return labels consistently; collect more historical periods; evaluate models against simple baselines; and pilot any predictions with human review before operational use.'),
      para('10. Reproduction','Heading2'),
      para('Install dependencies from App/requirements.txt or the README instructions, then run python run_swiftlog_project.py from the project root. The script regenerates processed datasets, model comparisons, saved model bundles, evaluation figures, reports, and the presentation. The notebook reviews the outputs.')]
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=1.3*cm,leftMargin=1.3*cm,topMargin=1.3*cm,bottomMargin=1.3*cm).build(story)

def make_exec_summary():
    path=REPORTS/'Executive_Summary.pdf'
    story=[para('SwiftLog Executive Summary','Title'),para('Machine Learning Track | Week 5','Heading2'),Spacer(1,10),
      para('Objective','Heading2'),para('Build an educational machine-learning prototype to explore delivery delays, product returns, weekly product demand, and next-month customer revenue using the supplied SwiftLog datasets.'),
      para('What was completed','Heading2'),para('Reviewed and cleaned source tables; engineered task-specific features; compared three algorithms for each of four tasks; evaluated held-out predictions; saved selected model pipelines; generated comparison tables and evaluation visuals; and prepared a Streamlit demonstration app.'),
      para('Results at a glance','Heading2')]
    cols=['Problem','Selected algorithm','F1','ROC-AUC','MAE','RMSE','R2']
    summary=selected.copy().rename(columns={'Algorithm':'Selected algorithm'})
    if 'Selected algorithm' in summary: summary['Selected algorithm']=summary['Selected algorithm'].astype(str)
    story.append(table_for(summary,cols))
    story += [Spacer(1,8),para('What the results mean','Heading2'),
      para('The current models are not ready for operational use. The delay classifier has high F1 but ROC-AUC close to 0.5, suggesting weak class separation. The return classifier also shows weak discrimination. Demand and next-month customer-value regression have negative held-out R-squared, so they do not outperform a simple mean baseline on the test period.'),
      para('Recommended next steps','Heading2'),
      para('Collect richer historical data, verify target definitions, add operationally available predictors, compare against simple baselines, and validate on later time periods. Use the app only as a demonstration and keep human review in the decision process.'),
      Spacer(1,8),para('Conclusion','Heading2'),para('The project delivers a complete prototype workflow and a documented evaluation. The most important outcome is identifying the data and validation improvements needed before any business deployment.')]
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.5*cm).build(story)
make_technical_report(); make_exec_summary()

# Executive presentation with current results and clean Unicode text.
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
def add_slide(title, body, note=''):
    sl=prs.slides.add_slide(prs.slide_layouts[6])
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb=RGBColor(245,248,252)
    shape=sl.shapes.add_shape(1,0,0,prs.slide_width,Inches(1.15)); shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor(23,50,77); shape.line.fill.background()
    tf=shape.text_frame; tf.text=title; tf.paragraphs[0].font.size=Pt(28); tf.paragraphs[0].font.bold=True; tf.paragraphs[0].font.color.rgb=RGBColor(255,255,255)
    tx=sl.shapes.add_textbox(Inches(.8), Inches(1.55), Inches(11.8), Inches(4.9)).text_frame; tx.word_wrap=True; tx.text=body; tx.paragraphs[0].font.size=Pt(22); tx.paragraphs[0].font.color.rgb=RGBColor(35,48,62)
    if note:
        nf=sl.shapes.add_textbox(Inches(.8), Inches(6.5), Inches(11.8), Inches(.4)).text_frame; nf.text=note; nf.paragraphs[0].font.size=Pt(11); nf.paragraphs[0].font.color.rgb=RGBColor(75,88,100)
    ft=sl.shapes.add_textbox(Inches(.8), Inches(7.05), Inches(11.5), Inches(.25)).text_frame; ft.text=f'SwiftLog ML05 | Week 5 | {len(prs.slides):02d}/15'; ft.paragraphs[0].font.size=Pt(10)

selected_map={str(r['Problem']):r for _,r in selected.iterrows()}
def metric_line(task):
    r=selected_map.get(task)
    if r is None: return 'Metrics not available.'
    if 'F1' in r and pd.notna(r.get('F1')): return f"Selected: {r['Algorithm']} | F1 {r['F1']:.3f} | ROC-AUC {r['ROC_AUC']:.3f}"
    return f"Selected: {r['Algorithm']} | MAE {r['MAE']:.2f} | RMSE {r['RMSE']:.2f} | R-squared {r['R2']:.3f}"
slides=[
('SwiftLog Logistics','Machine Learning for Logistics Prediction and Supply Chain Optimization','Educational prototype; not a production deployment.'),
('Business challenge','Explore whether available logistics data can support earlier review of delivery delays, product returns, product demand, and customer revenue.'),
('Project objectives','Prepare the data, engineer useful features, compare algorithms, evaluate results, interpret limitations, and demonstrate predictions in a simple app.'),
('Data used','Customers, Orders, Order Items, Products, Shipments, Delivery Performance, Returns, Warehouses, Inventory, Employees, and Suppliers.'),
('Data preparation','Reviewed row counts, missing cells, and duplicates; standardized text; joined related tables; and saved cleaned and task-specific feature datasets.'),
('Task 1: delivery delay','Target: DelayDays greater than zero. Predictors include warehouse, location, customer city, and order-date parts. DelayDays itself is excluded from the inputs.'),
('Task 2: product returns','Target: order appears in the Returns table. Predictors include order timing, city, and item-level aggregates available from the order.'),
('Task 3: weekly demand','Target: product units per week. Features use prior-week demand, demand four weeks earlier, and a rolling average of the four previous weeks.'),
('Task 4: customer value','Target: customer revenue in the next month. Features summarize the preceding three months of purchasing activity.'),
('Algorithms and evaluation','Compared Logistic Regression, Decision Tree, and Random Forest for classification; Linear Regression, Decision Tree Regressor, and Random Forest Regressor for regression.'),
('Delivery delay result',metric_line('Delivery Delay Classification'),'ROC-AUC near 0.5 means weak class discrimination despite the high F1 score.'),
('Return prediction result',metric_line('Product Return Classification'),'ROC-AUC near 0.5 means the current features do not reliably separate returned and non-returned orders.'),
('Regression results',metric_line('Product Demand Forecasting')+'\n'+metric_line('Customer Value Prediction (Next Month)'),'Negative R-squared means performance is worse than a simple mean baseline on the held-out period.'),
('Business recommendations','Improve delivery timestamps, reconcile status and delay fields, link suppliers to shipments, collect inventory snapshots, and define consistent return labels.'),
('Conclusion and next steps','The workflow and demo app are complete as a prototype. Collect more history, improve predictors, validate on future periods, and require human review before operational use.')]
for item in slides:
    if len(item) == 2: add_slide(item[0], item[1])
    else: add_slide(item[0], item[1], item[2])
prs.save(PRESENTATION/'Executive_Presentation.pptx')

# README updated to accurately describe the changed targets and limitations.
readme='''# SwiftLog Logistics Analytics and AI Transformation

**Machine Learning Track | Mission ML05: Intelligent Logistics Prediction and Supply Chain Optimization**

## Project overview
This educational prototype explores four machine-learning tasks using the supplied SwiftLog datasets: delivery-delay classification, product-return classification, weekly product-demand forecasting, and next-month customer-revenue prediction. The results are experimental and are not validated for production use.

## Project structure
- `Data/Original_Datasets/`: supplied source CSV files.
- `Data/Processed_Datasets/`: cleaned tables, task features, quality summary, and feature data dictionary.
- `Data/Feature_Engineered_Dataset.csv`: order-level engineered dataset used for the return-prediction task.
- `Notebook/Week5_MachineLearning.ipynb`: data-quality review, model comparison, evaluation artefacts, and interpretation.
- `Models/`: saved pipelines, model comparison, permutation importance, and confusion matrices.
- `App/`: Streamlit demonstration app.
- `Reports/`: technical report and executive summary.
- `Presentation/`: executive presentation.

## Run the project
1. Install Python packages: `python -m pip install -r App/requirements.txt`.
2. From the project root, run `python run_swiftlog_project.py` to regenerate the processed data, models, evaluation artefacts, reports, notebook, and presentation.
3. Launch the app with `python -m streamlit run App/app.py`.

## Prediction tasks
1. **Delivery delay classification:** target is `DelayDays > 0`; `DelayDays` is not used as an input feature.
2. **Product return classification:** target is whether an order appears in the Returns table.
3. **Weekly product demand forecasting:** predicts product units per week using prior-week and prior-four-week demand features. The test period is later than the training period.
4. **Next-month customer value:** predicts customer revenue for a target month using purchasing activity from the preceding three months. The test period is chronological.

## Outputs
- `Models/Best_Model.pkl`
- `Models/Model_Comparison.xlsx`
- `Models/Permutation_Importance.csv`
- `Models/Feature_Importance.png`
- `Reports/Technical_Report.pdf`
- `Reports/Executive_Summary.pdf`
- `Presentation/Executive_Presentation.pptx`

## Reproducibility note
The included model bundle was generated with scikit-learn 1.8.0. Install the matching version from `App/requirements.txt` when possible; loading a model with a different scikit-learn version can produce compatibility warnings.

## Results and limitations
Model comparison metrics are recorded in `Models/Model_Comparison.xlsx`. The current classification ROC-AUC scores indicate weak class discrimination. The selected regression models have negative held-out R-squared, meaning they performed worse than a simple mean baseline on the test period. These results are reported transparently and should not be presented as reliable production forecasts.

The supplied supplier data does not directly link suppliers to shipments, so supplier reliability prediction is not claimed. Delivery status may disagree with numeric delay days; this project defines a delay as `DelayDays > 0`. Customer value means next-month revenue, not lifetime value. The data covers a limited period and lacks some operationally important predictors.

## Recommended next steps
Collect richer historical data, verify target definitions, improve predictors, compare against simple baselines, validate on later periods, monitor performance, and require human review before any operational use.
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
print('Documentation, notebook, app warnings, README, and presentation refreshed.')
