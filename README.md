# SwiftLog Logistics Analytics and AI Transformation

**Machine Learning Track | Mission ML05: Intelligent Logistics Prediction and Supply Chain Optimization**

## Project Overview

This project explores how machine learning can support logistics operations and supply chain decision-making using the datasets provided for the AnalystLab Africa Week 5 assignment.

The project implements four machine-learning tasks:

- Delivery delay classification
- Product return classification
- Weekly product-demand forecasting
- Next-month customer-revenue prediction

The work covers data quality assessment, preprocessing, feature engineering, model comparison, evaluation, interpretation, and a Streamlit demonstration application.

This is an educational prototype. The models are experimental, and their results do not establish readiness for real-world operational use.

## Project Structure

- `Data/Original_Datasets/`: Original CSV datasets supplied for the assignment.
- `Data/Processed_Datasets/`: Cleaned tables, task-specific features, data-quality summaries, and the feature data dictionary.
- `Data/Feature_Engineered_Dataset.csv`: Engineered order-level dataset used for return prediction.
- `Notebook/Week5_MachineLearning.ipynb`: Notebook documenting data-quality review, model comparisons, evaluation outputs, and interpretation.
- `Models/`: Saved model pipelines, model comparisons, permutation importance, and confusion matrices.
- `App/`: Streamlit demonstration application and Python dependencies.
- `Reports/`: Technical report and executive summary.
- `Presentation/`: Executive presentation.
- `run_swiftlog_project.py`: Main project pipeline for regenerating project outputs.

## Requirements

- Python 3.11.9 was used in the working environment.
- The saved model bundle was generated using scikit-learn 1.8.0.
- Install the dependencies listed in `App/requirements.txt`.

## Setup and Execution

Run the following commands from the project root directory.

### 1. Install Dependencies

```bash
python -m pip install -r App/requirements.txt
```

### 2. Regenerate the Project Outputs

```bash
python run_swiftlog_project.py
```

This runs the project pipeline to regenerate the processed data, models, evaluation artefacts, reports, notebook, and presentation.

### 3. Launch the Streamlit Application

```bash
python -m streamlit run App/app.py
```

Follow the local URL displayed in the terminal to open the application in your browser.

**Note:** Regenerating the outputs may overwrite existing artefacts. Use the supplied Python environment and compatible package versions where possible.

## Prediction Tasks

### 1. Delivery Delay Classification

Classifies deliveries as delayed or not delayed. The target is defined as `DelayDays > 0`, and `DelayDays` is excluded from the input features to prevent direct target leakage.

### 2. Product Return Classification

Predicts whether an order appears in the Returns dataset. The task uses the available order-level information and engineered features.

### 3. Weekly Product-Demand Forecasting

Predicts product units demanded per week using historical demand features, including prior-week and prior-four-week demand. Evaluation uses a later test period than the training period.

### 4. Next-Month Customer-Revenue Prediction

Predicts customer revenue for a target month using purchasing activity from the preceding three months. Evaluation follows a chronological split.

## Key Outputs

- `Models/Best_Model.pkl`: Saved model bundle for the four prediction tasks.
- `Models/Model_Comparison.xlsx`: Comparison of candidate models and evaluation metrics.
- `Models/Permutation_Importance.csv`: Permutation feature-importance results.
- `Models/Feature_Importance.png`: Feature-importance visualisation.
- `Reports/Technical_Report.pdf`: Technical project report.
- `Reports/Executive_Summary.pdf`: Summary of the project and its findings.
- `Presentation/Executive_Presentation.pptx`: Executive presentation.

Additional evaluation visualisations and processed datasets are available in their respective folders.

## Results and Limitations

Detailed model-comparison metrics are recorded in `Models/Model_Comparison.xlsx`.

The evaluation highlights several limitations:

- **Weak classification discrimination:** The classification ROC-AUC results indicate limited ability to distinguish between classes. High accuracy, precision, recall, or F1 scores should therefore not be interpreted in isolation.
- **Limited forecasting performance:** The selected regression models have negative held-out R-squared values, meaning they performed worse than a baseline that predicts the test-set mean.
- **Supplier data limitations:** The supplied supplier dataset does not provide a direct link between suppliers and shipments. Supplier reliability prediction is therefore not claimed.
- **Target definition:** Delivery delay is defined using `DelayDays > 0`. The definition may not agree with every recorded delivery-status label.
- **Customer-value definition:** Customer value refers to predicted next-month revenue, not customer lifetime value.
- **Limited predictors and history:** The available data covers a limited period and lacks some operationally important predictors.

These findings mean the models should be treated as experimental demonstrations rather than reliable operational prediction systems.

## Reproducibility

The saved model bundle was generated using scikit-learn 1.8.0. Using the matching version is recommended because loading models across different scikit-learn versions may produce compatibility warnings or unexpected behaviour.

For reproducible results, use compatible dependency versions and retain the supplied datasets and project configuration.

## Recommended Next Steps

Further development should focus on:

1. Collecting richer historical logistics and customer data.
2. Verifying target definitions and resolving inconsistencies in source data.
3. Improving predictive features and comparing models against simple baselines.
4. Validating models on later, unseen periods.
5. Monitoring model performance and data quality.
6. Requiring human review before using predictions to inform operational decisions.

## Project Status

This project demonstrates an end-to-end educational machine-learning workflow, from data preparation and model evaluation to saved artefacts and an interactive prediction application. The evaluation results and data limitations are documented to support transparent interpretation and future improvement.