# SwiftLog Logistics Analytics and AI Transformation

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
