"""Supplementary EDA and baseline checks for the SwiftLog Week 5 project.
Run from the project root: python enhance_swiftlog_eda.py
Does not retrain or overwrite saved models.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'Data' / 'Original_Datasets'
PROCESSED = ROOT / 'Data' / 'Processed_Datasets'
REPORTS = ROOT / 'Reports'
REPORTS.mkdir(parents=True, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(REPORTS / name, dpi=160, bbox_inches='tight')
    plt.close()

# Missing values by source-table column
missing_rows = []
for path in sorted(RAW.glob('*.csv')):
    df = pd.read_csv(path)
    for col in df.columns:
        missing_rows.append({'Dataset': path.stem, 'Column': col, 'Rows': len(df), 'Missing_Count': int(df[col].isna().sum()), 'Missing_Percent': round(float(df[col].isna().mean() * 100), 3)})
missing = pd.DataFrame(missing_rows)
missing.to_csv(REPORTS / 'eda_missing_values_by_column.csv', index=False)
top_missing = missing[missing.Missing_Count > 0].sort_values('Missing_Percent', ascending=False).head(25)
if not top_missing.empty:
    plt.figure(figsize=(10, 7))
    labels = top_missing.Dataset + ': ' + top_missing.Column
    plt.barh(labels[::-1], top_missing.Missing_Percent[::-1])
    plt.xlabel('Missing values (%)')
    plt.title('Columns with the highest missing-value rates')
    savefig('eda_missing_values.png')

# Numeric distributions and IQR outlier summary
outlier_rows = []
for path in sorted(RAW.glob('*.csv')):
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        outlier_rows.append({'Dataset': path.stem, 'Column': col, 'Non_Missing_Values': len(series), 'Outlier_Count_IQR': count, 'Outlier_Percent': round(100 * count / len(series), 3), 'Median': float(series.median()), 'Q1': float(q1), 'Q3': float(q3)})
        if len(series) >= 5:
            plt.figure(figsize=(7, 3.5))
            plt.hist(series, bins=30)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title(f'{path.stem}: distribution of {col}')
            savefig(f'eda_distribution_{path.stem.lower()}_{col.lower()}.png')
pd.DataFrame(outlier_rows).to_csv(REPORTS / 'eda_numeric_outlier_summary.csv', index=False)

# Numeric correlation heatmaps
for path in sorted(RAW.glob('*.csv')):
    numeric = pd.read_csv(path).select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        continue
    corr = numeric.corr(numeric_only=True)
    if corr.empty:
        continue
    size = max(6, min(12, 0.7 * len(corr.columns)))
    plt.figure(figsize=(size, size * 0.8))
    image = plt.imshow(corr, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(image, label='Pearson correlation')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha='right', fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title(f'Numeric feature correlations: {path.stem}')
    savefig(f'eda_correlation_{path.stem.lower()}.png')

# Classification target distributions
for filename, target, title, output in [
    ('delivery_delay_features.csv', 'Target_Delayed', 'Delivery delay target', 'eda_target_delivery_delay.png'),
    ('product_return_features.csv', 'Target_Returned', 'Product return target', 'eda_target_product_return.png')]:
    path = PROCESSED / filename
    if not path.exists():
        continue
    df = pd.read_csv(path)
    if target not in df:
        continue
    counts = df[target].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([str(x) for x in counts.index], counts.values)
    plt.xlabel('Target class (0 = No, 1 = Yes)')
    plt.ylabel('Records')
    plt.title(title)
    savefig(output)

# Baselines using the same split conventions as the main pipeline
baseline_rows = []
def add_classification_baseline(problem, filename, target):
    path = PROCESSED / filename
    if not path.exists(): return
    df = pd.read_csv(path).dropna(subset=[target]).reset_index(drop=True)
    y = df[target].astype(int)
    strat = y if y.value_counts().min() >= 2 else None
    _, X_test, _, y_test = train_test_split(df.drop(columns=[target]), y, test_size=0.2, random_state=42, stratify=strat)
    X_train, _, y_train, _ = train_test_split(df.drop(columns=[target]), y, test_size=0.2, random_state=42, stratify=strat)
    majority = int(y_train.value_counts().idxmax())
    pred = np.full(len(y_test), majority, dtype=int)
    baseline_rows.append({'Problem': problem, 'Baseline': 'Training-set majority class', 'Test_Rows': len(y_test), 'Accuracy': accuracy_score(y_test, pred), 'Precision': precision_score(y_test, pred, zero_division=0), 'Recall': recall_score(y_test, pred, zero_division=0), 'F1': f1_score(y_test, pred, zero_division=0), 'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan})

def add_regression_baseline(problem, filename, target, date_col):
    path = PROCESSED / filename
    if not path.exists(): return
    df = pd.read_csv(path).dropna(subset=[target]).reset_index(drop=True)
    dates = pd.to_datetime(df[date_col], errors='coerce')
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) < 2: return
    cut = max(1, int(len(unique_dates) * 0.8))
    test_start = unique_dates[cut] if cut < len(unique_dates) else unique_dates[-1]
    train_mask, test_mask = dates < test_start, dates >= test_start
    y_train = pd.to_numeric(df.loc[train_mask, target], errors='coerce').dropna()
    y_test = pd.to_numeric(df.loc[test_mask, target], errors='coerce').dropna()
    if y_train.empty or y_test.empty: return
    pred = np.full(len(y_test), float(y_train.mean()))
    baseline_rows.append({'Problem': problem, 'Baseline': 'Training-period mean', 'Test_Rows': len(y_test), 'Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'MAE': mean_absolute_error(y_test, pred), 'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))), 'R2': r2_score(y_test, pred)})

add_classification_baseline('Delivery Delay Classification', 'delivery_delay_features.csv', 'Target_Delayed')
add_classification_baseline('Product Return Classification', 'product_return_features.csv', 'Target_Returned')
add_regression_baseline('Product Demand Forecasting', 'demand_forecasting_features.csv', 'DemandQuantity', 'WeekStart')
add_regression_baseline('Customer Value Prediction (Next Month)', 'customer_value_features.csv', 'MonthRevenue', 'MonthStart')
baselines = pd.DataFrame(baseline_rows)
baselines.to_csv(REPORTS / 'Baseline_Comparison.csv', index=False)
print('Supplementary EDA completed. Outputs saved to:', REPORTS)
print('\nBaseline comparison:')
print(baselines.to_string(index=False) if not baselines.empty else 'No baseline rows produced. Check processed feature files.')
