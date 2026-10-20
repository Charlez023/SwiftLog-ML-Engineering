
# SwiftLog Week 5 ML project builder
# Run from project root: python run_swiftlog_project.py

from pathlib import Path
import math
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

ROOT = Path.cwd()
ORIGINAL = ROOT / "Data" / "Original_Datasets"
PROCESSED = ROOT / "Data" / "Processed_Datasets"
MODELS = ROOT / "Models"
REPORTS = ROOT / "Reports"
APP_DIR = ROOT / "App"

for folder in [ORIGINAL, PROCESSED, ROOT / "Notebook", MODELS,
               APP_DIR / "assets", REPORTS, ROOT / "Presentation", ROOT / "References"]:
    folder.mkdir(parents=True, exist_ok=True)


def data_file(name):
    for folder in [ORIGINAL, ROOT / "Data"]:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {name}; expected it in Data/Original_Datasets or Data/.")


def load(name):
    df = pd.read_csv(data_file(name))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_text(df):
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )
    return df


def require(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")


def add_date_parts(df, col):
    df = df.copy()
    d = pd.to_datetime(df[col], errors="coerce")
    for suffix, values in [
        ("Year", d.dt.year), ("Month", d.dt.month),
        ("Day", d.dt.day), ("Weekday", d.dt.dayofweek),
    ]:
        df[f"{col}_{suffix}"] = values
    return df


def categorical_ids(X):
    X = X.copy()
    for c in X.columns:
        if c.lower().endswith("id"):
            X[c] = X[c].astype("string")
    return X


def preprocessor(X):
    X = categorical_ids(X)
    cats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    if not transformers:
        raise ValueError("No usable features.")
    return ColumnTransformer(transformers, remainder="drop")


def prob_positive(pipe, X):
    if not hasattr(pipe, "predict_proba"):
        return None
    classes = list(pipe.named_steps["model"].classes_)
    return pipe.predict_proba(X)[:, classes.index(1)] if 1 in classes else None


def cls_metrics(pipe, X, y, pred):
    m = {
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": np.nan,
    }
    if pd.Series(y).nunique() == 2:
        try:
            p = prob_positive(pipe, X)
            if p is not None:
                m["ROC_AUC"] = float(roc_auc_score(y, p))
        except (ValueError, IndexError):
            pass
    return m


def reg_metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
    }


def split_data(X, y, dates, kind, task):
    if dates is not None:
        dates = pd.to_datetime(dates, errors="coerce")
        unique_dates = np.array(sorted(dates.dropna().unique()))
        if len(unique_dates) < 5:
            raise ValueError(f"{task} has fewer than 5 distinct dates; chronological evaluation is unreliable.")
        cut1 = max(1, int(len(unique_dates) * .60))
        cut2 = min(max(cut1 + 1, int(len(unique_dates) * .80)), len(unique_dates) - 1)
        masks = [
            dates.isin(unique_dates[:cut1]),
            dates.isin(unique_dates[cut1:cut2]),
            dates.isin(unique_dates[cut2:]),
        ]
        parts = []
        for mask in masks:
            parts.extend([X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)])
        info = {
            "Split_Method": "Chronological by unique date",
            "Train_Date_Start": str(dates[masks[0]].min().date()),
            "Train_Date_End": str(dates[masks[0]].max().date()),
            "Validation_Date_Start": str(dates[masks[1]].min().date()),
            "Validation_Date_End": str(dates[masks[1]].max().date()),
            "Test_Date_Start": str(dates[masks[2]].min().date()),
            "Test_Date_End": str(dates[masks[2]].max().date()),
        }
        return (*parts, info)
    strat = y if kind == "classification" and y.value_counts().min() >= 2 else None
    Xtmp, Xte, ytmp, yte = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=strat
    )
    strat2 = ytmp if kind == "classification" and ytmp.value_counts().min() >= 2 else None
    Xtr, Xval, ytr, yval = train_test_split(
        Xtmp, ytmp, test_size=.25, random_state=42, stratify=strat2
    )
    return Xtr, ytr, Xval, yval, Xte, yte, {
        "Split_Method": "Random 60/20/20; stratified where possible"
    }


# Load, audit, and deduplicate working copies. Original CSVs are never modified.
names = [
    "SwiftLog_Customers.csv", "SwiftLog_Delivery_Performance.csv",
    "SwiftLog_Employees.csv", "SwiftLog_Inventory.csv", "SwiftLog_Orders.csv",
    "SwiftLog_Order_Items.csv", "SwiftLog_Products.csv", "SwiftLog_Returns.csv",
    "SwiftLog_Shipments.csv", "SwiftLog_Suppliers.csv", "SwiftLog_Warehouses.csv",
]
tables, quality = {}, []
for filename in names:
    key = filename.replace("SwiftLog_", "").replace(".csv", "")
    df = clean_text(load(filename))
    quality.append({
        "Dataset": key, "Rows_Before_Deduplication": len(df), "Columns": len(df.columns),
        "Duplicate_Rows": int(df.duplicated().sum()),
        "Missing_Cells": int(df.isna().sum().sum()),
        "Columns_With_Missing_Values": ", ".join(df.columns[df.isna().any()].tolist()),
    })
    tables[key] = df.drop_duplicates().reset_index(drop=True)
    tables[key].to_csv(PROCESSED / f"{key.lower()}_clean.csv", index=False)
pd.DataFrame(quality).to_csv(PROCESSED / "data_quality_summary.csv", index=False)

customers, delivery = tables["Customers"], tables["Delivery_Performance"]
employees, inventory, orders = tables["Employees"], tables["Inventory"], tables["Orders"]
items, products, returns = tables["Order_Items"], tables["Products"], tables["Returns"]
shipments, suppliers, warehouses = tables["Shipments"], tables["Suppliers"], tables["Warehouses"]

require(customers, ["CustomerID"], "Customers")
require(orders, ["OrderID", "CustomerID", "OrderDate"], "Orders")
require(items, ["OrderID", "ProductID", "OrderItemID", "Quantity"], "Order_Items")
require(products, ["ProductID", "UnitPrice"], "Products")
require(returns, ["OrderID"], "Returns")
require(shipments, ["ShipmentID", "OrderID", "WarehouseID", "DelayDays"], "Shipments")
require(warehouses, ["WarehouseID"], "Warehouses")

pd.DataFrame([
    {"Table": "Shipments", "Relationship": "OrderID -> Orders.OrderID", "Finding": "Join supported by supplied keys"},
    {"Table": "Shipments", "Relationship": "WarehouseID -> Warehouses.WarehouseID", "Finding": "Join supported; location only if supplied"},
    {"Table": "Delivery_Performance", "Relationship": "ShipmentID -> Shipments.ShipmentID", "Finding": "Potential outcome cross-check; status excluded from predictors"},
    {"Table": "Inventory", "Relationship": "WarehouseID/ProductID -> Warehouses/Products", "Finding": "Keys exist; repeated pairs are not assumed to be time history"},
    {"Table": "Employees", "Relationship": "WarehouseID -> Warehouses.WarehouseID", "Finding": "Warehouse association supported by schema"},
    {"Table": "Suppliers", "Relationship": "SupplierID -> other tables", "Finding": "No confirmed matching supplier key in inspected schemas"},
    {"Table": "Warehouses", "Relationship": "Capacity", "Finding": "Unavailable; supplied table has no capacity field"},
]).to_csv(PROCESSED / "relationship_audit.csv", index=False)

orders2 = add_date_parts(orders.copy(), "OrderDate")
orders2["OrderDate"] = pd.to_datetime(orders2["OrderDate"], errors="coerce")
order_items = items.merge(products, on="ProductID", how="left", validate="many_to_one")
order_items["Quantity"] = pd.to_numeric(order_items["Quantity"], errors="coerce")
order_items["UnitPrice"] = pd.to_numeric(order_items["UnitPrice"], errors="coerce")
order_items["LineRevenue"] = order_items["Quantity"] * order_items["UnitPrice"]
order_items.to_csv(PROCESSED / "order_items_enriched.csv", index=False)

# Task 1: shipment delay. DelayDays is only used to define the historical label.
ship = shipments.copy()
ship["DelayDays"] = pd.to_numeric(ship["DelayDays"], errors="coerce")
ship = ship.dropna(subset=["DelayDays"]).merge(
    orders2, on="OrderID", how="left", suffixes=("", "_Order"), validate="many_to_one"
)
if "Location" in warehouses.columns:
    ship = ship.merge(warehouses[["WarehouseID", "Location"]], on="WarehouseID",
                      how="left", validate="many_to_one")
if "City" in customers.columns and "CustomerID" in ship.columns:
    ship = ship.merge(customers[["CustomerID", "City"]], on="CustomerID",
                      how="left", validate="many_to_one")
ship["Target_Delayed"] = (ship["DelayDays"] > 0).astype(int)
delay_cols = [c for c in [
    "WarehouseID", "Location", "City", "OrderDate_Year", "OrderDate_Month",
    "OrderDate_Day", "OrderDate_Weekday"
] if c in ship.columns]
if not delay_cols:
    raise ValueError("No valid pre-outcome delivery features were found.")
delay_X, delay_y = categorical_ids(ship[delay_cols]), ship["Target_Delayed"].astype(int)
delay_X.assign(Target_Delayed=delay_y).to_csv(PROCESSED / "delivery_delay_features.csv", index=False)

# Task 2: recorded return classification. Features represent order details known at order time.
ret_ids = set(returns["OrderID"].dropna().astype(str))
order_feat = orders2.copy()
if "City" in customers.columns:
    order_feat = order_feat.merge(customers[["CustomerID", "City"]], on="CustomerID",
                                  how="left", validate="many_to_one")
item_agg = order_items.groupby("OrderID").agg(
    ItemCount=("OrderItemID", "count"), TotalQuantity=("Quantity", "sum"),
    DistinctProducts=("ProductID", "nunique"), TotalRevenue=("LineRevenue", "sum"),
    MeanUnitPrice=("UnitPrice", "mean"),
).reset_index()
order_feat = order_feat.merge(item_agg, on="OrderID", how="left", validate="one_to_one")
order_feat["Target_Returned"] = order_feat["OrderID"].astype(str).isin(ret_ids).astype(int)
return_cols = [c for c in [
    "City", "OrderDate_Year", "OrderDate_Month", "OrderDate_Day", "OrderDate_Weekday",
    "ItemCount", "TotalQuantity", "DistinctProducts", "TotalRevenue", "MeanUnitPrice"
] if c in order_feat.columns]
return_X, return_y = categorical_ids(order_feat[return_cols]), order_feat["Target_Returned"].astype(int)
return_X.assign(Target_Returned=return_y).to_csv(PROCESSED / "product_return_features.csv", index=False)

# Task 3: weekly product demand; all lag/rolling features are shifted before the target week.
item_date = items.merge(orders2[["OrderID", "OrderDate"]], on="OrderID", how="left",
                        validate="many_to_one").merge(products, on="ProductID", how="left",
                                                       validate="many_to_one")
item_date["OrderDate"] = pd.to_datetime(item_date["OrderDate"], errors="coerce")
item_date["Quantity"] = pd.to_numeric(item_date["Quantity"], errors="coerce")
item_date = item_date.dropna(subset=["OrderDate", "ProductID", "Quantity"]).copy()
item_date["WeekStart"] = item_date["OrderDate"].dt.to_period("W-SUN").dt.start_time
weekly = item_date.groupby(["ProductID", "WeekStart"], as_index=False).agg(
    DemandQuantity=("Quantity", "sum")
)
product_cols = ["ProductID"] + [c for c in ["Category", "UnitPrice"] if c in products.columns]
product_info = products[product_cols].drop_duplicates("ProductID")
if weekly.empty:
    raise ValueError("No valid dated order-item quantities are available for demand forecasting.")
weeks = pd.date_range(weekly["WeekStart"].min(), weekly["WeekStart"].max(), freq="7D")
grid = pd.MultiIndex.from_product(
    [product_info["ProductID"].dropna().tolist(), weeks],
    names=["ProductID", "WeekStart"]
).to_frame(index=False)
demand = grid.merge(weekly, on=["ProductID", "WeekStart"], how="left").merge(
    product_info, on="ProductID", how="left", validate="many_to_one"
)
demand["DemandQuantity"] = demand["DemandQuantity"].fillna(0.0)
demand = demand.sort_values(["ProductID", "WeekStart"]).reset_index(drop=True)
g = demand.groupby("ProductID")["DemandQuantity"]
demand["Lag_1_Week"] = g.shift(1)
demand["Lag_4_Weeks"] = g.shift(4)
demand["Rolling_Mean_4_Weeks"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
demand["Year"] = demand["WeekStart"].dt.year
demand["Month"] = demand["WeekStart"].dt.month
demand["DayOfWeek"] = demand["WeekStart"].dt.dayofweek
demand = demand.dropna(subset=["Lag_1_Week", "Lag_4_Weeks", "Rolling_Mean_4_Weeks"])
demand = demand.sort_values(["WeekStart", "ProductID"]).reset_index(drop=True)
demand_cols = [c for c in [
    "ProductID", "Category", "UnitPrice", "Year", "Month", "DayOfWeek",
    "Lag_1_Week", "Lag_4_Weeks", "Rolling_Mean_4_Weeks"
] if c in demand.columns]
demand_X, demand_y, demand_dates = categorical_ids(demand[demand_cols]), demand["DemandQuantity"].astype(float), demand["WeekStart"]
demand_X.assign(DemandQuantity=demand_y, WeekStart=demand_dates.astype(str)).to_csv(
    PROCESSED / "demand_forecasting_features.csv", index=False
)

# Task 4: target is each customer's revenue in the target month; predictors use only prior months.
order_month = orders2.dropna(subset=["OrderDate"]).copy()
order_month["MonthStart"] = order_month["OrderDate"].dt.to_period("M").dt.to_timestamp()
item_month = order_items.merge(order_month[["OrderID", "CustomerID", "MonthStart"]],
                               on="OrderID", how="inner", validate="many_to_one")
mi = item_month.groupby(["CustomerID", "MonthStart"]).agg(
    MonthRevenue=("LineRevenue", "sum"), MonthQuantity=("Quantity", "sum"),
    MonthItems=("OrderItemID", "count"), MonthAvgUnitPrice=("UnitPrice", "mean")
).reset_index()
mo = order_month.groupby(["CustomerID", "MonthStart"]).agg(
    MonthOrders=("OrderID", "nunique")
).reset_index()
monthly = mi.merge(mo, on=["CustomerID", "MonthStart"], how="outer", validate="one_to_one")
months = pd.date_range(order_month["MonthStart"].min(), order_month["MonthStart"].max(), freq="MS")
cust_ids = customers["CustomerID"].dropna().unique().tolist()
cg = pd.MultiIndex.from_product([cust_ids, months], names=["CustomerID", "MonthStart"]).to_frame(index=False)
monthly = cg.merge(monthly, on=["CustomerID", "MonthStart"], how="left", validate="one_to_one")
for c in ["MonthRevenue", "MonthQuantity", "MonthItems", "MonthOrders"]:
    monthly[c] = pd.to_numeric(monthly[c], errors="coerce").fillna(0.0)
monthly = monthly.sort_values(["CustomerID", "MonthStart"]).reset_index(drop=True)

def prior_sum(s):
    return s.shift(1).rolling(3, min_periods=3).sum()

def prior_mean(s):
    return s.shift(1).rolling(3, min_periods=3).mean()

for source, dest in [
    ("MonthRevenue", "Previous_3M_Revenue"),
    ("MonthOrders", "Previous_3M_Order_Count"),
    ("MonthQuantity", "Previous_3M_Quantity"),
    ("MonthItems", "Previous_3M_Item_Count"),
]:
    monthly[dest] = monthly.groupby("CustomerID")[source].transform(prior_sum)
monthly["Previous_3M_Avg_Unit_Price"] = monthly.groupby("CustomerID")["MonthAvgUnitPrice"].transform(prior_mean)
if "City" in customers.columns:
    monthly = monthly.merge(customers[["CustomerID", "City"]], on="CustomerID",
                            how="left", validate="many_to_one")
else:
    monthly["City"] = "Unknown"
monthly = monthly.dropna(subset=[
    "Previous_3M_Revenue", "Previous_3M_Order_Count",
    "Previous_3M_Quantity", "Previous_3M_Item_Count"
]).sort_values(["MonthStart", "CustomerID"]).reset_index(drop=True)
value_cols = [
    "City", "Previous_3M_Revenue", "Previous_3M_Order_Count",
    "Previous_3M_Quantity", "Previous_3M_Item_Count", "Previous_3M_Avg_Unit_Price"
]
value_X, value_y, value_dates = categorical_ids(monthly[value_cols]), monthly["MonthRevenue"].astype(float), monthly["MonthStart"]
monthly.to_csv(PROCESSED / "customer_value_features.csv", index=False)
value_X.assign(Target_Next_Month_Revenue=value_y, MonthStart=value_dates.astype(str)).to_csv(
    PROCESSED / "customer_value_model_features.csv", index=False
)

problems = {
    "Delivery Delay Classification": (delay_X, delay_y, "classification", None),
    "Product Return Classification": (return_X, return_y, "classification", None),
    "Product Demand Forecasting": (demand_X, demand_y, "regression", demand_dates),
    "Customer Value Prediction (Next Month)": (value_X, value_y, "regression", value_dates),
}
results, bundles, split_audit, importances = [], {}, [], []

for task, (X, y, kind, dates) in problems.items():
    valid = y.notna()
    X, y = categorical_ids(X.loc[valid].reset_index(drop=True)), y.loc[valid].reset_index(drop=True)
    if dates is not None:
        dates = dates.loc[valid].reset_index(drop=True)
    if len(X) < 30:
        raise ValueError(f"{task} has only {len(X)} rows; at least 30 are required.")
    if kind == "classification" and y.nunique() < 2:
        raise ValueError(f"{task} target has fewer than two classes.")

    Xtr, ytr, Xval, yval, Xte, yte, sinfo = split_data(X, y, dates, kind, task)
    if min(len(Xtr), len(Xval), len(Xte)) == 0:
        raise ValueError(f"{task} has an empty train/validation/test split.")
    split_audit.append({
        "Problem": task, "Rows": len(X), "Train_Rows": len(Xtr),
        "Validation_Rows": len(Xval), "Test_Rows": len(Xte),
        "Train_Percent": round(100 * len(Xtr) / len(X), 2),
        "Validation_Percent": round(100 * len(Xval) / len(X), 2),
        "Test_Percent": round(100 * len(Xte) / len(X), 2), **sinfo
    })

    if kind == "classification":
        algos = {
            "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                class_weight="balanced_subsample", random_state=42, n_jobs=1),
        }
    else:
        algos = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10,
                random_state=42, n_jobs=1),
        }

    candidates = []
    for name, estimator in algos.items():
        pipe = Pipeline([("preprocessor", preprocessor(Xtr)), ("model", estimator)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xval)
        row = {
            "Problem": task, "Algorithm": name, "Rows": len(X), "Features": X.shape[1],
            "Train_Rows": len(Xtr), "Validation_Rows": len(Xval), "Test_Rows": len(Xte),
            "Selected": False, "Tuned": False,
            "Selection_Basis": "Validation F1" if kind == "classification" else "Validation RMSE",
        }
        if kind == "classification":
            vm = cls_metrics(pipe, Xval, yval, pred)
            row.update({f"Validation_{k}": v for k, v in vm.items()})
            score = vm["F1"]
        else:
            vm = reg_metrics(yval, pred)
            row.update({f"Validation_{k}": v for k, v in vm.items()})
            score = -vm["RMSE"]
        results.append(row)
        candidates.append((score, name, pipe, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_pipe, best_row = candidates[0]

    # Tune only the validation-selected tree family. The held-out test set is not used here.
    grid = None
    if best_name in ("Random Forest", "Random Forest Regressor"):
        grid = {"model__n_estimators": [80, 120],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 3]}
    elif best_name in ("Decision Tree", "Decision Tree Regressor"):
        grid = {"model__max_depth": [5, 10, 15],
                "model__min_samples_leaf": [1, 3]}
    if grid:
        try:
            cv = TimeSeriesSplit(n_splits=2) if dates is not None else 3
            search = GridSearchCV(
                best_pipe, grid,
                scoring="f1" if kind == "classification" else "neg_root_mean_squared_error",
                cv=cv, n_jobs=1, error_score="raise"
            )
            search.fit(Xtr, ytr)
            tuned = search.best_estimator_
            tuned_pred = tuned.predict(Xval)
            if kind == "classification":
                tuned_score = cls_metrics(tuned, Xval, yval, tuned_pred)["F1"]
            else:
                tuned_score = -reg_metrics(yval, tuned_pred)["RMSE"]
            if tuned_score >= best_score:
                best_pipe, best_name, best_score = tuned, best_name + " (tuned)", tuned_score
                best_row["Tuned"] = True
                best_row["Tuned_Validation_Score"] = tuned_score
        except Exception as exc:
            print(f"Tuning skipped for {task}: {exc}")

    Xfinal, yfinal = pd.concat([Xtr, Xval], ignore_index=True), pd.concat([ytr, yval], ignore_index=True)
    best_pipe.fit(Xfinal, yfinal)
    test_pred = best_pipe.predict(Xte)
    if kind == "classification":
        tm = cls_metrics(best_pipe, Xte, yte, test_pred)
        cm = confusion_matrix(yte, test_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{task}: Held-out Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(MODELS / f"{task.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
        plt.close(fig)
    else:
        tm = reg_metrics(yte, test_pred)
    best_row["Algorithm"], best_row["Selected"] = best_name, True
    best_row.update({f"Test_{k}": v for k, v in tm.items()})

    defaults, types = {}, {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            med = pd.to_numeric(X[c], errors="coerce").median()
            defaults[c], types[c] = (float(med) if pd.notna(med) else 0.0), "numeric"
        else:
            mode = X[c].mode(dropna=True)
            defaults[c], types[c] = (str(mode.iloc[0]) if not mode.empty else "Unknown"), "categorical"
    bundles[task] = {
        "model": best_pipe, "features": list(X.columns), "defaults": defaults, "types": types,
        "kind": kind, "metrics": tm, "algorithm": best_name,
        "target_definition": {
            "Delivery Delay Classification": "1 when recorded DelayDays > 0; DelayDays itself is not a predictor.",
            "Product Return Classification": "1 when OrderID appears in the supplied Returns table.",
            "Product Demand Forecasting": "Recorded product quantity in the target week.",
            "Customer Value Prediction (Next Month)": "Customer revenue in the target month, using prior three-month features.",
        }[task],
        "warning": "Educational prototype. Historical test performance may be weak; validate before operational use.",
    }
    try:
        n = min(1000, len(Xte))
        Xi = Xte.sample(n=n, random_state=42)
        yi = yte.loc[Xi.index]
        pi = permutation_importance(
            best_pipe, Xi, yi, n_repeats=3, random_state=42, n_jobs=1,
            scoring="f1" if kind == "classification" else "neg_root_mean_squared_error"
        )
        for feature, score in zip(X.columns, pi.importances_mean):
            importances.append({"Problem": task, "Feature": feature, "Importance": float(score)})
    except Exception as exc:
        print(f"Feature importance unavailable for {task}: {exc}")
    print(f"Completed: {task} | selected={best_name} | held-out test={tm}")

joblib.dump(bundles, MODELS / "Best_Model.pkl")
pd.DataFrame(split_audit).to_csv(MODELS / "Data_Split_Summary.csv", index=False)
pd.DataFrame(results).to_excel(MODELS / "Model_Comparison.xlsx", index=False)
imp = pd.DataFrame(importances)
imp.to_csv(MODELS / "Permutation_Importance.csv", index=False)
if not imp.empty:
    top = imp.sort_values("Importance", ascending=False).head(20).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Problem"] + ": " + top["Feature"], top["Importance"])
    ax.axvline(0, linewidth=.8)
    ax.set_title("Permutation Feature Importance (top 20)")
    ax.set_xlabel("Score change after feature permutation")
    fig.tight_layout()
    fig.savefig(MODELS / "Feature_Importance.png", dpi=160)
    plt.close(fig)
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(.05, .5, "Permutation importance unavailable for this run")
    ax.axis("off")
    fig.savefig(MODELS / "Feature_Importance.png", dpi=150)
    plt.close(fig)

q = pd.DataFrame(quality)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(q["Dataset"], q["Rows_Before_Deduplication"])
ax.set_ylabel("Rows before exact-duplicate removal")
ax.set_title("Supplied dataset row counts")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(REPORTS / "dataset_row_counts.png", dpi=150)
plt.close(fig)

feature_dictionary = [
    ("Target_Delayed", "Delivery Delay Classification", "1 when DelayDays > 0; otherwise 0. DelayDays is excluded from predictors."),
    ("Target_Returned", "Product Return Classification", "1 when OrderID appears in Returns; otherwise 0."),
    ("Lag_1_Week", "Product Demand Forecasting", "Product quantity in the preceding week."),
    ("Lag_4_Weeks", "Product Demand Forecasting", "Product quantity four weeks before the target week."),
    ("Rolling_Mean_4_Weeks", "Product Demand Forecasting", "Mean demand in the four preceding weeks, excluding target week."),
    ("Previous_3M_Revenue", "Customer Value Prediction (Next Month)", "Customer revenue over the three months preceding target month."),
    ("Previous_3M_Order_Count", "Customer Value Prediction (Next Month)", "Customer order count over the three months preceding target month."),
    ("Previous_3M_Quantity", "Customer Value Prediction (Next Month)", "Customer quantity over the three months preceding target month."),
    ("Previous_3M_Item_Count", "Customer Value Prediction (Next Month)", "Order-item records over the three months preceding target month."),
    ("Previous_3M_Avg_Unit_Price", "Customer Value Prediction (Next Month)", "Mean unit price in prior months; can be missing if no prior purchases."),
    ("DemandQuantity", "Product Demand Forecasting", "Recorded units for a product-week; zero means no purchase was recorded."),
    ("MonthRevenue", "Customer Value Prediction (Next Month)", "Customer item revenue in target month."),
]
pd.DataFrame(feature_dictionary, columns=["Field", "Task", "Definition"]).to_csv(
    PROCESSED / "Feature_Data_Dictionary.csv", index=False
)
order_feat.to_csv(ROOT / "Data" / "Feature_Engineered_Dataset.csv", index=False)

app_source = r'''import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="SwiftLog ML Decision Support", page_icon="📦", layout="wide")
st.title("SwiftLog Logistics: ML Decision Support")
st.caption("Educational prototype using supplied Week 5 datasets. Predictions support review, not operational decisions.")
MODEL_PATH = Path(__file__).resolve().parents[1] / "Models" / "Best_Model.pkl"

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

models = load_models()
task = st.selectbox("Choose a prediction task", list(models.keys()))
bundle = models[task]
st.write("Selected model:", bundle["algorithm"])
st.write("Held-out test metrics:", bundle["metrics"])
st.caption(bundle.get("target_definition", ""))

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(bundle["features"]):
        with columns[i % 2]:
            default = bundle["defaults"].get(feature, 0)
            if bundle["types"].get(feature) == "numeric":
                values[feature] = st.number_input(feature, value=float(default))
            else:
                values[feature] = st.text_input(feature, value=str(default))
    submitted = st.form_submit_button("Generate prediction")

if submitted:
    X = pd.DataFrame([values], columns=bundle["features"])
    model = bundle["model"]
    prediction = model.predict(X)[0]
    if bundle["kind"] == "classification":
        positive = "Delayed" if "Delivery Delay" in task else "Returned"
        negative = "Not delayed" if "Delivery Delay" in task else "Not returned"
        st.subheader("Prediction")
        st.success(positive if int(prediction) == 1 else negative)
        if hasattr(model, "predict_proba"):
            try:
                classes = list(model.named_steps["model"].classes_)
                if 1 in classes:
                    p = model.predict_proba(X)[0][classes.index(1)]
                    st.metric("Estimated probability of positive class", f"{p:.1%}")
            except (ValueError, IndexError, AttributeError):
                pass
    else:
        label = "Predicted weekly product demand (units)" if "Demand" in task else "Predicted revenue for the target month"
        st.metric(label, f"{max(0.0, float(prediction)):,.2f}")

st.divider()
st.warning("Limitations: some relationships and business fields are absent; historical patterns can change; test scores may be weak. Validate independently before real operational use.")
'''
(APP_DIR / "app.py").write_text(app_source, encoding="utf-8")
(APP_DIR / "requirements.txt").write_text(
    "pandas\nnumpy\nscikit-learn==1.8.0\njoblib\nstreamlit\nopenpyxl\nmatplotlib\nreportlab\npython-pptx\nnbformat\n",
    encoding="utf-8",
)

finalizer = ROOT / "finalize_swiftlog_outputs.py"
if finalizer.exists():
    exec(compile(finalizer.read_text(encoding="utf-8"), str(finalizer), "exec"))
else:
    print("NOTE: finalize_swiftlog_outputs.py was not found; PDFs and presentation were not regenerated.")

print("\nSwiftLog build finished.")
print(f"Saved model: {MODELS / 'Best_Model.pkl'}")
print(f"Model comparison: {MODELS / 'Model_Comparison.xlsx'}")
print(f"Relationship audit: {PROCESSED / 'relationship_audit.csv'}")
