"""SwiftLog Week 5 ML project builder. Run from project root: python run_swiftlog_project.py"""
from pathlib import Path
import json, warnings, math, sys, traceback
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

ROOT = Path.cwd()
DATA_DIRS = [ROOT/'Data'/'Original_Datasets', ROOT/'Data']
OUT = ROOT
for folder in ['Data/Original_Datasets','Data/Processed_Datasets','Notebook','Models','App/assets','Reports','Presentation','References']:
    (ROOT/folder).mkdir(parents=True, exist_ok=True)

def data_file(name):
    for d in DATA_DIRS:
        p=d/name
        if p.exists(): return p
    raise FileNotFoundError(f"Could not find {name}. Put CSV files in Data/ or Data/Original_Datasets/.")

def load(name):
    df=pd.read_csv(data_file(name))
    df.columns=[str(c).strip() for c in df.columns]
    return df

def date_parts(df, col):
    d=pd.to_datetime(df[col], errors='coerce')
    df[col+'_Year']=d.dt.year
    df[col+'_Month']=d.dt.month
    df[col+'_Day']=d.dt.day
    df[col+'_Weekday']=d.dt.dayofweek
    return df

def clean_text(df):
    for c in df.select_dtypes(include='object').columns:
        df[c]=df[c].astype('string').str.strip().replace({'':pd.NA,'nan':pd.NA,'None':pd.NA})
    return df

def prep(X):
    X=X.copy()
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]): X[c]=X[c].astype(str)
    cats=X.select_dtypes(include=['object','string','category']).columns.tolist()
    nums=[c for c in X.columns if c not in cats]
    transformers=[]
    if nums: transformers.append(('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
    if cats: transformers.append(('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats))
    return ColumnTransformer(transformers, remainder='drop'), cats, nums

# Load and standardize all source tables
files=['SwiftLog_Customers.csv','SwiftLog_Delivery_Performance.csv','SwiftLog_Employees.csv','SwiftLog_Inventory.csv','SwiftLog_Orders.csv','SwiftLog_Order_Items.csv','SwiftLog_Products.csv','SwiftLog_Returns.csv','SwiftLog_Shipments.csv','SwiftLog_Suppliers.csv','SwiftLog_Warehouses.csv']
tables={}
for fn in files:
    d=clean_text(load(fn))
    tables[fn.replace('SwiftLog_','').replace('.csv','')]=d
customers=tables['Customers']; delivery=tables['Delivery_Performance']; employees=tables['Employees']; inventory=tables['Inventory']; orders=tables['Orders']; items=tables['Order_Items']; products=tables['Products']; returns=tables['Returns']; shipments=tables['Shipments']; suppliers=tables['Suppliers']; warehouses=tables['Warehouses']

# Data quality and preprocessing audit
quality=[]
for key,df in tables.items():
    quality.append({'Dataset':key,'Rows':len(df),'Columns':len(df.columns),'Duplicate rows':int(df.duplicated().sum()),'Missing cells':int(df.isna().sum().sum()),'Missing columns':', '.join(df.columns[df.isna().any()].tolist())})
quality_df=pd.DataFrame(quality)
quality_df.to_csv(ROOT/'Data/Processed_Datasets/data_quality_summary.csv',index=False)
for key,df in tables.items():
    df.drop_duplicates(inplace=True)
    df.to_csv(ROOT/f'Data/Processed_Datasets/{key.lower()}_clean.csv',index=False)

# Shared joined tables
orders2=orders.copy()
if 'OrderDate' in orders2: orders2=date_parts(orders2,'OrderDate')
order_items=items.merge(products,on='ProductID',how='left',validate='many_to_one')
order_items['LineRevenue']=pd.to_numeric(order_items['Quantity'],errors='coerce').fillna(0)*pd.to_numeric(order_items['UnitPrice'],errors='coerce').fillna(0)
order_items.to_csv(ROOT/'Data/Processed_Datasets/order_items_enriched.csv',index=False)

# Problem 1: shipment delay classification (delay > 0), avoid leakage by excluding DelayDays and delivery Status from predictors
ship=shipments.merge(orders2,on='OrderID',how='left',suffixes=('','_Order'))
ship=ship.merge(warehouses,on='WarehouseID',how='left')
ship=ship.merge(customers[['CustomerID','City']],on='CustomerID',how='left')
ship['DelayDays']=pd.to_numeric(ship['DelayDays'],errors='coerce')
ship['Target_Delayed']=(ship['DelayDays'].fillna(0)>0).astype(int)
delay_features=[c for c in ['WarehouseID','Location','City','Status','OrderDate_Year','OrderDate_Month','OrderDate_Day','OrderDate_Weekday'] if c in ship.columns]
# Status is an order status potentially known at prediction time but may be post-event; exclude it to reduce leakage.
delay_features=[c for c in delay_features if c!='Status']
delay_X=ship[delay_features].copy(); delay_y=ship['Target_Delayed']
delay_X.assign(Target_Delayed=delay_y).to_csv(ROOT/'Data/Processed_Datasets/delivery_delay_features.csv',index=False)

# Problem 2: order-level product return classification
ret_orders=set(returns['OrderID'].dropna().astype(str))
order_feat=orders2.merge(customers[['CustomerID','City']],on='CustomerID',how='left')
item_agg=order_items.groupby('OrderID').agg(ItemCount=('OrderItemID','count'),TotalQuantity=('Quantity','sum'),DistinctProducts=('ProductID','nunique'),TotalRevenue=('LineRevenue','sum'),MeanUnitPrice=('UnitPrice','mean')).reset_index()
order_feat=order_feat.merge(item_agg,on='OrderID',how='left')
order_feat['Target_Returned']=order_feat['OrderID'].astype(str).isin(ret_orders).astype(int)
ret_features=[c for c in ['City','OrderDate_Year','OrderDate_Month','OrderDate_Day','OrderDate_Weekday','ItemCount','TotalQuantity','DistinctProducts','TotalRevenue','MeanUnitPrice'] if c in order_feat.columns]
return_X=order_feat[ret_features].copy(); return_y=order_feat['Target_Returned']
return_X.assign(Target_Returned=return_y).to_csv(ROOT/'Data/Processed_Datasets/product_return_features.csv',index=False)

# Problem 3: weekly product demand forecasting with only prior-week information
item_date=items.merge(orders2[['OrderID','OrderDate']],on='OrderID',how='left').merge(products,on='ProductID',how='left')
item_date['OrderDate']=pd.to_datetime(item_date['OrderDate'],errors='coerce')
item_date['Quantity']=pd.to_numeric(item_date['Quantity'],errors='coerce').fillna(0)
item_date=item_date.dropna(subset=['OrderDate'])
item_date['WeekStart']=item_date['OrderDate'].dt.to_period('W-SUN').dt.start_time
daily=item_date.groupby(['ProductID','WeekStart'],as_index=False).agg(DemandQuantity=('Quantity','sum'))
product_info=products[['ProductID','Category','UnitPrice']].drop_duplicates('ProductID')
all_weeks=pd.date_range(daily['WeekStart'].min(),daily['WeekStart'].max(),freq='7D')
product_grid=pd.MultiIndex.from_product([product_info['ProductID'].tolist(),all_weeks],names=['ProductID','WeekStart']).to_frame(index=False)
demand=product_grid.merge(daily,on=['ProductID','WeekStart'],how='left').merge(product_info,on='ProductID',how='left')
demand['DemandQuantity']=demand['DemandQuantity'].fillna(0)
demand=demand.sort_values(['ProductID','WeekStart']).reset_index(drop=True)
demand['Lag_1_Week']=demand.groupby('ProductID')['DemandQuantity'].shift(1)
demand['Lag_4_Weeks']=demand.groupby('ProductID')['DemandQuantity'].shift(4)
demand['Rolling_Mean_4_Weeks']=demand.groupby('ProductID')['DemandQuantity'].transform(lambda x:x.shift(1).rolling(4,min_periods=4).mean())
demand['Year']=demand['WeekStart'].dt.year
demand['Month']=demand['WeekStart'].dt.month
demand['DayOfWeek']=demand['WeekStart'].dt.dayofweek
demand=demand.dropna(subset=['Lag_1_Week','Lag_4_Weeks','Rolling_Mean_4_Weeks']).sort_values(['WeekStart','ProductID']).reset_index(drop=True)
demand_features=['ProductID','Category','UnitPrice','Year','Month','DayOfWeek','Lag_1_Week','Lag_4_Weeks','Rolling_Mean_4_Weeks']
demand_X=demand[demand_features].copy(); demand_y=demand['DemandQuantity'].astype(float); demand_split_dates=demand['WeekStart'].copy()
demand_X.assign(DemandQuantity=demand_y, WeekStart=demand_split_dates.astype(str)).to_csv(ROOT/'Data/Processed_Datasets/demand_forecasting_features.csv',index=False)

# Problem 4: predict each customer's next-month value from the preceding three months
order_month=orders2.copy()
order_month['OrderDate']=pd.to_datetime(order_month['OrderDate'],errors='coerce')
order_month=order_month.dropna(subset=['OrderDate'])
order_month['MonthStart']=order_month['OrderDate'].dt.to_period('M').dt.to_timestamp()
item_month=order_items.merge(order_month[['OrderID','CustomerID','MonthStart']],on='OrderID',how='left')
item_month['LineRevenue']=pd.to_numeric(item_month['Quantity'],errors='coerce').fillna(0)*pd.to_numeric(item_month['UnitPrice'],errors='coerce').fillna(0)
monthly_items=item_month.groupby(['CustomerID','MonthStart']).agg(MonthRevenue=('LineRevenue','sum'),MonthQuantity=('Quantity','sum'),MonthItems=('OrderItemID','count'),MonthAvgUnitPrice=('UnitPrice','mean')).reset_index()
monthly_orders=order_month.groupby(['CustomerID','MonthStart']).agg(MonthOrders=('OrderID','nunique')).reset_index()
monthly=monthly_items.merge(monthly_orders,on=['CustomerID','MonthStart'],how='outer')
month_range=pd.date_range(order_month['MonthStart'].min(),order_month['MonthStart'].max(),freq='MS')
customer_month_grid=pd.MultiIndex.from_product([customers['CustomerID'].dropna().unique().tolist(),month_range],names=['CustomerID','MonthStart']).to_frame(index=False)
monthly=customer_month_grid.merge(monthly,on=['CustomerID','MonthStart'],how='left')
for c in ['MonthRevenue','MonthQuantity','MonthItems','MonthOrders']:
    monthly[c]=pd.to_numeric(monthly[c],errors='coerce').fillna(0)
monthly=monthly.sort_values(['CustomerID','MonthStart']).reset_index(drop=True)
monthly['Previous_3M_Revenue']=monthly.groupby('CustomerID')['MonthRevenue'].transform(lambda x:x.shift(1).rolling(3,min_periods=3).sum())
monthly['Previous_3M_Order_Count']=monthly.groupby('CustomerID')['MonthOrders'].transform(lambda x:x.shift(1).rolling(3,min_periods=3).sum())
monthly['Previous_3M_Quantity']=monthly.groupby('CustomerID')['MonthQuantity'].transform(lambda x:x.shift(1).rolling(3,min_periods=3).sum())
monthly['Previous_3M_Item_Count']=monthly.groupby('CustomerID')['MonthItems'].transform(lambda x:x.shift(1).rolling(3,min_periods=3).sum())
monthly['Previous_3M_Avg_Unit_Price']=monthly.groupby('CustomerID')['MonthAvgUnitPrice'].transform(lambda x:x.shift(1).rolling(3,min_periods=3).mean())
monthly=monthly.merge(customers[['CustomerID','City']],on='CustomerID',how='left')
monthly=monthly.dropna(subset=['Previous_3M_Revenue','Previous_3M_Order_Count','Previous_3M_Quantity','Previous_3M_Item_Count']).sort_values(['MonthStart','CustomerID']).reset_index(drop=True)
value_features=['City','Previous_3M_Revenue','Previous_3M_Order_Count','Previous_3M_Quantity','Previous_3M_Item_Count','Previous_3M_Avg_Unit_Price']
value_X=monthly[value_features].copy(); value_y=pd.to_numeric(monthly['MonthRevenue'],errors='coerce').fillna(0).astype(float); value_split_dates=monthly['MonthStart'].copy()
monthly.to_csv(ROOT/'Data/Processed_Datasets/customer_value_features.csv',index=False)

problems={
 'Delivery Delay Classification':(delay_X,delay_y,'classification',None),
 'Product Return Classification':(return_X,return_y,'classification',None),
 'Product Demand Forecasting':(demand_X,demand_y,'regression',demand_split_dates),
 'Customer Value Prediction (Next Month)':(value_X,value_y,'regression',value_split_dates)}
results=[]; bundles={}; best_global=None; best_global_score=-np.inf; all_importances=[]
for pname,(X,y,kind,split_dates) in problems.items():
    valid=y.notna(); X=X.loc[valid].reset_index(drop=True); y=y.loc[valid].reset_index(drop=True)
    if split_dates is not None: split_dates=split_dates.loc[valid].reset_index(drop=True)
    if len(X)<30: raise ValueError(f'{pname} has too few rows ({len(X)}).')
    if kind=='classification' and y.nunique()<2: raise ValueError(f'{pname} target has fewer than two classes.')

    # Create separate training, validation, and test sets.
    # Time-based tasks keep later dates out of training to reduce temporal leakage.
    if split_dates is not None:
        dates = pd.to_datetime(split_dates, errors='coerce')
        unique_dates = np.array(sorted(dates.dropna().unique()))
        if len(unique_dates) < 3:
            raise ValueError(f'{pname} needs at least 3 distinct dates for train/validation/test splitting.')
        train_end = max(1, int(len(unique_dates) * 0.6))
        val_end = max(train_end + 1, int(len(unique_dates) * 0.8))
        val_end = min(val_end, len(unique_dates) - 1)
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:]
        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)
        test_mask = dates.isin(test_dates)
        Xtr, ytr = X.loc[train_mask].reset_index(drop=True), y.loc[train_mask].reset_index(drop=True)
        Xval, yval = X.loc[val_mask].reset_index(drop=True), y.loc[val_mask].reset_index(drop=True)
        Xte, yte = X.loc[test_mask].reset_index(drop=True), y.loc[test_mask].reset_index(drop=True)
    else:
        strat = y if kind=='classification' and y.value_counts().min() >= 2 else None
        Xtemp, Xte, ytemp, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
        strat_temp = ytemp if kind=='classification' and ytemp.value_counts().min() >= 2 else None
        Xtr, Xval, ytr, yval = train_test_split(
            Xtemp, ytemp, test_size=0.25, random_state=42, stratify=strat_temp
        )

    if len(Xtr)==0 or len(Xval)==0 or len(Xte)==0:
        raise ValueError(f'{pname} produced an empty train, validation, or test split.')

    if kind=='classification':
        algos={'Logistic Regression':LogisticRegression(max_iter=1200,class_weight='balanced'),'Decision Tree':DecisionTreeClassifier(max_depth=10,class_weight='balanced',random_state=42),'Random Forest':RandomForestClassifier(n_estimators=60,max_depth=10,class_weight='balanced_subsample',random_state=42,n_jobs=1)}
    else:
        algos={'Linear Regression':LinearRegression(),'Decision Tree Regressor':DecisionTreeRegressor(max_depth=10,random_state=42),'Random Forest Regressor':RandomForestRegressor(n_estimators=60,max_depth=10,random_state=42,n_jobs=1)}

    fitted=[]
    for aname,algo in algos.items():
        # Give each candidate its own preprocessing transformer to avoid shared fitted state.
        transformer, cats, nums = prep(Xtr)
        pipe=Pipeline([('preprocessor',transformer),('model',algo)])
        pipe.fit(Xtr,ytr)
        pred_val=pipe.predict(Xval)
        row={'Problem':pname,'Algorithm':aname,'Rows':len(X),'Features':X.shape[1],'Selected':False,'Tuned':False}
        if kind=='classification':
            row.update({'Accuracy':accuracy_score(yval,pred_val),'Precision':precision_score(yval,pred_val,zero_division=0),'Recall':recall_score(yval,pred_val,zero_division=0),'F1':f1_score(yval,pred_val,zero_division=0),'ROC_AUC':np.nan})
            if hasattr(pipe,'predict_proba') and yval.nunique()==2:
                try: row['ROC_AUC']=roc_auc_score(yval,pipe.predict_proba(Xval)[:,1])
                except Exception: pass
            score=row['F1']
        else:
            rmse=math.sqrt(mean_squared_error(yval,pred_val)); row.update({'MAE':mean_absolute_error(yval,pred_val),'RMSE':rmse,'R2':r2_score(yval,pred_val)})
            score=-row['RMSE']
        results.append(row)
        fitted.append((score,aname,pipe,row))

    fitted.sort(key=lambda z:z[0],reverse=True)
    best_score,best_name,best_pipe,best_row=fitted[0]

    # Tune the selected algorithm using training data only; choose it using validation performance.
    try:
        if kind=='classification' and best_name=='Random Forest':
            search=GridSearchCV(best_pipe,{'model__n_estimators':[40,60],'model__max_depth':[8,10]},cv=(TimeSeriesSplit(n_splits=2) if split_dates is not None else 3),scoring='f1',n_jobs=1)
        elif kind=='regression' and best_name=='Random Forest Regressor':
            search=GridSearchCV(best_pipe,{'model__n_estimators':[40,60],'model__max_depth':[8,10]},cv=(TimeSeriesSplit(n_splits=2) if split_dates is not None else 3),scoring='neg_root_mean_squared_error',n_jobs=1)
        elif kind=='classification' and best_name=='Decision Tree':
            search=GridSearchCV(best_pipe,{'model__max_depth':[5,10,15],'model__min_samples_leaf':[1,3]},cv=(TimeSeriesSplit(n_splits=2) if split_dates is not None else 3),scoring='f1',n_jobs=1)
        elif kind=='regression' and best_name=='Decision Tree Regressor':
            search=GridSearchCV(best_pipe,{'model__max_depth':[5,10,15],'model__min_samples_leaf':[1,3]},cv=(TimeSeriesSplit(n_splits=2) if split_dates is not None else 3),scoring='neg_root_mean_squared_error',n_jobs=1)
        else:
            search=None
        if search:
            search.fit(Xtr,ytr)
            tuned=search.best_estimator_
            tuned_val=tuned.predict(Xval)
            if kind=='classification':
                tuned_score=f1_score(yval,tuned_val,zero_division=0)
            else:
                tuned_score=-math.sqrt(mean_squared_error(yval,tuned_val))
            if tuned_score >= best_score:
                best_pipe=tuned
                best_name += ' (tuned)'
                best_score=tuned_score
                best_row['Tuned']=True
    except Exception as e:
        print(f'Tuning note for {pname}: {e}')

    # Refit the selected pipeline on training + validation data, then evaluate once on the untouched test set.
    Xfinal=pd.concat([Xtr,Xval],ignore_index=True)
    yfinal=pd.concat([ytr,yval],ignore_index=True)
    best_pipe.fit(Xfinal,yfinal)
    pred_test=best_pipe.predict(Xte)

    if kind=='classification':
        metric={'Accuracy':accuracy_score(yte,pred_test),'Precision':precision_score(yte,pred_test,zero_division=0),'Recall':recall_score(yte,pred_test,zero_division=0),'F1':f1_score(yte,pred_test,zero_division=0)}
        if hasattr(best_pipe,'predict_proba') and yte.nunique()==2:
            try: metric['ROC_AUC']=roc_auc_score(yte,best_pipe.predict_proba(Xte)[:,1])
            except Exception: metric['ROC_AUC']=np.nan
        fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(yte,pred_test)).plot(ax=ax,colorbar=False); ax.set_title(pname+': Test Confusion Matrix'); fig.tight_layout(); fig.savefig(ROOT/f'Models/{pname.lower().replace(" ","_")}_confusion_matrix.png',dpi=150); plt.close(fig)
    else:
        metric={'MAE':mean_absolute_error(yte,pred_test),'RMSE':math.sqrt(mean_squared_error(yte,pred_test)),'R2':r2_score(yte,pred_test)}

    best_row['Algorithm']=best_name
    best_row['Selected']=True
    # Keep candidate comparison metrics (validation) and record final test metrics separately.
    best_row.update({f'Test_{key}': value for key,value in metric.items()})
    bundles[pname]={'model':best_pipe,'features':list(X.columns),'defaults':{c:(float(pd.to_numeric(X[c],errors='coerce').median()) if pd.api.types.is_numeric_dtype(X[c]) else (str(X[c].mode(dropna=True).iloc[0]) if not X[c].mode(dropna=True).empty else 'Unknown')) for c in X.columns},'types':{c:('numeric' if pd.api.types.is_numeric_dtype(X[c]) else 'categorical') for c in X.columns},'kind':kind,'metrics':metric,'algorithm':best_name}

    # Estimate feature importance on held-out test rows; this is descriptive, not used for model selection.
    try:
        if len(Xte)>1000:
            sample_idx=Xte.sample(n=1000,random_state=42).index
            Ximp=Xte.loc[sample_idx]; yimp=yte.loc[sample_idx]
        else:
            Ximp=Xte; yimp=yte
        pi=permutation_importance(best_pipe,Ximp,yimp,n_repeats=2,random_state=42,n_jobs=1,scoring='f1' if kind=='classification' else 'neg_root_mean_squared_error')
        for c,v in zip(X.columns,pi.importances_mean): all_importances.append({'Problem':pname,'Feature':c,'Importance':float(v)})
    except Exception as e:
        print(f'Feature-importance note for {pname}: {e}')
    print(f'Completed: {pname} | best={best_name} | test metrics={metric}')

# Save artefacts
joblib.dump(bundles,ROOT/'Models/Best_Model.pkl')
res=pd.DataFrame(results)
res.to_excel(ROOT/'Models/Model_Comparison.xlsx',index=False)
imp=pd.DataFrame(all_importances)
if not imp.empty:
    imp.to_csv(ROOT/'Models/Permutation_Importance.csv',index=False)
    top=imp.sort_values('Importance',ascending=False).head(20).sort_values('Importance')
    fig,ax=plt.subplots(figsize=(9,6)); ax.barh(top['Problem']+': '+top['Feature'],top['Importance']); ax.set_title('Permutation Feature Importance (top 20)'); ax.set_xlabel('Importance score'); fig.tight_layout(); fig.savefig(ROOT/'Models/Feature_Importance.png',dpi=160); plt.close(fig)
else:
    plt.figure(figsize=(8,4)); plt.text(.1,.5,'Permutation importance unavailable for this run'); plt.axis('off'); plt.savefig(ROOT/'Models/Feature_Importance.png'); plt.close()
# General EDA visualizations
fig,ax=plt.subplots(figsize=(8,4)); ax.bar(quality_df['Dataset'],quality_df['Rows']); ax.set_ylabel('Rows'); ax.set_title('Dataset row counts'); ax.tick_params(axis='x',rotation=45); fig.tight_layout(); fig.savefig(ROOT/'Reports/dataset_row_counts.png',dpi=150); plt.close(fig)
# Field-level dictionary for engineered features and targets
feature_dictionary = [
 {'Field':'Target_Delayed','Task':'Delivery Delay Classification','Definition':'1 when DelayDays > 0; otherwise 0.','Business meaning':'Historical delay label; DelayDays itself is excluded from model inputs.'},
 {'Field':'Target_Returned','Task':'Product Return Classification','Definition':'1 when OrderID appears in Returns; otherwise 0.','Business meaning':'Historical recorded-return label.'},
 {'Field':'Lag_1_Week','Task':'Product Demand Forecasting','Definition':'Product quantity in the previous week.','Business meaning':'Recent demand signal available before the prediction date.'},
 {'Field':'Lag_4_Weeks','Task':'Product Demand Forecasting','Definition':'Product quantity four weeks before the prediction week.','Business meaning':'Weekly demand signal available before the prediction date.'},
 {'Field':'Rolling_Mean_4_Weeks','Task':'Product Demand Forecasting','Definition':'Mean weekly product demand over the preceding four weeks, excluding the prediction week.','Business meaning':'Recent baseline demand; calculated without using the target day.'},
 {'Field':'Previous_3M_Revenue','Task':'Customer Value Prediction (Next Month)','Definition':'Customer revenue summed over the three months before the target month.','Business meaning':'Prior spending history used to estimate next-month revenue.'},
 {'Field':'Previous_3M_Order_Count','Task':'Customer Value Prediction (Next Month)','Definition':'Customer order count over the three months before the target month.','Business meaning':'Recent purchasing frequency.'},
 {'Field':'Previous_3M_Quantity','Task':'Customer Value Prediction (Next Month)','Definition':'Customer item quantity over the three months before the target month.','Business meaning':'Recent purchase volume.'},
 {'Field':'Previous_3M_Item_Count','Task':'Customer Value Prediction (Next Month)','Definition':'Number of order-item records over the three months before the target month.','Business meaning':'Recent breadth of purchasing activity.'},
 {'Field':'Previous_3M_Avg_Unit_Price','Task':'Customer Value Prediction (Next Month)','Definition':'Mean unit price observed in the preceding three months.','Business meaning':'Recent price level of purchased products.'},
 {'Field':'DemandQuantity','Task':'Product Demand Forecasting','Definition':'Total units purchased for a product in a week; zero when no purchase was recorded.','Business meaning':'Weekly demand target.'},
 {'Field':'MonthRevenue','Task':'Customer Value Prediction (Next Month)','Definition':'Customer item revenue in the target month.','Business meaning':'Next-month revenue target.'}
]
pd.DataFrame(feature_dictionary).to_csv(ROOT/'Data/Processed_Datasets/Feature_Data_Dictionary.csv',index=False)

# Feature engineered combined dataset (documented as modelling-ready consolidated samples)
feature_file=ROOT/'Data/Feature_Engineered_Dataset.csv'
order_feat.to_csv(feature_file,index=False)
# This file is the order-level engineered dataset used for return prediction; other task-specific features are saved separately.

# Streamlit app with user inputs for saved models
app=r'''import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(page_title="SwiftLog ML Decision Support", page_icon="📦", layout="wide")
st.title("SwiftLog Logistics: ML Decision Support")
st.caption("Educational prototype based on the supplied Week 5 datasets. Predictions support review; they do not replace operational judgement.")
MODEL_PATH=Path(__file__).resolve().parents[1]/"Models"/"Best_Model.pkl"
@st.cache_resource
def load_models(): return joblib.load(MODEL_PATH)
models=load_models()
name=st.selectbox("Choose a prediction task",list(models.keys()))
b=models[name]
st.write("Selected model:", b["algorithm"])
st.write("Held-out test metrics:", b["metrics"])
with st.form("predict"):
    vals={}
    cols=st.columns(2)
    for i,c in enumerate(b["features"]):
        with cols[i%2]:
            default=b["defaults"].get(c,0)
            if b["types"].get(c)=="numeric":
                try: vals[c]=st.number_input(c,value=float(default))
                except: vals[c]=0.0
            else:
                vals[c]=st.text_input(c,value=str(default))
    submitted=st.form_submit_button("Generate prediction")
if submitted:
    X=pd.DataFrame([vals],columns=b["features"])
    model=b["model"]
    pred=model.predict(X)[0]
    if b["kind"]=="classification":
        st.subheader("Prediction")
        positive_label = "Delayed" if "Delivery Delay" in name else "Returned"
        negative_label = "Not delayed" if "Delivery Delay" in name else "Not returned"
        st.success(positive_label if int(pred)==1 else negative_label)
        if hasattr(model,"predict_proba"):
            try: st.metric("Model probability for positive class",f"{model.predict_proba(X)[0][1]:.1%}")
            except: pass
    else:
        st.subheader("Estimated value")
        metric_label = "Predicted weekly product demand (units)" if "Demand" in name else "Predicted next-month customer revenue"
        st.metric(metric_label,f"{max(0.0, float(pred)):,.2f}")
st.divider()
st.caption("Model limitations: synthetic/limited fields, historical patterns may change, and predictions require validation before production use.")
'''
(ROOT/'App/app.py').write_text(app,encoding='utf-8')
(ROOT/'App/requirements.txt').write_text('pandas\nnumpy\nscikit-learn==1.8.0\njoblib\nstreamlit\nopenpyxl\nmatplotlib\nreportlab\npython-pptx\nnbformat\n',encoding='utf-8')

# Refresh documentation, notebook, app warnings, and presentation from the saved outputs.
finalizer = ROOT / 'finalize_swiftlog_outputs.py'
exec(compile(finalizer.read_text(encoding='utf-8'), str(finalizer), 'exec'))
