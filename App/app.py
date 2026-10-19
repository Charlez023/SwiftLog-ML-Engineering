import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(page_title="SwiftLog ML Decision Support", page_icon="Ã°Å¸Å¡Å¡", layout="wide")
st.title("SwiftLog Logistics: ML Decision Support")
st.caption("Educational prototype based on the supplied Week 5 datasets. Predictions support review; they do not replace operational judgement.")
MODEL_PATH=Path(__file__).resolve().parents[1]/"Models"/"Best_Model.pkl"
@st.cache_resource
def load_models(): return joblib.load(MODEL_PATH)
models=load_models()
name=st.selectbox("Choose a prediction task",list(models.keys()))
b=models[name]
st.write("Selected model:", b["algorithm"])
st.json(b["metrics"])
if b["kind"] == "classification" and b["metrics"].get("ROC_AUC", 1) < 0.60:
    st.warning("The ROC-AUC indicates weak separation between classes. Treat this output as an experimental signal, not a reliable operational decision.")
if b["kind"] == "regression" and b["metrics"].get("R2", 0) < 0:
    st.warning("The held-out R-squared is below zero, so this model performed worse than a simple test-set mean baseline. Use the estimate for demonstration only.")
if b["kind"] == "classification" and b["metrics"].get("ROC_AUC", 1) < 0.60:
    st.warning("The ROC-AUC indicates weak separation between classes. Treat this output as an experimental signal, not a reliable operational decision.")
if b["kind"] == "regression" and b["metrics"].get("R2", 0) < 0:
    st.warning("The held-out R-squared is below zero, so this model performed worse than a simple test-set mean baseline. Use the estimate for demonstration only.")
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
