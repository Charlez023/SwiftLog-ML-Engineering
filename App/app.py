import streamlit as st
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
