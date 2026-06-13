import streamlit as st
import requests
import pandas as pd
import json
import os

st.set_page_config(
    page_title="Sepsis Prediction App",
    page_icon="🏥",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "sk-sepsis-2024-dev-key")
HEADERS = {"X-API-Key": API_KEY}

st.title("🏥 Sepsis Prediction System")
st.markdown("Machine learning powered sepsis risk assessment with AI explanations")

with st.sidebar:
    st.header("Settings")
    st.markdown(f"**API URL:** `{API_URL}`")
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This app uses a trained ML model to predict sepsis risk based on patient features.
    """)
    st.markdown("---")
    if st.button("Check API Health"):
        try:
            resp = requests.get(f"{API_URL}/health", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ API is healthy\nModel: {data.get('model_type', 'Unknown')}")
            else:
                st.error(f"❌ API returned {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

with tab1:
    st.header("Single Patient Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        PRG = st.number_input("Plasma Glucose (PRG)", min_value=0.0, value=1.0)
        PL = st.number_input("Blood Pressure - PL", min_value=0.0, value=85.0)
        PR = st.number_input("Diastolic BP - PR", min_value=0.0, value=66.0)
    with col2:
        SK = st.number_input("Skin Thickness (SK)", min_value=0.0, value=29.0)
        TS = st.number_input("Insulin (TS)", min_value=0.0, value=0.0)
        M11 = st.number_input("BMI (M11)", min_value=0.0, value=26.6)
    with col3:
        BD2 = st.number_input("Diabetes Pedigree (BD2)", min_value=0.0, value=0.351)
        Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=31.0)
        Insurance = st.selectbox("Insurance", [0, 1])

    use_explanation = st.checkbox("Include Gemini AI Explanation", value=False)

    if st.button("Predict", type="primary"):
        payload = {
            "PRG": PRG, "PL": PL, "PR": PR, "SK": SK,
            "TS": TS, "M11": M11, "BD2": BD2,
            "Age": Age, "Insurance": Insurance,
        }
        try:
            resp = requests.post(
                f"{API_URL}/predict?explain={'true' if use_explanation else 'false'}",
                json=payload,
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                pred = data["prediction"]
                prob = data["probability"]
                conf = data["confidence"]
                color = "🔴" if pred == "Positive" else "🟢"
                st.markdown(f"### {color} Prediction: **{pred}**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Probability", f"{prob:.2%}")
                with col_b:
                    st.metric("Confidence", conf)
                st.progress(prob if pred == "Positive" else 1 - prob)
                if data.get("explanation"):
                    st.markdown("---")
                    st.subheader("🤖 AI Explanation")
                    st.info(data["explanation"])
            else:
                st.error(f"Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab2:
    st.header("Batch Prediction (CSV Upload)")
    st.markdown("Upload a CSV file with patient data to get predictions for all rows.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.markdown("**Preview of uploaded data:**")
            st.dataframe(df_preview.head())
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Could not preview CSV: {e}")
        if st.button("Run Batch Prediction", type="primary"):
            try:
                uploaded_file.seek(0)
                resp = requests.post(
                    f"{API_URL}/predict-batch",
                    files={"file": ("upload.csv", uploaded_file, "text/csv")},
                    headers=HEADERS,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ Predicted {data['count']} patients")
                    results_df = pd.DataFrame(data["results"])
                    st.dataframe(results_df, use_container_width=True)
                    csv_download = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv_download,
                        "predictions.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab3:
    st.header("Model Information")
    try:
        resp = requests.get(f"{API_URL}/model-info", headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Type", data.get("model_type", "Unknown"))
            with col2:
                st.metric("Accuracy", f"{data.get('accuracy', 0):.2%}")
            with col3:
                st.metric("F1 Score", f"{data.get('f1_score', 0):.2%}")
            with col4:
                st.metric("ROC AUC", f"{data.get('roc_auc', 0):.2%}")
            st.markdown("**Features used:**")
            st.write(data.get("features", []))
        else:
            st.error(f"Could not load model info: {resp.status_code}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
