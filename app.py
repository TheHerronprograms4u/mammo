import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="MAMMO AI - Breast Cancer Diagnosis",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_models():
    if os.path.exists("breast_cancer_cfDNA_pipeline.pkl") and os.path.exists("breast_cancer_cfDNA_labelencoder.pkl"):
        pipeline = joblib.load("breast_cancer_cfDNA_pipeline.pkl")
        label_encoder = joblib.load("breast_cancer_cfDNA_labelencoder.pkl")
        return pipeline, label_encoder
    return None, None

@st.cache_data
def load_data():
    if os.path.exists("Breast_cancer_cfDNA_10k.csv"):
        return pd.read_csv("Breast_cancer_cfDNA_10k.csv")
    return None

# Sidebar Navigation
st.sidebar.title("🧬 MAMMO AI")
page = st.sidebar.radio("Navigate", ["Diagnosis Dashboard", "Data Insights", "Model Performance"])

pipeline, label_encoder = load_models()
df = load_data()

if page == "Diagnosis Dashboard":
    st.title("🩺 Breast Cancer Diagnosis")
    st.markdown("Enter the patient's cfDNA markers below to predict the diagnosis.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Patient Markers")
        with st.form("input_form"):
            fragment_size = st.slider("Mean Fragment Size (bp)", 100.0, 250.0, 166.0, step=0.1)
            methylation_index = st.slider("Methylation Index", 0.0, 1.0, 0.72, step=0.01)
            mutation_load = st.slider("Mutation Load", 0.0, 5.0, 1.56, step=0.01)
            
            submit = st.form_submit_button("Run Diagnostic Prediction")

    with col2:
        if submit:
            if pipeline and label_encoder:
                input_data = pd.DataFrame([{
                    "Mean Fragment Size": fragment_size,
                    "Methylation Index": methylation_index,
                    "Mutation Load": mutation_load
                }])
                
                prediction = pipeline.predict(input_data)[0]
                probs = pipeline.predict_proba(input_data)[0]
                label = label_encoder.inverse_transform([prediction])[0]
                
                st.subheader("Diagnostic Result")
                
                # Visual Feedback
                color = "#ff4b4b" if label == "Malignant" else "#28a745"
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h2 style='color: {color};'>{label}</h2>
                        <p>Confidence: {max(probs)*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability Chart
                st.write("### Probability Distribution")
                prob_df = pd.DataFrame({
                    "Diagnosis": label_encoder.classes_,
                    "Probability (%)": probs * 100
                })
                st.bar_chart(prob_df.set_index("Diagnosis"))
                
                if label == "Malignant":
                    st.error("⚠️ The model indicates a high probability of malignant markers. Please consult with a medical professional immediately.")
                else:
                    st.success("✅ The model indicates the markers are within the benign range.")
            else:
                st.error("Model files not found. Please ensure the model is trained.")

elif page == "Data Insights":
    st.title("📊 Data Insights")
    if df is not None:
        st.write("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write("### Marker Distributions")
        marker = st.selectbox("Select Marker", ["Mean Fragment Size", "Methylation Index", "Mutation Load"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df, x=marker, hue="Diagnosis", fill=True, ax=ax, palette="coolwarm")
        plt.title(f"{marker} Distribution by Diagnosis")
        st.pyplot(fig)
        
        st.write("### Feature Correlations")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        corr = df.drop("Diagnosis", axis=1).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("Dataset not found.")

elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    st.info("The model is trained using a Random Forest Classifier with balanced class weights.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Type", "Random Forest")
        st.metric("N Estimators", "200")
        st.metric("Validation Strategy", "5-Fold Cross-Val")

    with col2:
        st.write("### Feature Importance")
        if pipeline:
            model = pipeline.named_steps["model"]
            feature_names = ["Mean Fragment Size", "Methylation Index", "Mutation Load"]
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
            st.bar_chart(feat_imp_df.set_index("Feature"))
        else:
            st.error("Model not loaded.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for MAMMO AI Research")
