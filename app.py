import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["PClass"] = LabelEncoder().fit(["First", "Second", "Third"])
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["Sibling"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Embarked"] = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])
    return encoders

model = load_model()
encoders = build_label_encoders()

st.title("🚢 Titanic Passenger Survival Predictor")
st.markdown(
    "This is a teaching demo based on the Titanic Passenger Survival Prediction notebook. "
    "Enter the four passenger attributes and the trained Support Vector Classifier (SVC) "
    "will predict whether the passenger survived."
)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class (PClass)", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings/Spouses Aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict Survival", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked],
    })
    encoded_input = pd.DataFrame({
        "PClass": encoders["PClass"].transform(user_input["PClass"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "Sibling": encoders["Sibling"].transform(user_input["Sibling"]),
        "Embarked": encoders["Embarked"].transform(user_input["Embarked"]),
    })
    prediction = model.predict(encoded_input)[0]
    if prediction == 1:
        st.success("### ✅ SURVIVED")
    else:
        st.error("### ❌ NOT SURVIVED")
    with st.expander("See encoded feature vector"):
        st.dataframe(encoded_input, hide_index=True)

st.caption("Built for AIC354 Machine Learning Fundamentals — Assignment 2.")
