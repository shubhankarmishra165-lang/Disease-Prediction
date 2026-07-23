import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptoms = joblib.load("symptoms.pkl")
if "Disease" in symptoms:
    symptoms.remove("Disease")
display_map = {
    s: s.strip().replace("_", " ").title()
    for s in symptoms
}
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Disease Prediction System")

st.write("Select your symptoms and click Predict.")

selected_display = st.multiselect(
    "Select Symptoms",
    options=list(display_map.values())
)
reverse_map = {v: k for k, v in display_map.items()}
selected_symptoms = [reverse_map[s] for s in selected_display]
if st.button("Predict Disease"):
    with st.spinner("Predicting..."):
        if not selected_symptoms:
            st.error("🚨 Please select one or more symptoms to continue.")
        else:
            x = pd.DataFrame(0, index=[0], columns=symptoms)
            for s in selected_symptoms:
                x[s] = 1
            
            prediction = model.predict(x)

            disease = le.inverse_transform(prediction)[0]

            probability = model.predict_proba(x).max() * 100
            probs = model.predict_proba(x)[0]

            top3 = probs.argsort()[-3:][::-1]

            st.subheader("Top Predictions")

            for i in top3:
                st.write(
                    le.inverse_transform([i])[0],
                    f"{probs[i]*100:.2f}%"
                )

            st.success(f"Predicted Disease: {disease}")

            st.info(f"Confidence : {probability:.2f}%")
if st.button("Clear"):
    st.rerun()
st.warning(
        "This prediction is for educational purposes only. Please consult a qualified healthcare professional for diagnosis and treatment."
        )