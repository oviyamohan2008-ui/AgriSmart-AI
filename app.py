import streamlit as st
from PIL import Image
from crop_model import train_demo_model, recommend
from disease_model import predict_disease

st.set_page_config(page_title="AgriSmart AI", page_icon="🌱", layout="wide")
st.title("🌱 AgriSmart AI")
st.subheader("Smart Crop Recommendation Using AI and Machine Learning")
st.write("A student project prototype for crop suggestions and plant-leaf image screening.")
crop_tab, disease_tab, about_tab = st.tabs(["🌾 Crop Recommendation", "🍃 Disease Detection", "ℹ️ About"])

with crop_tab:
    st.header("Enter soil and climate details")
    a,b,c = st.columns(3)
    n = a.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    p = a.number_input("Phosphorus (P)", 0.0, 200.0, 40.0)
    k = a.number_input("Potassium (K)", 0.0, 250.0, 40.0)
    temp = b.number_input("Temperature (°C)", 0.0, 55.0, 25.0)
    humidity = b.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = c.number_input("Soil pH", 0.0, 14.0, 6.5)
    rain = c.number_input("Rainfall (mm)", 0.0, 5000.0, 100.0)
    if st.button("Recommend crops", type="primary"):
        model = train_demo_model()
        st.write("### Top suggestions")
        for i, (name, score) in enumerate(recommend(model, [n,p,k,temp,humidity,ph,rain]), 1):
            st.write(f"**{i}. {name.title()}** — demo model score: {score:.1f}%")
        st.warning("Demo only: the included model uses synthetic sample data, not validated agricultural data. Do not use these suggestions for real farming decisions.")

with disease_tab:
    st.header("Plant disease image screening")
    st.write("Upload a clear photo of one plant leaf.")
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze image", type="primary"):
            result = predict_disease(img)
            if result["ok"]:
                st.success(f"Prediction: {result['label']} ({result['confidence']:.1f}%)")
            else:
                st.warning(result["message"])
        st.caption("Confirm suspected disease with an agricultural expert before treatment.")

with about_tab:
    st.markdown("""
    **Workflow:** Soil test → NPK/pH and climate inputs → ML crop ranking.

    **Disease workflow:** Upload leaf photo → optional trained image model → predicted class.

    **Technology:** Python, Streamlit, scikit-learn, NumPy, Pillow; TensorFlow is optional.
    """)
