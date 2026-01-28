import streamlit as st
from helper import predict
st.title("🚗 Car Damage Detection")
uploaded_file = st.file_uploader("Upload your car image",type=["jpg","jpeg","png"])

if uploaded_file:
    image_path = "t_file.jpg"
    with open(image_path, "wb") as f:

        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")

