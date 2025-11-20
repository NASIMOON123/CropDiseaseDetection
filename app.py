import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------- Load Model ----------
MODEL_PATH = "model/model2.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------- Load Metadata ----------
with open("class_map.json", "r") as f:
    class_map = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

with open("disease_symptoms.json", "r") as f:
    disease_symptoms = json.load(f)

with open("disease_remedies.json", "r") as f:
    disease_remedies = json.load(f)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🌿 Crop Disease Detection", layout="wide")

# Add CSS styling for card and centering
st.markdown("""
    <style>
    .image-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        text-align: center;
        width: 90%;
        margin: auto;
    }
    .st-emotion-cache-1y4p8pa {  /* Fix column vertical alignment */
        display: flex;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Crop Disease Detection System")
st.write("Upload a leaf image to detect the disease and learn more about it.")

uploaded_file = st.file_uploader(" Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Two-column layout
    col1, col2 = st.columns([1, 2], vertical_alignment="center")

    with col1:
        # Image inside a card
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=False,
            width=350
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_index = np.argmax(preds[0])
        confidence = np.max(preds[0])

        predicted_class = (
            "Unknown___Unrecognized_leaf" if confidence < 0.6 else class_map[str(class_index)]
        )

        st.subheader(f" Predicted Disease: {predicted_class}")
        # st.write(f"**Confidence:** {confidence * 100:.2f}%")

        # Display additional info
        if predicted_class in disease_info:
            st.markdown("###  Disease Information")
            st.write(disease_info[predicted_class])

            if predicted_class in disease_symptoms:
                st.markdown("###  Common Symptoms")
                for s in disease_symptoms[predicted_class]:
                    st.markdown(f"- {s}")

            if predicted_class in disease_remedies:
                st.markdown("### 🌱 Control Measures")
                for s in disease_remedies[predicted_class]:
                    st.markdown(f"- {s}")

        elif predicted_class == "Unknown___Unrecognized_leaf":
            st.warning(" The uploaded leaf does not match any crop in our database.")
        else:
            st.warning("No additional information found for this disease.")
