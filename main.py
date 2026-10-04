import os
import json

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)


# ============================================================
# PROJECT PATHS
# ============================================================

working_dir = os.path.dirname(os.path.abspath(__file__))

class_indices_path = os.path.join(
    working_dir,
    "class_indices.json"
)

local_model_path = os.path.join(
    working_dir,
    "trained_model",
    "plant_disease_pred_model.h5"
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_REPO_ID = "HaridossB/plant-disease-classifier"

HF_MODEL_FILENAME = "plant_disease_pred_model.h5"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    try:

        # ----------------------------------------------------
        # OPTION 1:
        # Local model exists
        # ----------------------------------------------------

        if os.path.exists(local_model_path):

            st.info("📦 Loading model from local project...")

            model_path = local_model_path

        # ----------------------------------------------------
        # OPTION 2:
        # Local model does NOT exist
        # Download from Hugging Face
        # ----------------------------------------------------

        else:

            st.info(
                "☁️ Local model not found. "
                "Downloading model from Hugging Face..."
            )

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME
            )

        # ----------------------------------------------------
        # Load TensorFlow / Keras model
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as e:

        st.error("❌ Failed to load the plant disease model.")

        st.exception(e)

        st.stop()


# Load model
model = get_model()


# ============================================================
# LOAD CLASS INDICES
# ============================================================

if not os.path.exists(class_indices_path):

    st.error(
        "❌ class_indices.json was not found."
    )

    st.stop()


try:

    with open(class_indices_path, "r") as file:

        class_indices = json.load(file)

    # JSON keys are strings.
    # Convert them into integers.

    class_indices = {
        int(key): value
        for key, value in class_indices.items()
    }

except Exception as e:

    st.error(
        "❌ Failed to load class_indices.json."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_and_preprocess_image(
    image,
    target_size=(224, 224)
):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(target_size)

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Convert to float32
    image_array = image_array.astype("float32")

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image_class(
    model,
    image,
    class_indices
):

    processed_image = load_and_preprocess_image(
        image
    )

    # Make prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Get prediction for first image
    prediction = predictions[0]

    # Get class index
    predicted_index = int(
        np.argmax(prediction)
    )

    # Get confidence
    confidence = float(
        np.max(prediction)
    )

    # Get class name
    predicted_label = class_indices.get(
        predicted_index,
        f"Unknown Class ({predicted_index})"
    )

    return predicted_label, confidence


# ============================================================
# APPLICATION UI
# ============================================================

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image to identify "
    "the predicted disease."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_image = st.file_uploader(
    "📂 Upload Leaf Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_image is not None:

    try:

        image = Image.open(
            uploaded_image
        )

        # ----------------------------------------------------
        # Display uploaded image
        # ----------------------------------------------------

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Plant Leaf",
            width=300
        )

        # ----------------------------------------------------
        # Classification button
        # ----------------------------------------------------

        if st.button(
            "🔍 Classify",
            type="primary"
        ):

            with st.spinner(
                "🔄 Analyzing the plant leaf..."
            ):

                label, confidence = predict_image_class(
                    model,
                    image,
                    class_indices
                )

            # ------------------------------------------------
            # Prediction result
            # ------------------------------------------------

            st.success(
                f"🌱 Prediction: {label}"
            )

            st.info(
                f"📊 Confidence: {confidence * 100:.2f}%"
            )

    except Exception as e:

        st.error(
            "❌ Failed to process the uploaded image."
        )

        st.exception(e)