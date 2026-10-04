import os
import json

from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

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


# --------------------------------------------------
# HUGGING FACE MODEL DETAILS
# --------------------------------------------------

HF_REPO_ID = "HaridossB/plant-disease-classifier"
HF_MODEL_FILENAME = "plant_disease_pred_model.h5"


# --------------------------------------------------
# DOWNLOAD / GET MODEL
# --------------------------------------------------

@st.cache_resource
def get_model():

    try:

        # ------------------------------------------
        # 1. LOCAL MODEL EXISTS
        # ------------------------------------------

        if os.path.exists(local_model_path):

            st.info("📦 Loading local model...")

            model_path = local_model_path

        # ------------------------------------------
        # 2. LOCAL MODEL DOES NOT EXIST
        #    DOWNLOAD FROM HUGGING FACE
        # ------------------------------------------

        else:

            st.info("☁️ Downloading model from Hugging Face...")

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME
            )

        # ------------------------------------------
        # 3. LOAD MODEL
        # ------------------------------------------

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as e:

        st.error("❌ Failed to load the model.")

        st.exception(e)

        st.stop()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = get_model()


# --------------------------------------------------
# LOAD CLASS INDICES
# --------------------------------------------------

if not os.path.exists(class_indices_path):

    st.error("❌ class_indices.json not found.")

    st.stop()


try:

    with open(class_indices_path, "r") as f:

        class_indices = json.load(f)

    class_indices = {
        int(k): v
        for k, v in class_indices.items()
    }

except Exception as e:

    st.error("❌ Failed to load class_indices.json.")

    st.exception(e)

    st.stop()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def load_and_preprocess_image(
    image,
    target_size=(224, 224)
):

    img = image.convert("RGB")

    img = img.resize(target_size)

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image_class(
    model,
    image,
    class_indices
):

    processed_img = load_and_preprocess_image(
        image
    )

    predictions = model.predict(
        processed_img,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        np.max(predictions[0])
    )

    predicted_label = class_indices.get(
        predicted_index,
        f"Unknown class ({predicted_index})"
    )

    return predicted_label, confidence


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image to predict the disease."
)


uploaded_image = st.file_uploader(
    "📂 Upload Leaf Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_image is not None:

    try:

        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                width=250
            )

        with col2:

            if st.button(
                "🔍 Classify",
                type="primary"
            ):

                with st.spinner(
                    "Analyzing image..."
                ):

                    label, confidence = (
                        predict_image_class(
                            model,
                            image,
                            class_indices
                        )
                    )

                st.success(
                    f"🌱 Prediction: {label}"
                )

                st.info(
                    f"📊 Confidence: "
                    f"{confidence * 100:.2f}%"
                )

    except Exception as e:

        st.error(
            "❌ Failed to process the image."
        )

        st.exception(e)