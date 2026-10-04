import os
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)


# ============================================================
# PATH SETUP
# ============================================================

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    WORKING_DIR,
    "trained_model",
    "plant_disease_pred_model.h5"
)

CLASS_INDICES_PATH = os.path.join(
    WORKING_DIR,
    "class_indices.json"
)


# ============================================================
# MODEL LOADING
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )
    st.stop()


@st.cache_resource
def load_trained_model():
    """
    Load the trained TensorFlow/Keras model only once.

    compile=False is used because this application is
    performing inference only and does not need the
    original optimizer/training configuration.
    """

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


try:
    model = load_trained_model()

except Exception as e:
    st.error("❌ Failed to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# CLASS LABEL LOADING
# ============================================================

if not os.path.exists(CLASS_INDICES_PATH):
    st.error(
        "❌ class_indices.json was not found.\n\n"
        f"Expected location:\n{CLASS_INDICES_PATH}"
    )
    st.stop()


try:

    with open(
        CLASS_INDICES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_indices = json.load(file)

    # JSON object keys are strings.
    # Convert them to integers because model prediction
    # returns an integer class index.

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
    """
    Prepare uploaded image for the trained model.

    Steps:
    1. Convert image to RGB
    2. Resize to 224x224
    3. Convert to NumPy array
    4. Convert to float32
    5. Normalize pixel values to 0-1
    6. Add batch dimension
    """

    # Ensure exactly 3 channels
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(target_size)

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Convert datatype
    image_array = image_array.astype(
        np.float32
    )

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    #
    # Before:
    # (224, 224, 3)
    #
    # After:
    # (1, 224, 224, 3)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image_class(
    trained_model,
    image,
    labels
):
    """
    Predict the disease class of an uploaded image.
    """

    # Preprocess image
    processed_image = load_and_preprocess_image(
        image
    )

    # Run model prediction
    predictions = trained_model.predict(
        processed_image,
        verbose=0
    )

    # Find class with highest probability
    predicted_index = int(
        np.argmax(predictions[0])
    )

    # Get confidence
    confidence = float(
        np.max(predictions[0])
    )

    # Check whether predicted index exists
    if predicted_index not in labels:

        raise ValueError(
            "Predicted class index "
            f"{predicted_index} was not found "
            "in class_indices.json."
        )

    # Get disease name
    predicted_label = labels[
        predicted_index
    ]

    return predicted_label, confidence


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image and the trained "
    "AI model will predict the corresponding "
    "plant disease."
)


# ============================================================
# IMAGE UPLOADER
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
# IMAGE PROCESSING
# ============================================================

if uploaded_image is not None:

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_image
        )

    except Exception as e:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # --------------------------------------------------------
    # Display image and prediction section
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    with col1:

        st.image(
            image,
            caption="Uploaded Leaf Image",
            width=200
        )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🔍 Classify",
            type="primary"
        ):

            try:

                predicted_label, confidence = (
                    predict_image_class(
                        model,
                        image,
                        class_indices
                    )
                )

                # --------------------------------------------
                # Prediction result
                # --------------------------------------------

                st.success(
                    f"🌱 Prediction: {predicted_label}"
                )

                st.info(
                    "📊 Confidence: "
                    f"{confidence * 100:.2f}%"
                )

            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.exception(e)