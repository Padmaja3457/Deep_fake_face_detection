import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization



def get_model(model_weights_path: str):
    # 1. Rebuild the exact architecture shell
    base_model = InceptionResNetV2(weights=None, include_top=False, input_shape=(299, 299, 3))
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # 2. Load the H5 weights
    if os.path.exists(model_weights_path):
        try:
            # H5 files are much more stable for weight loading
            model.load_weights(model_weights_path)
            print("✅ SUCCESS: Model brain loaded perfectly!")
        except Exception as e:
            print(f"❌ Error loading H5: {e}")
            # Fallback for naming issues
            model.load_weights(model_weights_path, by_name=True, skip_mismatch=True)
    
    return model
def get_model_summary(model):
    """
    Returns the Keras model summary as a string for Streamlit display.
    """
    model_summary_string = []
    model.summary(print_fn=lambda x: model_summary_string.append(x), line_length=78)
    return '\n'.join(model_summary_string)

def model_prediction(image_path: str, model):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use the same preprocessing used in your training code
    img_array = preprocess_input(img_array)

    # Get the raw prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    # Since class_indices are {'fake': 0, 'real': 1}:
    # A value closer to 1 is REAL, a value closer to 0 is FAKE.
    if prediction > 0.5:
        label = 'Real Face'
        real_prob = round(float(prediction) * 100)
        fake_prob = 100 - real_prob
    else:
        label = 'Fake Face'
        fake_prob = round(float(1 - prediction) * 100)
        real_prob = 100 - fake_prob

    return label, real_prob, fake_prob

def _prediction_probability(prediction: float, prediction_label: str, prediction_classes: list):
    """
    Calculates percentage probabilities for display.
    """
    temp_classes = list(prediction_classes)
    temp_classes.remove(prediction_label)
    
    prediction_prob = prediction if prediction_label == 'Real Face' else (1 - prediction)
    probabilities = {
        prediction_label: round(float(prediction_prob) * 100),
        temp_classes[0]: round((1 - float(prediction_prob)) * 100)
    }
    return probabilities
