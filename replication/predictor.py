#
# Produces an output deforestation mask
#

# Import packages

import os
# FORCE Legacy behavior before any other imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import load_model
import numpy as np
from PIL import Image
import sys
from tqdm import tqdm

# Compatibility Fix: Tell TensorFlow how to handle legacy layers
custom_objects = {
    "TFOpLambda": tf.keras.layers.Lambda,
    "broadcast_to": tf.broadcast_to
}

def run_prediction(model_id, image_path):
    # 1. Map Model ID to filename
    model_files = {
        '1': 'unet-attention-3d.hdf5',
        '2': 'unet-attention-4d.hdf5',
        '3': 'unet-attention-4d-atlantic.hdf5'
    }
    
    model_name = model_files.get(str(model_id))
    if not model_name or not os.path.exists(model_name):
        print(f"Error: Model file {model_name} not found in current folder.")
        return

    # 2. Load Model with Legacy Support
    print(f"--- Loading {model_name} ---")

    # We use tf_keras specifically and ensure custom_objects are passed correctly
    with tf.keras.utils.custom_object_scope(custom_objects):
        # The 'safe_mode=False' allows loading of legacy Lambda layers
        model = load_model(model_name, compile=False, safe_mode=False, custom_objects=custom_objects)

    print("Model loaded successfully!")

    # 3. Pre-process Image
    print(f"--- Processing: {image_path} ---")
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predict with Progress Bar
    # Since model.predict() is one step, we use a manual tqdm bar for visual feedback
    with tqdm(total=100, desc="AI Analysis") as pbar:
        prediction = model.predict(img_array, verbose=0)
        pbar.update(100) # Jump to 100% when prediction completes

    # 5. Save Results
    mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    output_name = f"result_{os.path.basename(image_path)}.png"
    Image.fromarray(mask.squeeze()).save(output_name)
    print(f"Success! Deforestation mask saved as: {output_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predictor.py [ModelID] [ImagePath]")
    else:
        run_prediction(sys.argv[1], sys.argv[2])