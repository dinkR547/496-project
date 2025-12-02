import os
import io
import numpy as np

try:
    import flask
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ModuleNotFoundError:
    print("Error: Flask not found. Run 'pip install flask flask-cors.'")
    exit()

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("Error: TensorFlow not found. Run 'pip install tensorflow.'")
    exit()

try:
    from PIL import Image
except ModuleNotFoundError:
    print("Error: Pillow not found. Run 'pip install Pillow'")
    exit()

print("--- Starting Server ---")

# Serve static files (like index.html) from the current folder
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

MODEL_PATH = 'cvd_cnn.h5'
model = None

# --- Model Architecture Functions ---

def create_conv_layer(x, filters, kernel_size, strides, activation):
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation)(x)
    pool = tf.keras.layers.MaxPool2D(strides=strides)(conv)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(pool)

def create_dense_layers(x, hl_list, hl_conv_activation, dropout_vals):
    for units, drop in zip(hl_list, dropout_vals):
        x = tf.keras.layers.Dense(units=units, activation=hl_conv_activation)(x)
        x = tf.keras.layers.Dropout(drop)(x)
    return x

def CNN_Mod(input_shape=(300, 300, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # FIX: Removed the broken 'data_augmentation' block.
    # We connect 'inputs' directly to the first layer.
    x = create_conv_layer(inputs, 32, (3,3), (2,2), 'relu')
    x = create_conv_layer(x, 32, (3,3), (2,2), 'relu')
    
    x = tf.keras.layers.Flatten()(x)
    x = create_dense_layers(x, [128, 64, 32, 16], 'relu', [0.3]*4)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: {MODEL_PATH} not found in current folder.")
        return

    try:
        print("Loading model weights...")
        model = CNN_Mod()
        
        # FIX: Added skip_mismatch=True to ignore the missing data_augmentation layer weights
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")

def preprocess_image(image_file):
    img = Image.open(io.BytesIO(image_file.read())).convert('L')
    img = img.resize((300, 300))
    arr = np.array(img).reshape((1, 300, 300, 1)) / 255.0
    return arr

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    try:
        prediction = model.predict(preprocess_image(request.files['file']))
        
        # NOTE: The notebook uses [NonCOVID, COVID]
        # argmax(0) = NonCOVID, argmax(1) = COVID
        is_covid = np.argmax(prediction[0]) == 1
        
        return jsonify({
            'prediction': 'COVID-19 Positive' if is_covid else 'COVID-19 Negative',
            'confidence': float(np.max(prediction[0]))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    if model:
        app.run(debug=True, port=5000, host='0.0.0.0')