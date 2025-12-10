from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Build the same model architecture as in training
def get_conv_model_normal():
    inp_shape = (96, 96, 3)
    act = 'relu'
    drop = .5
    kernal_reg = regularizers.l1(.001)
    dil_rate = 2

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation=act, input_shape=inp_shape,
                     kernel_regularizer=kernal_reg,
                     kernel_initializer='he_uniform', padding='same', name='Input_Layer'))
    model.add(Dense(64, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer=kernal_reg, dilation_rate=dil_rate,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Dense(64, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernal_reg, dilation_rate=dil_rate,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer=kernal_reg, dilation_rate=dil_rate,
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(drop))

    model.add(Dense(1, activation='sigmoid', name='Output_Layer'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Load model at startup
print("Loading model...")
model = get_conv_model_normal()
model.load_weights('models/CNN-Final.h5')
print("Model loaded successfully!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img):
    """Preprocess image same as training"""
    img = img.resize((96, 96))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            # Read image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            # Preprocess
            processed_img = preprocess_image(img)

            # Predict
            threshold = 0.75
            prediction = model.predict(processed_img, verbose=0).squeeze()
            probability = float(prediction)

            if probability < threshold:
                result = "Normal"
                result_class = "normal"
            else:
                result = "Non-Normal (Có dấu hiệu bất thường)"
                result_class = "non-normal"

            # Convert image to base64 for display
            buffered = io.BytesIO()
            img_display = img.resize((300, 300))
            img_display.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return jsonify({
                'success': True,
                'result': result,
                'result_class': result_class,
                'probability': round(probability * 100, 2),
                'image': img_base64
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
