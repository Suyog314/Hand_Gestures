
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


def get_hand_vector(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None


## this is for random forest:
# import joblib
# model = joblib.load('number_model.pkl')  # Load trained model

# def predict_sign(vector):
#     if vector is None:
#         return "No Hand Detected"
#     return str(model.predict([vector])[0])


## this is for nn:
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('number_model_nn.h5')

def predict_sign(vector):
    if vector is None:
        return "No Hand Detected"
    
    vector = np.array(vector).reshape(1, -1)
    vector = vector / np.max(vector)  # normalize same way as training
    
    pred = model.predict(vector)
    return str(np.argmax(pred))




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    vector = get_hand_vector(frame)
    prediction = predict_sign(vector)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
