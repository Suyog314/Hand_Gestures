
# for collecting 0-9 hand landmark data:


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []
labels = []

print("Show a number with your hand (0-9). Press keys 0-9 to label, q to quit and save.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Show Number with Hand", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key != -1 and chr(key).isdigit() and int(chr(key)) in range(10):
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            data.append(vector)
            labels.append(int(chr(key)))
            print(f"Captured for label: {chr(key)}")

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('hand_number_data.csv', index=False)
print("Saved to hand_number_data.csv")
