
# for training and saving the classifier:


## using random forest:

# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import joblib

# # Load your dataset
# df = pd.read_csv('hand_number_data.csv')
# X = df.drop('label', axis=1)
# y = df['label']

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, y_train)

# # Evaluate model
# accuracy = model.score(X_test, y_test)
# print("✅ Model trained! Accuracy:", round(accuracy * 100, 2), "%")

# # Save the model
# joblib.dump(model, 'number_model.pkl')
# print("📦 Model saved as number_model.pkl")



## using neural networks:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv('hand_number_data.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Normalize data
X = X / np.max(X)  # simple normalization

# Convert labels to one-hot encoding
y_cat = to_categorical(y, num_classes=10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build NN model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for digits 0–9
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Neural Net trained! Accuracy: {round(acc * 100, 2)}%")

# Save model
model.save('number_model_nn.h5')
print("📦 Saved as number_model_nn.h5")
