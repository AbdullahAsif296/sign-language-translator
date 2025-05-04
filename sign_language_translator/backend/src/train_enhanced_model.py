import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

# Load processed dataset
with open('data/processed/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X = np.array([x.flatten() for x in dataset['X']])  # Flatten landmarks
y = dataset['y']
classes = dataset['classes']
num_classes = len(classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load existing models
print("Loading existing models...")
nn_model = load_model('models/nn_model.h5')
rf_model = joblib.load('models/keypoint_model.pkl')

# Create ensemble predictions
print("Creating ensemble predictions...")
nn_pred = nn_model.predict(X_test)
rf_pred = rf_model.predict_proba(X_test)

# Combine predictions (simple averaging)
ensemble_pred = (nn_pred + rf_pred) / 2
ensemble_pred_labels = np.argmax(ensemble_pred, axis=1)

# Calculate accuracy
accuracy = np.mean(ensemble_pred_labels == y_test)
print(f"Ensemble Test Accuracy: {accuracy:.4f}")

# Save ensemble weights
ensemble_weights = {'nn': 0.5, 'rf': 0.5}  # Equal weights for now
with open('models/ensemble_weights.pkl', 'wb') as f:
    pickle.dump(ensemble_weights, f)

print("Ensemble weights saved in 'models/ensemble_weights.pkl'") 