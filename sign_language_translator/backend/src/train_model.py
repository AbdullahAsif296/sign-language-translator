import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# Load processed dataset
with open('data/processed/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X = np.array([x.flatten() for x in dataset['X']])  # Flatten landmarks
y = dataset['y']
classes = dataset['classes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=classes))

# Save model and class names
Path('models').mkdir(exist_ok=True)
joblib.dump(clf, 'models/keypoint_model.pkl')
with open('models/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Model and class labels saved in 'models/' directory.") 