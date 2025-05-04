import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
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

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
Path('models').mkdir(exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build model with enhanced architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train with callbacks
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
mc = ModelCheckpoint(
    'models/nn_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=[es, mc],
    verbose=2
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")

# Save final model and class names
model.save('models/nn_model.h5')
with open('models/classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Neural network model and class labels saved in 'models/' directory.") 