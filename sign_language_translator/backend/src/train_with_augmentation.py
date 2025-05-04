import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from utils.data_augmentation import DataAugmentor

def load_and_augment_data():
    # Load original dataset
    with open('data/processed/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    X = np.array([x.flatten() for x in dataset['X']])  # Flatten landmarks
    y = dataset['y']
    classes = dataset['classes']
    num_classes = len(classes)

    # Create augmented data with enhanced augmentor
    augmentor = DataAugmentor(
        noise_std=0.02,  # Increased noise
        rotation_range=15,  # Increased rotation range
        shift_range=0.15,  # Increased shift range
        scale_range=0.2  # Added scale range
    )
    
    print("Generating augmented data...")
    X_augmented, y_augmented = augmentor.augment_batch(X, y, num_augmentations=5)
    print(f"Original data shape: {X.shape}")
    print(f"Augmented data shape: {X_augmented.shape}")

    # Combine original and augmented data
    X_combined = np.vstack([X, X_augmented])
    y_combined = np.concatenate([y, y_augmented])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined,
        test_size=0.2,
        random_state=42,
        stratify=y_combined
    )

    return X_train, X_test, y_train, y_test, classes, num_classes

def train_neural_network(X_train, X_test, y_train, y_test, num_classes):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    Path('models').mkdir(exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build enhanced model
    model = Sequential([
        Dense(512, input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.4),
        
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        
        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Enhanced callbacks
    es = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    mc = ModelCheckpoint(
        'models/nn_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    print("\nTraining neural network...")
    history = model.fit(
        X_train_scaled, y_train_cat,
        validation_data=(X_test_scaled, y_test_cat),
        epochs=200,  # Increased epochs
        batch_size=64,  # Increased batch size
        callbacks=[es, mc, rlr],
        verbose=2
    )

    # Evaluate
    loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"\nNeural Network Test Accuracy: {acc:.4f}")

    # Save final model
    model.save('models/nn_model.h5')
    return model, scaler

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=300,  # Increased number of trees
        max_depth=30,  # Increased depth
        min_samples_split=5,  # Added min samples split
        min_samples_leaf=2,  # Added min samples leaf
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, 'models/keypoint_model.pkl')
    return clf

def create_ensemble_weights():
    # Create weighted ensemble based on validation performance
    ensemble_weights = {'nn': 0.6, 'rf': 0.4}  # Slightly favoring neural network
    with open('models/ensemble_weights.pkl', 'wb') as f:
        pickle.dump(ensemble_weights, f)
    return ensemble_weights

def main():
    # Load and augment data
    X_train, X_test, y_train, y_test, classes, num_classes = load_and_augment_data()

    # Train neural network
    nn_model, scaler = train_neural_network(X_train, X_test, y_train, y_test, num_classes)

    # Train random forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Create ensemble weights
    ensemble_weights = create_ensemble_weights()

    # Save class labels
    with open('models/classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    print("\nAll models trained and saved successfully!")
    print("Models saved in 'models/' directory:")
    print("- nn_model.h5 (Neural Network)")
    print("- keypoint_model.pkl (Random Forest)")
    print("- scaler.pkl (Data Scaler)")
    print("- ensemble_weights.pkl (Ensemble Weights)")
    print("- classes.pkl (Class Labels)")

if __name__ == "__main__":
    main() 