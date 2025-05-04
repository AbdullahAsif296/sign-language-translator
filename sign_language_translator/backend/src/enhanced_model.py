import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
from pathlib import Path
import traceback

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, features):
        attention_weights = tf.nn.softmax(self.V(tf.nn.tanh(self.W(features))), axis=1)
        context = attention_weights * features
        return tf.reduce_sum(context, axis=1)

class EnhancedModel:
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.classes = []
        self.scaler = None
        self.load_models()

    def preprocess_landmarks(self, landmarks):
        """
        Preprocess the landmarks data before making predictions.
        This includes scaling and any other necessary transformations.
        """
        try:
            # Ensure landmarks is a numpy array
            landmarks = np.array(landmarks, dtype=np.float32)
            
            # Reshape if needed
            if len(landmarks.shape) == 1:
                landmarks = landmarks.reshape(1, -1)
            
            # Scale the landmarks if scaler is available
            if self.scaler is not None:
                landmarks = self.scaler.transform(landmarks)
            
            return landmarks
        except Exception as e:
            print(f"Error in preprocessing landmarks: {str(e)}")
            print(traceback.format_exc())
            raise

    def load_models(self):
        try:
            print("Loading models...")
            
            # Load scaler
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")

            # Load neural network model
            self.models['nn'] = load_model('models/nn_model.h5')
            print("Neural network model loaded successfully")

            # Load random forest model with memory mapping
            try:
                self.models['rf'] = joblib.load('models/keypoint_model.pkl', mmap_mode='r')
                print("Random forest model loaded successfully")
            except MemoryError:
                print("Warning: Could not load Random Forest model due to memory constraints. Using only Neural Network model.")
                self.models['rf'] = None
                self.ensemble_weights['rf'] = 0.0
                # Adjust other weights proportionally
                total_weight = sum(self.ensemble_weights.values()) - self.ensemble_weights['rf']
                for model in self.ensemble_weights:
                    if model != 'rf':
                        self.ensemble_weights[model] /= total_weight

            # Load ensemble weights
            with open('models/ensemble_weights.pkl', 'rb') as f:
                self.ensemble_weights = pickle.load(f)
            print("Ensemble weights loaded successfully")
            print("Current ensemble weights:", self.ensemble_weights)

            # Load class labels
            with open('models/classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            print("Class labels loaded successfully")

            print(f"Models loaded successfully. Number of classes: {len(self.classes)}")
            print(f"Ensemble weights: {self.ensemble_weights}")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def predict(self, landmarks):
        try:
            print("Starting prediction process...")
            print(f"Input landmarks shape: {landmarks.shape}")
            
            # Validate input
            if landmarks is None:
                print("Error: Input landmarks is None")
                return None, 0.0
                
            if not isinstance(landmarks, np.ndarray):
                print(f"Error: Input landmarks must be numpy array, got {type(landmarks)}")
                return None, 0.0
                
            if landmarks.size == 0:
                print("Error: Input landmarks is empty")
                return None, 0.0
                
            if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
                print("Error: Input landmarks contains NaN or Inf values")
                return None, 0.0
            
            # Preprocess landmarks
            try:
                # If scaler is not available, just reshape the landmarks
                if self.scaler is None:
                    print("Warning: Scaler not available, using raw landmarks")
                    processed_landmarks = landmarks.reshape(1, -1)
                else:
                    processed_landmarks = self.preprocess_landmarks(landmarks)
                print(f"Processed landmarks shape: {processed_landmarks.shape}")
            except Exception as e:
                print(f"Error in preprocessing landmarks: {str(e)}")
                print(traceback.format_exc())
                return None, 0.0
            
            # Get predictions from each model
            predictions = {}
            
            # Neural Network prediction
            try:
                print("Making Neural Network prediction...")
                if 'nn' not in self.models or self.models['nn'] is None:
                    print("Error: Neural Network model not loaded")
                    return None, 0.0
                    
                nn_pred = self.models['nn'].predict(processed_landmarks.reshape(1, -1))
                predictions['nn'] = nn_pred[0]
                print(f"NN prediction shape: {nn_pred.shape}")
            except Exception as e:
                print(f"Error in Neural Network prediction: {str(e)}")
                print(traceback.format_exc())
                return None, 0.0
            
            # Random Forest prediction (if available)
            if self.models['rf'] is not None:
                try:
                    print("Making Random Forest prediction...")
                    rf_pred = self.models['rf'].predict_proba(processed_landmarks.reshape(1, -1))
                    predictions['rf'] = rf_pred[0]
                    print(f"RF prediction shape: {rf_pred.shape}")
                except Exception as e:
                    print(f"Error in Random Forest prediction: {str(e)}")
                    print(traceback.format_exc())
                    # Continue with just NN prediction
                    self.models['rf'] = None
            else:
                print("Random Forest model not available")
            
            # Combine predictions using ensemble weights
            print("Combining predictions...")
            try:
                final_prediction = np.zeros_like(predictions['nn'])
                for model, pred in predictions.items():
                    print(f"Model {model} weight: {self.ensemble_weights[model]}")
                    final_prediction += pred * self.ensemble_weights[model]
                
                # Get the predicted class
                predicted_class = np.argmax(final_prediction)
                confidence = np.max(final_prediction)
                
                print(f"Final prediction: {predicted_class}, Confidence: {confidence}")
                return predicted_class, confidence
            except Exception as e:
                print(f"Error combining predictions: {str(e)}")
                print(traceback.format_exc())
                return None, 0.0
            
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            print(traceback.format_exc())
            return None, 0.0

    def build_lstm_model(self, input_shape, num_classes):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            AttentionLayer(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn_model(self, input_shape, num_classes):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_rf_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def train_models(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        # Train LSTM model
        lstm_model = self.build_lstm_model(X_train.shape[1:], y_train.shape[1])
        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ModelCheckpoint('models/lstm_best.h5', save_best_only=True)
            ]
        )
        self.models['lstm'] = lstm_model

        # Train CNN model
        cnn_model = self.build_cnn_model(X_train.shape[1:], y_train.shape[1])
        cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ModelCheckpoint('models/cnn_best.h5', save_best_only=True)
            ]
        )
        self.models['cnn'] = cnn_model

        # Train Random Forest
        rf_model = self.build_rf_model()
        rf_model.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
        self.models['rf'] = rf_model

    def save_models(self):
        Path('models').mkdir(exist_ok=True)
        self.models['lstm'].save('models/lstm_model.h5')
        self.models['cnn'].save('models/cnn_model.h5')
        joblib.dump(self.models['rf'], 'models/rf_model.pkl')
        with open('models/ensemble_weights.pkl', 'wb') as f:
            pickle.dump(self.ensemble_weights, f)
        with open('models/classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)

    def load_models(self):
        try:
            print("Loading models...")
            # Load the scaler
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")
            
            # Load the neural network model
            self.models['nn'] = load_model('models/nn_model.h5')
            print("Neural Network model loaded successfully")
            
            # Load the random forest model with memory mapping
            try:
                self.models['rf'] = joblib.load('models/keypoint_model.pkl', mmap_mode='r')
                print("Random Forest model loaded successfully")
            except MemoryError:
                print("Warning: Could not load Random Forest model due to memory constraints. Using only Neural Network model.")
                self.models['rf'] = None
                self.ensemble_weights['rf'] = 0.0
                # Adjust other weights proportionally
                total_weight = sum(self.ensemble_weights.values()) - self.ensemble_weights['rf']
                for model in self.ensemble_weights:
                    if model != 'rf':
                        self.ensemble_weights[model] /= total_weight
            
            # Load ensemble weights
            with open('models/ensemble_weights.pkl', 'rb') as f:
                self.ensemble_weights = pickle.load(f)
            print("Ensemble weights loaded successfully")
            print("Current ensemble weights:", self.ensemble_weights)
            
            # Load class labels
            with open('models/classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            print("Class labels loaded successfully")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise 