from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
from pathlib import Path
from enhanced_model import EnhancedModel
from utils.sequence_validator import SequenceValidator
from utils.data_augmentation import DataAugmentor
import logging
import traceback

app = Flask(__name__)

# Configure CORS to allow all origins during development
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize models and validators
MODEL_PATH = os.path.join('models', 'enhanced_model')
CLASSES_PATH = os.path.join('models', 'classes.pkl')
SEQUENCE_VALIDATOR_PATH = os.path.join('models', 'sequence_validator.pkl')

# Load class labels
try:
    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)
    logger.info(f"Successfully loaded {len(classes)} classes")
except Exception as e:
    logger.error(f"Error loading classes: {str(e)}")
    classes = []

# Initialize enhanced model
try:
    model = EnhancedModel()
    logger.info("Successfully initialized enhanced model")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model = None

# Initialize sequence validator
try:
    sequence_validator = SequenceValidator()
    if os.path.exists(SEQUENCE_VALIDATOR_PATH):
        sequence_validator.load(SEQUENCE_VALIDATOR_PATH)
    logger.info("Successfully initialized sequence validator")
except Exception as e:
    logger.error(f"Error initializing sequence validator: {str(e)}")
    sequence_validator = None

# Initialize data augmentor
try:
    data_augmentor = DataAugmentor()
    logger.info("Successfully initialized data augmentor")
except Exception as e:
    logger.error(f"Error initializing data augmentor: {str(e)}")
    data_augmentor = None

# Store recent predictions for sequence validation
recent_predictions = []
MAX_SEQUENCE_LENGTH = 10

@app.route('/health', methods=['GET'])
def health_check():
    try:
        status = {
            "status": "healthy",
            "models_loaded": bool(model and model.models),
            "num_classes": len(classes),
            "sequence_validator_loaded": bool(sequence_validator),
            "data_augmentor_loaded": bool(data_augmentor)
        }
        logger.info(f"Health check: {status}")
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get landmarks from request
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
            
        landmarks = data.get('landmarks', [])
        if not landmarks:
            logger.error("No landmarks provided in request")
            return jsonify({'error': 'No landmarks provided'}), 400
            
        # Convert landmarks to numpy array and validate shape
        try:
            landmarks_array = np.array(landmarks, dtype=np.float32)
            logger.info(f"Received landmarks with shape: {landmarks_array.shape}")
            logger.info(f"First few landmarks: {landmarks_array[:5]}")
            
            # Validate landmarks array
            if np.any(np.isnan(landmarks_array)) or np.any(np.isinf(landmarks_array)):
                logger.error("Landmarks contain NaN or Inf values")
                return jsonify({'error': 'Invalid landmarks data: contains NaN or Inf values'}), 400
            
            # Reshape landmarks to match model input shape
            landmarks_array = landmarks_array.reshape(1, -1)
            logger.info(f"Reshaped landmarks to: {landmarks_array.shape}")
            
        except Exception as e:
            logger.error(f"Error processing landmarks: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing landmarks: {str(e)}'}), 400
        
        try:
            # Make prediction
            if not model:
                logger.error("Model not initialized")
                return jsonify({'error': 'Model not initialized'}), 500
                
            # Get prediction from model
            logger.info("Making prediction with model...")
            prediction, confidence = model.predict(landmarks_array)
            
            logger.info(f"Raw prediction: {prediction}, Confidence: {confidence}")
            
            # Validate prediction
            if prediction is None:
                logger.error("Model returned None prediction")
                return jsonify({
                    'error': 'Model returned None prediction',
                    'prediction': None,
                    'confidence': 0.0
                }), 500
                
            if not isinstance(prediction, (int, np.integer)):
                logger.error(f"Invalid prediction type: {type(prediction)}")
                return jsonify({
                    'error': f'Invalid prediction type: {type(prediction)}',
                    'prediction': None,
                    'confidence': 0.0
                }), 500
                
            if prediction < 0 or prediction >= len(classes):
                logger.error(f"Invalid prediction index: {prediction}")
                return jsonify({
                    'error': f'Invalid prediction index: {prediction}',
                    'prediction': None,
                    'confidence': 0.0
                }), 500
                
            prediction_label = classes[prediction]
            logger.info(f"Final prediction: {prediction_label}, Confidence: {confidence}")
            
            return jsonify({
                'prediction': prediction_label,
                'confidence': float(confidence)
            })
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': f'Prediction error: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        sequences = data.get('sequences', [])
        
        # Update sequence validator
        sequence_validator.train(sequences)
        sequence_validator.save(SEQUENCE_VALIDATOR_PATH)
        
        return jsonify({'message': 'Training completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        # TODO: Implement file processing logic
        return jsonify({'message': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 