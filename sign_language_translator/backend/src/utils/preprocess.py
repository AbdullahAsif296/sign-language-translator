import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
from pathlib import Path

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
    def extract_landmarks(self, image):
        """Extract hand landmarks from an image using MediaPipe."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Get landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        return landmarks_array
    
    def preprocess_image(self, image_path):
        """Preprocess a single image."""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        # Extract landmarks
        landmarks = self.extract_landmarks(image)
        if landmarks is None:
            return None
            
        return landmarks
    
    def process_dataset(self):
        """Process all images in the dataset."""
        # Create processed data directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        class_dirs = [d for d in self.raw_data_path.iterdir() if d.is_dir()]
        
        # Process each class
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            class_name = class_dir.name
            processed_class_dir = self.processed_data_path / class_name
            processed_class_dir.mkdir(exist_ok=True)
            
            # Process each image in the class
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                for img_path in class_dir.glob(ext):
                    landmarks = self.preprocess_image(img_path)
                    if landmarks is not None:
                        # Save landmarks as numpy array
                        save_path = processed_class_dir / f"{img_path.stem}.npy"
                        np.save(save_path, landmarks)
    
    def create_dataset_file(self):
        """Create a single dataset file with all processed data."""
        dataset = {
            'X': [],
            'y': [],
            'classes': []
        }
        
        # Get all class directories
        class_dirs = [d for d in self.processed_data_path.iterdir() if d.is_dir()]
        
        # Process each class
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            dataset['classes'].append(class_name)
            
            # Load all landmarks for this class
            for landmark_file in class_dir.glob("*.npy"):
                landmarks = np.load(landmark_file)
                dataset['X'].append(landmarks)
                dataset['y'].append(class_idx)
        
        # Convert to numpy arrays
        dataset['X'] = np.array(dataset['X'])
        dataset['y'] = np.array(dataset['y'])
        
        # Save dataset
        save_path = self.processed_data_path / "dataset.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset

if __name__ == "__main__":
    # Define paths
    raw_data_path = "data/raw/train"
    processed_data_path = "data/processed"
    
    # Create preprocessor
    preprocessor = DataPreprocessor(raw_data_path, processed_data_path)
    
    # Process dataset
    print("Processing dataset...")
    preprocessor.process_dataset()
    
    # Create dataset file
    print("Creating dataset file...")
    dataset = preprocessor.create_dataset_file()
    
    print(f"Dataset processed successfully!")
    print(f"Number of samples: {len(dataset['X'])}")
    print(f"Number of classes: {len(dataset['classes'])}")
    print(f"Classes: {dataset['classes']}") 