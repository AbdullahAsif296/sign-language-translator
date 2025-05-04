import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def visualize_landmarks(image_path, output_path=None):
    """Visualize hand landmarks on an image."""
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        results = hands.process(image_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Save or show the image
        if output_path:
            cv2.imwrite(str(output_path), image)
        else:
            cv2.imshow('Hand Landmarks', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    image_path = Path("backend/data/raw/train/A/1.jpg")  # Replace with your image path
    output_path = Path("backend/data/processed/visualization.jpg")
    
    visualize_landmarks(image_path, output_path) 