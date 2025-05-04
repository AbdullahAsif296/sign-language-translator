import numpy as np
import cv2
from scipy.ndimage import rotate, shift
import random
from scipy.spatial.transform import Rotation

class DataAugmentor:
    def __init__(self, noise_std=0.01, rotation_range=15, shift_range=0.15, scale_range=0.2):
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.scale_range = scale_range

    def add_noise(self, landmarks):
        """Add Gaussian noise to landmarks with varying intensity"""
        noise_std = random.uniform(0.5 * self.noise_std, 1.5 * self.noise_std)
        noise = np.random.normal(0, noise_std, landmarks.shape)
        return landmarks + noise

    def rotate_landmarks(self, landmarks):
        """Rotate landmarks in 3D space"""
        # Reshape flattened landmarks to 3D points
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Get wrist point (first landmark)
        wrist_point = landmarks_3d[0]
        
        # Center landmarks around wrist
        centered = landmarks_3d - wrist_point
        
        # Generate random rotation angles for each axis
        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        
        # Create rotation matrix
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        rotated = rotation.apply(centered)
        
        # Restore original position and flatten
        return (rotated + wrist_point).flatten()

    def shift_landmarks(self, landmarks):
        """Randomly shift landmarks in 3D space"""
        # Reshape flattened landmarks to 3D points
        landmarks_3d = landmarks.reshape(-1, 3)
        
        shift_x = random.uniform(-self.shift_range, self.shift_range)
        shift_y = random.uniform(-self.shift_range, self.shift_range)
        shift_z = random.uniform(-self.shift_range/2, self.shift_range/2)  # Smaller Z shift
        
        # Apply shift and flatten
        shifted = landmarks_3d + np.array([shift_x, shift_y, shift_z])
        return shifted.flatten()

    def scale_landmarks(self, landmarks):
        """Randomly scale landmarks with different scales for each axis"""
        # Reshape flattened landmarks to 3D points
        landmarks_3d = landmarks.reshape(-1, 3)
        
        wrist_point = landmarks_3d[0]
        centered = landmarks_3d - wrist_point
        
        # Different scale factors for each axis
        scale_x = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        scale_y = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        scale_z = random.uniform(1 - self.scale_range/2, 1 + self.scale_range/2)  # Smaller Z scale
        
        # Create scale matrix for each point
        scale_matrix = np.array([scale_x, scale_y, scale_z])
        scaled = centered * scale_matrix
        
        # Restore original position and flatten
        return (scaled + wrist_point).flatten()

    def perspective_transform(self, landmarks):
        """Apply perspective transformation to landmarks"""
        # Reshape flattened landmarks to 3D points
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Get wrist point
        wrist_point = landmarks_3d[0]
        centered = landmarks_3d - wrist_point
        
        # Create perspective matrix
        perspective = np.eye(3)
        perspective[0, 2] = random.uniform(-0.1, 0.1)  # x perspective
        perspective[1, 2] = random.uniform(-0.1, 0.1)  # y perspective
        
        # Apply perspective transform and flatten
        transformed = np.dot(centered, perspective)
        return (transformed + wrist_point).flatten()

    def mirror_landmarks(self, landmarks):
        """Mirror landmarks horizontally"""
        # Reshape flattened landmarks to 3D points
        landmarks_3d = landmarks.reshape(-1, 3)
        
        wrist_point = landmarks_3d[0]
        centered = landmarks_3d - wrist_point
        
        # Create mirror matrix for each point
        mirror_matrix = np.array([-1, 1, 1])
        mirrored = centered * mirror_matrix
        
        # Restore original position and flatten
        return (mirrored + wrist_point).flatten()

    def augment_landmarks(self, landmarks):
        """Apply all augmentation techniques with adaptive probabilities"""
        augmented = landmarks.copy()
        
        # Apply augmentations with adaptive probabilities
        if random.random() < 0.8:  # Increased probability for noise
            augmented = self.add_noise(augmented)
            
        if random.random() < 0.6:  # Increased probability for rotation
            augmented = self.rotate_landmarks(augmented)
            
        if random.random() < 0.6:  # Increased probability for shift
            augmented = self.shift_landmarks(augmented)
            
        if random.random() < 0.5:  # Moderate probability for scale
            augmented = self.scale_landmarks(augmented)
            
        if random.random() < 0.4:  # Lower probability for perspective
            augmented = self.perspective_transform(augmented)
            
        if random.random() < 0.3:  # Lower probability for mirroring
            augmented = self.mirror_landmarks(augmented)
            
        return augmented

    def augment_batch(self, X, y, num_augmentations=5):
        """Augment a batch of data with class balancing"""
        augmented_X = []
        augmented_y = []
        
        # Count samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_samples = np.max(class_counts)
        
        for class_idx in unique_classes:
            # Get indices of samples for this class
            class_indices = np.where(y == class_idx)[0]
            
            # Add original data
            for idx in class_indices:
                augmented_X.append(X[idx])
                augmented_y.append(y[idx])
            
            # Calculate how many augmentations we need to balance the class
            num_needed = max_samples - len(class_indices)
            num_augmentations_per_sample = max(1, num_needed // len(class_indices))
            
            # Add augmented versions
            for idx in class_indices:
                for _ in range(num_augmentations_per_sample):
                    augmented_X.append(self.augment_landmarks(X[idx]))
                    augmented_y.append(y[idx])
                
        return np.array(augmented_X), np.array(augmented_y) 