# brain_tumor_detector.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
import os
import glob
from sklearn.ensemble import RandomForestClassifier
import pickle
import io

class BrainTumorDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def load_dataset(self, tumor_path, healthy_path):
        """Load and prepare the dataset from tumor and healthy brain scan directories"""
        print("Loading dataset...")
        
        # Load tumor images
        tumor_images = []
        for img_path in glob.glob(os.path.join(tumor_path, "*.jpg")) + glob.glob(os.path.join(tumor_path, "*.png")):
            img = cv2.imread(img_path)
            if img is not None:
                tumor_images.append((img, 1))  # 1 = tumor
        
        # Load healthy images
        healthy_images = []
        for img_path in glob.glob(os.path.join(healthy_path, "*.jpg")) + glob.glob(os.path.join(healthy_path, "*.png")):
            img = cv2.imread(img_path)
            if img is not None:
                healthy_images.append((img, 0))  # 0 = healthy
        
        dataset = tumor_images + healthy_images
        print(f"Loaded {len(tumor_images)} tumor images and {len(healthy_images)} healthy images")
        return dataset
    
    def preprocess_image(self, image):
        """Preprocess the brain MRI image"""
        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Resize for consistency
        resized = cv2.resize(gray, (256, 256))
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        
        # Denoise
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return denoised
    
    def segment_tumor(self, preprocessed_img):
        """Segment potential tumor regions"""
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for the largest contour (potential tumor)
        mask = np.zeros_like(preprocessed_img)
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only consider contours with minimum area to avoid noise
            if cv2.contourArea(largest_contour) > 100:
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        return mask, opening
    
    def extract_features(self, preprocessed_img, mask):
        """Extract texture and shape features from the image and segmented region"""
        features = []
        
        # GLCM Texture features
        glcm = graycomatrix(preprocessed_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        features.append(graycoprops(glcm, 'contrast')[0, 0])
        features.append(graycoprops(glcm, 'homogeneity')[0, 0])
        features.append(graycoprops(glcm, 'energy')[0, 0])
        features.append(graycoprops(glcm, 'correlation')[0, 0])
        
        # Shape features from the mask
        if np.max(mask) > 0:  # If segmentation found something
            # Calculate region properties
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask)
            
            if props:
                # Area
                features.append(props[0].area)
                # Eccentricity
                features.append(props[0].eccentricity)
                # Solidity
                features.append(props[0].solidity)
                # Extent
                features.append(props[0].extent)
            else:
                # Default values if no properties found
                features.extend([0, 0, 0, 0])
        else:
            # Default values if no segmentation
            features.extend([0, 0, 0, 0])
            
        # Image intensity statistics
        features.append(np.mean(preprocessed_img))
        features.append(np.std(preprocessed_img))
        
        return np.array(features)
    
    def prepare_data(self, dataset):
        """Prepare features and labels from the dataset"""
        X = []
        y = []
        
        for image, label in dataset:
            preprocessed = self.preprocess_image(image)
            mask, _ = self.segment_tumor(preprocessed)
            features = self.extract_features(preprocessed, mask)
            X.append(features)
            y.append(label)
            
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """Train the tumor classifier"""
        print("Training the classifier...")
        self.classifier.fit(X, y)
        
    def predict(self, image):
        """Predict whether the image contains a tumor"""
        preprocessed = self.preprocess_image(image)
        mask, segmented = self.segment_tumor(preprocessed)
        features = self.extract_features(preprocessed, mask)
        
        prediction = self.classifier.predict([features])[0]
        probability = self.classifier.predict_proba([features])[0]
        
        return prediction, probability, preprocessed, segmented, mask
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        return accuracy
    
    def save_model(self, filename='brain_tumor_model.pkl'):
        """Save the trained model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='brain_tumor_model.pkl'):
        """Load a trained model from a file"""
        with open(filename, 'rb') as f:
            self.classifier = pickle.load(f)
        print(f"Model loaded from {filename}")


def generate_plots(original, preprocessed, segmented, mask, prediction, probability):
    """Generate and return plot images for Streamlit"""
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Preprocessed Image")
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Segmented Image")
    plt.imshow(segmented, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Tumor Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title(f"Prediction: {'Tumor' if prediction == 1 else 'Healthy'}")
    plt.text(0.5, 0.5, f"Confidence: {max(probability):.2f}", ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Instead of showing, save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def train_and_save_model():
    """Train and save the model"""
    # Create detector instance
    detector = BrainTumorDetector()
    
    # Example paths - replace with your actual dataset paths
    tumor_path = "dataset/tumor"
    healthy_path = "dataset/healthy"
    
    # Check if paths exist
    if not os.path.exists(tumor_path) or not os.path.exists(healthy_path):
        print("Dataset directories not found. Creating example directories.")
        os.makedirs(tumor_path, exist_ok=True)
        os.makedirs(healthy_path, exist_ok=True)
        print(f"Please place tumor images in {tumor_path} and healthy brain images in {healthy_path}")
        return None
    
    # Load dataset
    dataset = detector.load_dataset(tumor_path, healthy_path)
    
    if len(dataset) < 2:
        print("Not enough images found in the dataset directories.")
        return None
    
    # Prepare data
    print("Preparing data and extracting features...")
    X, y = detector.prepare_data(dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model performance...")
    detector.evaluate(X_test, y_test)
    
    # Save the model
    detector.save_model()
    
    return detector


if __name__ == "__main__":
    train_and_save_model()