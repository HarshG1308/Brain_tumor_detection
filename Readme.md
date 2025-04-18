# Brain Tumor Detection System

A Python-based application that uses digital image processing and machine learning techniques to detect brain tumors in MRI scans.

## Overview

This project implements a simple yet effective brain tumor detection system that analyzes MRI scans using various image processing techniques and machine learning. The system includes both a backend processing library and a user-friendly Streamlit web interface for easy interaction.

## Features

- **Image Preprocessing**: Converts images to grayscale, enhances contrast, and reduces noise
- **Tumor Segmentation**: Identifies potential tumor regions using thresholding and morphological operations
- **Feature Extraction**: Extracts texture features using GLCM and shape features from segmented regions
- **Machine Learning Classification**: Uses Random Forest algorithm to classify images as either tumor or healthy
- **Model Persistence**: Saves and loads trained models for future use
- **Visualization**: Provides detailed visual feedback of the detection process
- **Web Interface**: User-friendly Streamlit application for easy interaction

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/HarshG1308/Brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create dataset directories:
   ```
   mkdir -p dataset/tumor
   mkdir -p dataset/healthy
   ```

4. Add MRI images to their respective folders:
   - Place brain MRI images with tumors in `dataset/tumor`
   - Place healthy brain MRI images in `dataset/healthy`

## Usage

### Training the Model

Run the following command to train and save the model:

```
python brain_tumor_detector.py
```

This will process your dataset, train a Random Forest classifier, and save the model as `brain_tumor_model.pkl`.

### Running the Web Application

Launch the Streamlit app with:

```
streamlit run app.py
```

The application will be accessible in your web browser (typically at http://localhost:8501).

### Using the Application

1. Upload a brain MRI scan through the web interface
2. The system will process the image and display the results
3. The results include the original image, preprocessed image, segmentation, tumor mask, and prediction

## Project Structure

```
brain-tumor-detection/
├── app.py                      # Streamlit web application
├── brain_tumor_detector.py     # Core detection functionality
├── requirements.txt            # Python dependencies
├── brain_tumor_model.pkl       # Trained model (generated after training)
└── dataset/                    # Dataset directory
    ├── tumor/                  # MRI images with tumors
    └── healthy/                # MRI images without tumors
```

## Requirements

- streamlit
- opencv-python
- numpy
- scikit-image
- scikit-learn
- matplotlib
- pillow

## Technical Details

### Image Processing Pipeline

1. **Preprocessing**:
   - Grayscale conversion
   - Resizing to standardized dimensions (256×256)
   - Contrast enhancement using CLAHE
   - Noise reduction using Gaussian blur

2. **Segmentation**:
   - Otsu's thresholding
   - Morphological operations
   - Contour detection

3. **Feature Extraction**:
   - GLCM texture features (contrast, homogeneity, energy, correlation)
   - Shape features (area, eccentricity, solidity, extent)
   - Intensity statistics (mean, standard deviation)

4. **Classification**:
   - Random Forest classifier with 100 estimators

## Customization

### Changing the Classifier

Edit `brain_tumor_detector.py` to use a different classifier:

```python
# Change from
self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# To another classifier like
self.classifier = GradientBoostingClassifier()
```

### Adding More Features

Modify the `extract_features` method in `BrainTumorDetector` class to include additional features.

## Performance

The system typically achieves accuracy around 85-95% depending on the quality and size of the training dataset.

## Limitations

- Performance depends on the quality of MRI scans
- Not a substitute for professional medical diagnosis
- Limited to binary classification (tumor/no tumor)
- No tumor type classification

## Future Work

- Implement deep learning models for better accuracy
- Add tumor type classification
- Integrate with DICOM file format
- Improve segmentation with advanced techniques

## Acknowledgments

- This project is for educational purposes only
- Not intended for clinical use without professional validation
