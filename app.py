# app.py
import io
import os

import cv2
import numpy as np
import streamlit as st
from main import BrainTumorDetector, generate_plots
from PIL import Image


def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
    
    st.title("Brain Tumor Detection System")
    st.write("Upload a brain MRI scan to detect the presence of tumors")
    
    # Check if model exists, if not offer to train
    model_path = 'brain_tumor_model.pkl'
    detector = BrainTumorDetector()
    
    if not os.path.exists(model_path):
        st.warning("Pre-trained model not found. You need to train the model first.")
        if st.button("Train New Model"):
            with st.spinner("Training model... This might take a while."):
                detector = train_model()
                if detector is None:
                    st.error("Failed to train model. Please check your dataset.")
                    return
                st.success("Model trained successfully!")
    else:
        # Load the pre-trained model
        detector.load_model(model_path)
        st.success("Pre-trained model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=300)
        
        # Process the image and make prediction
        with st.spinner("Analyzing image..."):
            try:
                prediction, probability, preprocessed, segmented, mask = detector.predict(image)
                
                # Get visualization plots
                plot_buf = generate_plots(image, preprocessed, segmented, mask, prediction, probability)
                
                # Display results
                st.subheader("Analysis Results")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(plot_buf, caption="Detection Results", use_column_width=True)
                
                with col2:
                    result_text = "Tumor Detected" if prediction == 1 else "No Tumor Detected"
                    result_color = "red" if prediction == 1 else "green"
                    
                    st.markdown(f"<h2 style='color: {result_color};'>{result_text}</h2>", unsafe_allow_html=True)
                    
                    confidence = max(probability) * 100
                    st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Display confidence gauge
                    st.progress(confidence/100)
                    
                    if prediction == 1:
                        st.warning("Please consult with a medical professional for confirmation.")
                    else:
                        st.info("The scan appears to be normal.")
                
                # Additional information
                with st.expander("See Processing Details"):
                    st.write("This system uses digital image processing techniques to detect potential brain tumors:")
                    st.write("1. **Preprocessing**: Converts to grayscale, enhances contrast, and reduces noise")
                    st.write("2. **Segmentation**: Identifies potential tumor regions using thresholding")
                    st.write("3. **Feature Extraction**: Analyzes texture and shape features")
                    st.write("4. **Classification**: Uses a machine learning model to determine if a tumor is present")
                    
                    st.text(f"Tumor probability: {probability[1]:.4f}")
                    st.text(f"Healthy probability: {probability[0]:.4f}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please try with another image or ensure the image is a clear brain MRI scan.")


def train_model():
    """Function to train a new model for the Streamlit app"""
    from main import train_and_save_model
    detector = train_and_save_model()
    return detector


if __name__ == "__main__":
    main()