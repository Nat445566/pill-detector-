import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile

# Initialize session state
if 'pill_count' not in st.session_state:
    st.session_state.pill_count = 0
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = {}

st.title("💊 Pill Counter - BMDS2133 Image Processing")
st.write("Upload an image of pills to count them automatically")

# Sidebar for parameters
st.sidebar.header("Processing Parameters")
min_area = st.sidebar.slider("Minimum Pill Area", 100, 1000, 500)
max_area = st.sidebar.slider("Maximum Pill Area", 1000, 10000, 5000)
blur_size = st.sidebar.slider("Blur Size", 1, 15, 7)
if blur_size % 2 == 0:  # Ensure odd number
    blur_size += 1

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
    
    if st.button("Count Pills"):
        with st.spinner("Processing image..."):
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            st.session_state.processing_steps['Grayscale'] = gray
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            st.session_state.processing_steps['Blurred'] = blurred
            
            # Apply adaptive thresholding - FIXED: Changed cv to cv2
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            st.session_state.processing_steps['Thresholded'] = thresh
            
            # Perform morphological operations
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
            st.session_state.processing_steps['Morphological Operations'] = cleaned
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and draw them
            pill_count = 0
            pill_data = []
            result_image = image.copy()
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    pill_count += 1
                    cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                    
                    # Get pill properties
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(result_image, str(pill_count), (cX - 10, cY - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Store pill data
                    pill_data.append({
                        'Pill #': pill_count,
                        'Area': area,
                        'Center X': cX,
                        'Center Y': cY
                    })
            
            # Update session state
            st.session_state.pill_count = pill_count
            
            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Detected {pill_count} pills", use_container_width=True)
            
            with col2:
                st.metric("Total Pills Counted", pill_count)
                
                if pill_data:
                    df = pd.DataFrame(pill_data)
                    st.dataframe(df, use_container_width=True)
            
            # Show processing steps
            st.subheader("Processing Steps")
            cols = st.columns(len(st.session_state.processing_steps))
            
            for idx, (step_name, step_image) in enumerate(st.session_state.processing_steps.items()):
                with cols[idx]:
                    st.image(step_image, caption=step_name, use_container_width=True)

# Display final count
st.success(f"Total pills counted: {st.session_state.pill_count}")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Upload a clear image of pills on a contrasting background
2. Adjust parameters if needed:
   - Min/Max Area: Filter by pill size
   - Blur Size: Reduce noise (odd numbers only)
3. Click 'Count Pills' to process the image
4. Review results and processing steps
""")
