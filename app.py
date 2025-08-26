import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Initialize session state
if 'pill_count' not in st.session_state:
    st.session_state.pill_count = 0
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = {}

st.title("💊 Advanced Pill Counter - BMDS2133")
st.write("Upload an image of pills to count them automatically")

# Sidebar for advanced parameters
st.sidebar.header("Processing Parameters")
detection_method = st.sidebar.selectbox(
    "Detection Method", 
    ["Adaptive Thresholding", "Canny Edge Detection", "Otsu's Thresholding", "HSV Color Segmentation"]
)

min_area = st.sidebar.slider("Minimum Pill Area", 50, 1000, 100)
max_area = st.sidebar.slider("Maximum Pill Area", 500, 10000, 3000)

if detection_method == "Adaptive Thresholding":
    block_size = st.sidebar.slider("Block Size", 3, 21, 11, step=2)
    c_value = st.sidebar.slider("C Value", 1, 10, 2)
elif detection_method == "Canny Edge Detection":
    threshold1 = st.sidebar.slider("Canny Threshold 1", 50, 200, 100)
    threshold2 = st.sidebar.slider("Canny Threshold 2", 100, 300, 200)
elif detection_method == "HSV Color Segmentation":
    h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
    h_max = st.sidebar.slider("Hue Max", 0, 179, 179)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 0)
    s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
    v_min = st.sidebar.slider("Value Min", 0, 255, 0)
    v_max = st.sidebar.slider("Value Max", 0, 255, 255)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

def detect_pills(image, method, params):
    """Detect pills using different methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    processing_steps = {'Original': image, 'Grayscale': gray, 'Blurred': blurred}
    
    if method == "Adaptive Thresholding":
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, params['block_size'], params['c_value']
        )
        processing_steps['Thresholded'] = thresh
        
    elif method == "Canny Edge Detection":
        # Canny edge detection
        edges = cv2.Canny(blurred, params['threshold1'], params['threshold2'])
        processing_steps['Edges'] = edges
        thresh = edges
        
    elif method == "Otsu's Thresholding":
        # Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processing_steps['Thresholded'] = thresh
        
    elif method == "HSV Color Segmentation":
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        processing_steps['HSV'] = hsv
        
        # Create mask based on HSV range
        lower_bound = np.array([params['h_min'], params['s_min'], params['v_min']])
        upper_bound = np.array([params['h_max'], params['s_max'], params['v_max']])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        processing_steps['HSV Mask'] = mask
        thresh = mask
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    processing_steps['Cleaned'] = cleaned
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return processing_steps, contours

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
    
    # Prepare parameters based on selected method
    params = {}
    if detection_method == "Adaptive Thresholding":
        params = {'block_size': block_size, 'c_value': c_value}
    elif detection_method == "Canny Edge Detection":
        params = {'threshold1': threshold1, 'threshold2': threshold2}
    elif detection_method == "HSV Color Segmentation":
        params = {
            'h_min': h_min, 'h_max': h_max,
            's_min': s_min, 's_max': s_max,
            'v_min': v_min, 'v_max': v_max
        }
    
    if st.button("Count Pills"):
        with st.spinner("Processing image..."):
            # Detect pills
            processing_steps, contours = detect_pills(image, detection_method, params)
            st.session_state.processing_steps = processing_steps
            
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
                else:
                    st.warning("No pills detected. Try adjusting parameters.")
            
            # Show processing steps
            st.subheader("Processing Steps")
            step_names = list(processing_steps.keys())
            cols = st.columns(len(step_names))
            
            for idx, step_name in enumerate(step_names):
                with cols[idx]:
                    step_img = processing_steps[step_name]
                    # Convert to RGB if needed for display
                    if len(step_img.shape) == 2:  # Grayscale
                        display_img = step_img
                    else:
                        display_img = cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB)
                    st.image(display_img, caption=step_name, use_container_width=True)

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Upload a clear image of pills
2. Choose detection method:
   - Adaptive: Good for varying lighting
   - Canny: Good for edge detection
   - Otsu: Automatic thresholding
   - HSV: Color-based segmentation
3. Adjust parameters as needed
4. Click 'Count Pills'
""")

# Debug info
st.sidebar.header("Debug Info")
if uploaded_file and st.session_state.pill_count == 0:
    st.sidebar.warning("""
    No pills detected. Try:
    1. Different detection method
    2. Adjusting area thresholds
    3. Improving image contrast
    4. Using a plain background
    """)
