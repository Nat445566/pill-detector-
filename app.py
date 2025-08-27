import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Initialize session state
if 'pill_count' not in st.session_state:
    st.session_state.pill_count = 0
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = {}
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

st.title("💊 Smart Pill Counter with ROI Selection")
st.write("Upload an image, draw an area to count, and get accurate results")

# Sidebar for mode selection
mode = st.sidebar.radio("Select Mode", ["Full Image Analysis", "Draw ROI Area"])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], 
                                key="file_uploader")

def auto_detect_pills(image):
    """Automatically detect pills using optimized parameters"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement (helps with different lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter (preserves edges while reducing noise)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return gray, blurred, thresh, cleaned, contours

def count_pills_in_contours(contours, min_area=150, max_area=5000):
    """Count pills from contours with size filtering"""
    pill_count = 0
    pill_data = []
    areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
        
        # Auto-detect appropriate size range if not specified
        if len(areas) > 5:
            avg_area = np.median(areas)
            auto_min = avg_area * 0.3
            auto_max = avg_area * 3.0
        else:
            auto_min = min_area
            auto_max = max_area
            
        if auto_min < area < auto_max:
            pill_count += 1
            
            # Get contour properties
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            pill_data.append({
                'Pill #': pill_count,
                'Area': int(area),
                'Center X': cX,
                'Center Y': cY
            })
    
    return pill_count, pill_data

def apply_roi_mask(image, points):
    """Apply ROI mask to image based on drawn points"""
    if len(points) < 3:
        return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    
    # Apply mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image, mask

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image_uploaded = True
    
    # Store original image in session state for drawing
    if 'original_image' not in st.session_state:
        st.session_state.original_image = image.copy()
    
    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
             caption="Uploaded Image", use_container_width=True)
    
    if mode == "Draw ROI Area":
        st.info("🔴 Click the points to draw a polygon around the area you want to count. Double-click to complete the shape.")
        
        # Create a PIL image for drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Draw existing points if any
        if st.session_state.roi_points:
            for i, point in enumerate(st.session_state.roi_points):
                x, y = point
                draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='red')
                if i > 0:
                    prev_point = st.session_state.roi_points[i-1]
                    draw.line([prev_point, point], fill='red', width=2)
            
            # Connect first and last points if we have a polygon
            if len(st.session_state.roi_points) > 2:
                draw.line([st.session_state.roi_points[-1], st.session_state.roi_points[0]], 
                         fill='red', width=2)
        
        # Display image with drawn points
        st.image(pil_image, use_container_width=True, caption="Draw area to count")
        
        # Point selection
        if st.button("Add Point"):
            st.session_state.roi_points.append((100, 100))  # Default point
            st.rerun()
            
        if st.session_state.roi_points and st.button("Remove Last Point"):
            st.session_state.roi_points.pop()
            st.rerun()
            
        if st.session_state.roi_points and st.button("Clear All Points"):
            st.session_state.roi_points = []
            st.rerun()
            
        if st.session_state.roi_points and len(st.session_state.roi_points) >= 3:
            if st.button("Count Pills in Selected Area"):
                with st.spinner("Counting pills in selected area..."):
                    # Apply ROI mask
                    masked_image, roi_mask = apply_roi_mask(image, st.session_state.roi_points)
                    
                    # Detect pills in the masked area
                    gray, blurred, thresh, cleaned, contours = auto_detect_pills(masked_image)
                    
                    # Count pills
                    pill_count, pill_data = count_pills_in_contours(contours)
                    
                    # Create result image
                    result_image = image.copy()
                    
                    # Draw ROI area
                    points_array = np.array(st.session_state.roi_points, dtype=np.int32)
                    cv2.polylines(result_image, [points_array], True, (0, 0, 255), 2)
                    
                    # Draw detected pills
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if 150 < area < 5000:  # Reasonable pill size range
                            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                            
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(result_image, str(i+1), (cX - 10, cY - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                                caption=f"Detected {pill_count} pills in selected area", 
                                use_container_width=True)
                    
                    with col2:
                        st.metric("Pills in Selected Area", pill_count)
                        
                        if pill_data:
                            df = pd.DataFrame(pill_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No pills detected in the selected area.")
    
    else:  # Full Image Analysis mode
        if st.button("Count All Pills in Image"):
            with st.spinner("Analyzing entire image..."):
                # Detect pills
                gray, blurred, thresh, cleaned, contours = auto_detect_pills(image)
                
                # Count pills
                pill_count, pill_data = count_pills_in_contours(contours)
                
                # Create result image
                result_image = image.copy()
                
                # Draw detected pills
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if 150 < area < 5000:  # Reasonable pill size range
                        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                        
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(result_image, str(i+1), (cX - 10, cY - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                            caption=f"Detected {pill_count} pills", 
                            use_container_width=True)
                
                with col2:
                    st.metric("Total Pills", pill_count)
                    
                    if pill_data:
                        df = pd.DataFrame(pill_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No pills detected in the image.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.info("""
**How to use:**
1. Upload an image of pills
2. Choose mode:
   - **Full Image Analysis**: Count all pills
   - **Draw ROI Area**: Select specific area to count
3. For ROI mode:
   - Click "Add Point" to place points
   - Create a polygon around the area
   - Click "Count Pills in Selected Area"
4. View results and pill data
""")

# Tips for better accuracy
st.sidebar.header("Tips for Best Results")
st.sidebar.info("""
1. Use good lighting with minimal shadows
2. Place pills on contrasting background
3. Avoid overlapping pills when possible
4. For ROI mode, draw around a well-defined area
5. Ensure pills are in focus and clearly visible
""")

# Add some sample images for testing
st.sidebar.header("Sample Images")
st.sidebar.info("""
Try these for testing:
- Pills on plain background
- Different colored pills
- Various pill sizes
- Both scattered and organized arrangements
""")
