import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io

# Initialize session state
if 'pill_count' not in st.session_state:
    st.session_state.pill_count = 0
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

st.title("💊 Accurate Pill Counter with ROI Drawing")
st.write("Upload an image, draw an area to count, and get precise results")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read and store the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.original_image = image.copy()
    st.session_state.image_uploaded = True
    
    # Display original image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
             caption="Uploaded Image", use_container_width=True)

def auto_detect_pills(image):
    """Automatically detect pills using optimized parameters"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter (preserves edges while reducing noise)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Otsu's thresholding for better results
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return gray, blurred, thresh, cleaned, contours

def count_pills_in_contours(contours):
    """Count pills from contours with smart size filtering"""
    pill_count = 0
    pill_data = []
    areas = []
    
    # First pass: collect all areas
    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)
    
    # Determine size thresholds based on median area
    if areas:
        median_area = np.median(areas)
        min_area = median_area * 0.3  # 30% of median size
        max_area = median_area * 3.0  # 300% of median size
    else:
        min_area, max_area = 100, 5000  # Default values
    
    # Second pass: count pills with size filtering
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
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
    
    return pill_count, pill_data, min_area, max_area

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

# Drawing interface
if st.session_state.image_uploaded and st.session_state.original_image is not None:
    st.subheader("🔴 Draw Area to Count")
    
    # Get image dimensions
    img_height, img_width = st.session_state.original_image.shape[:2]
    
    # Create columns for drawing interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Click the buttons to add points at default positions, then adjust coordinates below")
        
        # Display image with current points
        display_image = st.session_state.original_image.copy()
        
        # Draw points and lines if we have them
        if st.session_state.roi_points:
            for i, point in enumerate(st.session_state.roi_points):
                x, y = point
                cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
                if i > 0:
                    prev_point = st.session_state.roi_points[i-1]
                    cv2.line(display_image, prev_point, point, (0, 0, 255), 2)
            
            # Connect first and last points if we have a polygon
            if len(st.session_state.roi_points) > 2:
                cv2.line(display_image, st.session_state.roi_points[-1], 
                         st.session_state.roi_points[0], (0, 0, 255), 2)
        
        st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), 
                 use_container_width=True, caption="Image with drawn points")
    
    with col2:
        st.write("**Point Management**")
        
        # Add point buttons
        if st.button("Add Top-Left Point"):
            st.session_state.roi_points.append((50, 50))
            st.rerun()
            
        if st.button("Add Top-Right Point"):
            st.session_state.roi_points.append((img_width - 50, 50))
            st.rerun()
            
        if st.button("Add Bottom-Left Point"):
            st.session_state.roi_points.append((50, img_height - 50))
            st.rerun()
            
        if st.button("Add Bottom-Right Point"):
            st.session_state.roi_points.append((img_width - 50, img_height - 50))
            st.rerun()
        
        # Point management
        if st.session_state.roi_points:
            st.write(f"**Current Points ({len(st.session_state.roi_points)})**")
            
            # Point editor
            for i, point in enumerate(st.session_state.roi_points):
                col_x, col_y = st.columns(2)
                with col_x:
                    new_x = st.number_input(f"Point {i+1} X", value=point[0], 
                                          min_value=0, max_value=img_width, key=f"x_{i}")
                with col_y:
                    new_y = st.number_input(f"Point {i+1} Y", value=point[1], 
                                          min_value=0, max_value=img_height, key=f"y_{i}")
                
                # Update point if changed
                if new_x != point[0] or new_y != point[1]:
                    st.session_state.roi_points[i] = (new_x, new_y)
                    st.rerun()
            
            if st.button("Remove Last Point"):
                st.session_state.roi_points.pop()
                st.rerun()
                
            if st.button("Clear All Points"):
                st.session_state.roi_points = []
                st.rerun()
        
        # Count pills button
        if st.session_state.roi_points and len(st.session_state.roi_points) >= 3:
            if st.button("🎯 Count Pills in Selected Area", type="primary"):
                with st.spinner("Counting pills in selected area..."):
                    # Apply ROI mask
                    masked_image, roi_mask = apply_roi_mask(
                        st.session_state.original_image, st.session_state.roi_points
                    )
                    
                    # Detect pills in the masked area
                    gray, blurred, thresh, cleaned, contours = auto_detect_pills(masked_image)
                    
                    # Count pills
                    pill_count, pill_data, min_area, max_area = count_pills_in_contours(contours)
                    
                    # Create result image
                    result_image = st.session_state.original_image.copy()
                    
                    # Draw ROI area
                    points_array = np.array(st.session_state.roi_points, dtype=np.int32)
                    cv2.polylines(result_image, [points_array], True, (0, 0, 255), 2)
                    
                    # Draw detected pills
                    for i, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if min_area < area < max_area:
                            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                            
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(result_image, str(i+1), (cX - 10, cY - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Display results
                    st.subheader("📊 Results")
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                                caption=f"Detected {pill_count} pills in selected area", 
                                use_container_width=True)
                    
                    with result_col2:
                        st.metric("Pills in Selected Area", pill_count)
                        st.write(f"Size range: {int(min_area)} to {int(max_area)} pixels")
                        
                        if pill_data:
                            df = pd.DataFrame(pill_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No pills detected in the selected area.")
                    
                    # Show processing steps
                    with st.expander("View Processing Steps"):
                        steps_col1, steps_col2 = st.columns(2)
                        
                        with steps_col1:
                            st.image(gray, caption="Grayscale", use_container_width=True)
                            st.image(thresh, caption="Thresholded", use_container_width=True)
                        
                        with steps_col2:
                            st.image(blurred, caption="Blurred", use_container_width=True)
                            st.image(cleaned, caption="Cleaned", use_container_width=True)

# Full image analysis option
if st.session_state.image_uploaded and st.session_state.original_image is not None:
    st.subheader("Full Image Analysis")
    
    if st.button("🔍 Count All Pills in Image"):
        with st.spinner("Analyzing entire image..."):
            # Detect pills
            gray, blurred, thresh, cleaned, contours = auto_detect_pills(st.session_state.original_image)
            
            # Count pills
            pill_count, pill_data, min_area, max_area = count_pills_in_contours(contours)
            
            # Create result image
            result_image = st.session_state.original_image.copy()
            
            # Draw detected pills
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(result_image, str(i+1), (cX - 10, cY - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display results
            st.subheader("📊 Results")
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                        caption=f"Detected {pill_count} pills", 
                        use_container_width=True)
            
            with result_col2:
                st.metric("Total Pills", pill_count)
                st.write(f"Size range: {int(min_area)} to {int(max_area)} pixels")
                
                if pill_data:
                    df = pd.DataFrame(pill_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No pills detected in the image.")

# Instructions
st.sidebar.header("📋 Instructions")
st.sidebar.info("""
**How to use:**
1. Upload an image of pills
2. Use buttons to add points at corners
3. Adjust point coordinates as needed
4. Create at least 3 points to form a polygon
5. Click "Count Pills in Selected Area"
6. Or use "Count All Pills" for full image analysis
""")

# Tips for better accuracy
st.sidebar.header("💡 Tips for Best Results")
st.sidebar.info("""
1. Use good lighting with minimal shadows
2. Place pills on contrasting background
3. Avoid overlapping pills when possible
4. Draw ROI around a well-defined area
5. Ensure pills are in focus and clearly visible
""")
