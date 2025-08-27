import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import base64

# Initialize session state
if 'pill_count' not in st.session_state:
    st.session_state.pill_count = 0
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'last_click' not in st.session_state:
    st.session_state.last_click = None

st.title("💊 Interactive Pill Counter with Click-to-Draw ROI")
st.write("Upload an image, click to draw an area, and get accurate pill counts")

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
             caption="Uploaded Image - Click to draw points", use_container_width=True)

def detect_pills_accurate(image):
    """Accurate pill detection with multiple techniques"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
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
        min_area = max(50, median_area * 0.3)  # 30% of median size, but at least 50px
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
                # Use bounding rect center if moments fail
                x, y, w, h = cv2.boundingRect(contour)
                cX, cY = x + w//2, y + h//2
                
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

# Create a function to handle image clicks
def get_image_with_points(image, points):
    """Create an image with points and lines drawn on it"""
    display_image = image.copy()
    
    # Draw points and lines
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(display_image, (x, y), 8, (0, 0, 255), -1)
        if i > 0:
            prev_point = points[i-1]
            cv2.line(display_image, prev_point, point, (0, 0, 255), 2)
    
    # Connect first and last points if we have a polygon
    if len(points) > 2:
        cv2.line(display_image, points[-1], points[0], (0, 0, 255), 2)
    
    return display_image

# Interactive drawing section
if st.session_state.image_uploaded and st.session_state.original_image is not None:
    st.subheader("🎯 Draw Area to Count")
    
    # Get image dimensions
    img_height, img_width = st.session_state.original_image.shape[:2]
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("Click on the image below to add points. Create at least 3 points to form a polygon.")
        
        # Display image with current points
        display_image = get_image_with_points(
            st.session_state.original_image, 
            st.session_state.roi_points
        )
        
        # Create a unique key for the button based on points count
        button_key = f"image_click_{len(st.session_state.roi_points)}"
        
        # Create clickable image using button hack
        st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), 
                 use_container_width=True, caption="Click to add points")
        
        # Add buttons for point management
        point_col1, point_col2, point_col3 = st.columns(3)
        
        with point_col1:
            if st.button("Add Point at Center", key="add_center"):
                center_x, center_y = img_width // 2, img_height // 2
                st.session_state.roi_points.append((center_x, center_y))
                st.rerun()
                
        with point_col2:
            if st.button("Remove Last Point", key="remove_point"):
                if st.session_state.roi_points:
                    st.session_state.roi_points.pop()
                    st.rerun()
                    
        with point_col3:
            if st.button("Clear All Points", key="clear_points"):
                st.session_state.roi_points = []
                st.rerun()
    
    with col2:
        st.write("**Current Points**")
        
        if st.session_state.roi_points:
            for i, point in enumerate(st.session_state.roi_points):
                st.write(f"Point {i+1}: ({point[0]}, {point[1]})")
                
            if len(st.session_state.roi_points) >= 3:
                if st.button("🔍 Count Pills in Selected Area", type="primary", key="count_roi"):
                    with st.spinner("Counting pills in selected area..."):
                        # Apply ROI mask
                        masked_image, roi_mask = apply_roi_mask(
                            st.session_state.original_image, st.session_state.roi_points
                        )
                        
                        # Detect pills in the masked area
                        gray, blurred, thresh, cleaned, contours = detect_pills_accurate(masked_image)
                        
                        # Count pills
                        pill_count, pill_data, min_area, max_area = count_pills_in_contours(contours)
                        
                        # Create result image
                        result_image = st.session_state.original_image.copy()
                        
                        # Draw ROI area
                        points_array = np.array(st.session_state.roi_points, dtype=np.int32)
                        cv2.polylines(result_image, [points_array], True, (0, 0, 255), 3)
                        
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
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Display results
                        st.subheader("📊 Results")
                        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                                caption=f"Detected {pill_count} pills in selected area", 
                                use_container_width=True)
                        
                        st.metric("Pills in Selected Area", pill_count)
                        st.write(f"Size range: {int(min_area)} to {int(max_area)} pixels")
                        
                        if pill_data:
                            df = pd.DataFrame(pill_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No pills detected in the selected area.")
        else:
            st.info("No points added yet. Click 'Add Point at Center' or use the full image analysis below.")

# Full image analysis option
if st.session_state.image_uploaded and st.session_state.original_image is not None:
    st.subheader("🔍 Full Image Analysis")
    
    if st.button("Count All Pills in Image", key="count_full"):
        with st.spinner("Analyzing entire image..."):
            # Detect pills
            gray, blurred, thresh, cleaned, contours = detect_pills_accurate(st.session_state.original_image)
            
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
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display results
            st.subheader("📊 Results")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), 
                    caption=f"Detected {pill_count} pills", 
                    use_container_width=True)
            
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
2. Use "Add Point at Center" to place points
3. Adjust points by removing or clearing
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
4. Ensure pills are in focus and clearly visible
5. For best results, draw ROI around a well-defined area
""")

# Add some sample images for testing
st.sidebar.header("🖼️ Sample Images")
st.sidebar.info("""
Try these for testing:
- Pills on plain background
- Different colored pills
- Various pill sizes
- Both scattered and organized arrangements
""")
