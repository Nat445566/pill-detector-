import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io

# ====================================================================
# 1. Improved Helper Functions for Color and Shape
# ====================================================================

def get_color_name(hsv_color):
    """Takes an HSV color value and returns a descriptive color name."""
    h, s, v = hsv_color
    if v < 50: return "Black"
    if s < 40 and v > 200: return "White"
    if s < 40 and 100 <= v <= 200: return "Gray"
    if (0 <= h <= 10) or (170 <= h <= 180): return "Red"
    elif 11 <= h <= 25: return "Orange"
    elif 26 <= h <= 35: return "Yellow"
    elif 36 <= h <= 85: return "Green"
    elif 86 <= h <= 125: return "Blue"
    elif 126 <= h <= 145: return "Purple"
    elif 146 <= h <= 169: return "Pink"
    else: return "Brown/Other"

def get_shape_name(contour):
    """Takes a contour and returns a shape name based on its geometry."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    if len(approx) > 8 and extent > 0.8:
        return "Circle" if 0.8 <= aspect_ratio <= 1.2 else "Oval"
    elif len(approx) == 4:
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif len(approx) == 3:
        return "Triangle"
    else:
        return "Irregular"

# ====================================================================
# 2. Improved Main Image Processing Function
# ====================================================================

def analyze_pills(image, roi, bg_threshold, min_area):
    """The main pipeline to detect, count, and analyze pills."""
    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w].copy()
    
    if cropped_image.size == 0:
        return cropped_image, []
    
    # Convert to different color spaces for better processing
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Try multiple segmentation methods
    methods = []
    
    # Method 1: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    methods.append(adaptive_thresh)
    
    # Method 2: Otsu's thresholding
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    methods.append(otsu_thresh)
    
    # Method 3: Color-based segmentation (for colored pills)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_colors = np.array([0, 40, 40])
    upper_colors = np.array([180, 255, 255])
    mask_colors = cv2.inRange(hsv, lower_colors, upper_colors)
    
    color_mask = cv2.bitwise_or(mask_white, mask_colors)
    methods.append(color_mask)
    
    # Combine all methods
    combined_mask = np.zeros_like(gray)
    for mask in methods:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Remove small noise
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_data = []
    output_image = cropped_image.copy()
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Create mask for this pill
            single_pill_mask = np.zeros_like(gray)
            cv2.drawContours(single_pill_mask, [cnt], -1, 255, -1)
            
            # Get average color
            mean_bgr = cv2.mean(cropped_image, mask=single_pill_mask)[:3]
            mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Get shape
            shape = get_shape_name(cnt)
            
            # Get bounding box info
            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
            
            pill_data.append({
                "Pill ID": i + 1,
                "Color": get_color_name(mean_hsv),
                "Shape": shape,
                "Area (px)": int(area),
                "Width": w_bb,
                "Height": h_bb
            })
            
            # Draw on output image
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, str(i + 1), (cX - 10, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return output_image, pill_data

# ====================================================================
# 3. Streamlit User Interface
# ====================================================================

st.set_page_config(layout="wide")
st.title("💊 Advanced Pill Detector Pro")
st.write("Upload an image and specify the region to analyze pills.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("1. Choose an image...", type=["jpg", "jpeg", "png"])
    
    st.subheader("Detection Parameters")
    min_pill_area = st.slider("Minimum Pill Area (pixels)", 100, 5000, 500, help="Adjust based on pill size")
    bg_std_threshold = st.slider("Background Sensitivity", 1.0, 50.0, 20.0, help="Lower for uniform backgrounds")
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        img_w, img_h = image_pil.size
        
        st.subheader("ROI Selection")
        col_a, col_b = st.columns(2)
        with col_a:
            x = st.slider("X coordinate", 0, img_w, 0)
            y = st.slider("Y coordinate", 0, img_h, 0)
        with col_b:
            w = st.slider("Width", 50, img_w - x, min(400, img_w - x))
            h = st.slider("Height", 50, img_h - y, min(400, img_h - y))
        
        roi = (x, y, w, h)
        
        if st.button("🔍 Analyze Pills", type="primary"):
            with st.spinner("Analyzing pills..."):
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                processed_image, pill_results = analyze_pills(image_bgr, roi, bg_std_threshold, min_pill_area)
                
                st.session_state.processed_image = processed_image
                st.session_state.pill_results = pill_results
                st.session_state.roi = roi

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Original Image")
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        
        if 'roi' in st.session_state:
            # Draw ROI rectangle on the image
            draw_image = np.array(image_pil.copy())
            roi = st.session_state.roi
            cv2.rectangle(draw_image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), 
                         (255, 0, 0), 3)
            st.image(draw_image, caption=f"Selected ROI", use_column_width=True)

    with col2:
        if 'processed_image' in st.session_state and 'pill_results' in st.session_state:
            st.subheader("Detection Results")
            
            if st.session_state.pill_results:
                st.success(f"✅ Detected {len(st.session_state.pill_results)} pills")
                st.image(st.session_state.processed_image, channels="BGR", 
                        use_column_width=True, caption="Processed Image with Detections")
                
                st.subheader("📊 Pill Analysis")
                df = pd.DataFrame(st.session_state.pill_results)
                st.dataframe(df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pills", len(df))
                with col2:
                    st.metric("Unique Colors", df['Color'].nunique())
                with col3:
                    st.metric("Unique Shapes", df['Shape'].nunique())
                
            else:
                st.warning("❌ No pills detected in the selected ROI")
                st.info("""
                **Tips to improve detection:**
                - Adjust the ROI to focus on pill area
                - Increase/decrease Minimum Pill Area
                - Try different Background Sensitivity
                - Ensure good lighting and contrast
                """)
                
                # Show the processed image even if no pills detected
                st.image(st.session_state.processed_image, channels="BGR", 
                        use_column_width=True, caption="Processed Image")

else:
    st.info("📸 Please upload an image to begin analysis")

# Add some tips
with st.expander("💡 Tips for better detection"):
    st.write("""
    1. **Good Lighting**: Ensure the image has even lighting without shadows
    2. **Contrast**: Pills should contrast well with the background
    3. **Focus**: Image should be in focus and not blurry
    4. **ROI Selection**: Select a region that contains only pills
    5. **Parameter Tuning**: Adjust Minimum Pill Area based on actual pill size
    6. **Background**: Use a plain, contrasting background for best results
    """)
