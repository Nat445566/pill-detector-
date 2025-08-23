import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

# ====================================================================
# 1. Enhanced Helper Functions
# ====================================================================

def get_dominant_color(image, mask=None, k=3):
    """Get dominant color using K-means clustering"""
    if mask is not None:
        image = image[mask > 0]
    else:
        image = image.reshape(-1, 3)
    
    if len(image) == 0:
        return [0, 0, 0]
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(image)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)

def get_color_name(bgr_color):
    """Convert BGR color to descriptive name using HSV"""
    bgr_array = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
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
    else: return "Other"

def get_shape_name(contour):
    """Improved shape detection"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(contour)
    
    if area < 100:  # Too small to classify
        return "Small"
    
    # Calculate shape properties
    aspect_ratio = w / float(h)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Shape classification
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
    elif len(approx) > 8:
        if 0.8 <= aspect_ratio <= 1.2 and solidity > 0.85:
            return "Circle"
        else:
            return "Oval"
    else:
        return "Irregular"

# ====================================================================
# 2. Advanced Pill Detection with Multiple Strategies
# ====================================================================

def detect_pills_advanced(image, roi, min_area=300):
    """Advanced pill detection using multiple strategies"""
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w].copy()
    
    if cropped.size == 0:
        return cropped, []
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    
    # Strategy 1: Edge-based detection
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # Strategy 2: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Strategy 3: Color-based segmentation
    # Detect both light and dark pills
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 80, 255])
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    color_mask = cv2.bitwise_or(mask_light, mask_dark)
    
    # Strategy 4: Background subtraction (for uniform backgrounds)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bg_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine all strategies
    combined_mask = cv2.bitwise_or(edges, adaptive)
    combined_mask = cv2.bitwise_or(combined_mask, color_mask)
    combined_mask = cv2.bitwise_or(combined_mask, bg_mask)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.medianBlur(combined_mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_data = []
    output_image = cropped.copy()
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Create mask for this pill
            pill_mask = np.zeros_like(gray)
            cv2.drawContours(pill_mask, [cnt], -1, 255, -1)
            
            # Get dominant color
            dominant_color = get_dominant_color(cropped, pill_mask)
            color_name = get_color_name(dominant_color)
            
            # Get shape
            shape = get_shape_name(cnt)
            
            # Get bounding box
            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(cnt)
            
            pill_data.append({
                "Pill ID": i + 1,
                "Color": color_name,
                "Shape": shape,
                "Area": int(area),
                "Width": w_bb,
                "Height": h_bb,
                "Color RGB": f"RGB({dominant_color[2]}, {dominant_color[1]}, {dominant_color[0]})"
            })
            
            # Draw on output image
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, f"{i+1}", (cX - 10, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return output_image, pill_data

# ====================================================================
# 3. Streamlit UI
# ====================================================================

st.set_page_config(layout="wide", page_title="Advanced Pill Detector")
st.title("💊 Advanced Pill Detection System")
st.write("Upload an image to detect and analyze pills with various backgrounds")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'roi' not in st.session_state:
    st.session_state.roi = None

with st.sidebar:
    st.header("📷 Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file:
        st.header("⚙️ Detection Settings")
        min_area = st.slider("Minimum Pill Size", 100, 2000, 300, 
                           help="Adjust based on expected pill size in pixels")
        sensitivity = st.slider("Detection Sensitivity", 1, 10, 5,
                              help="Higher values detect more objects (may include noise)")
        
        st.header("🎯 ROI Selection")
        image_pil = Image.open(uploaded_file)
        img_w, img_h = image_pil.size
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("X", 0, img_w, 0)
            y = st.number_input("Y", 0, img_h, 0)
        with col2:
            w = st.number_input("Width", 100, img_w, min(400, img_w))
            h = st.number_input("Height", 100, img_h, min(400, img_h))
        
        roi = (x, y, w, h)
        
        if st.button("🚀 Detect Pills", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                processed_img, results = detect_pills_advanced(image_bgr, roi, min_area)
                
                st.session_state.processed = True
                st.session_state.results = results
                st.session_state.processed_img = processed_img
                st.session_state.roi = roi

# Main content
col1, col2 = st.columns(2)

if uploaded_file:
    with col1:
        st.subheader("📸 Original Image")
        st.image(uploaded_file, use_column_width=True)
        
        if st.session_state.roi:
            # Show ROI on image
            img_array = np.array(Image.open(uploaded_file))
            roi_img = img_array.copy()
            x, y, w, h = st.session_state.roi
            cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            st.image(roi_img, caption="Selected Region of Interest", use_column_width=True)

    with col2:
        if st.session_state.processed:
            st.subheader("🔍 Detection Results")
            
            if st.session_state.results:
                st.success(f"✅ Found {len(st.session_state.results)} pills")
                st.image(st.session_state.processed_img, channels="BGR", 
                        use_column_width=True, caption="Detected Pills")
                
                # Display results
                df = pd.DataFrame(st.session_state.results)
                
                st.subheader("📊 Pill Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pills", len(df))
                with col2:
                    st.metric("Unique Colors", df['Color'].nunique())
                with col3:
                    st.metric("Unique Shapes", df['Shape'].nunique())
                with col4:
                    avg_size = df['Area'].mean()
                    st.metric("Avg Size", f"{avg_size:.0f} px")
                
                st.subheader("📋 Detailed Results")
                st.dataframe(df.drop('Color RGB', axis=1), use_container_width=True)
                
                # Color distribution using Streamlit native charts
                st.subheader("🎨 Color Distribution")
                color_counts = df['Color'].value_counts()
                if not color_counts.empty:
                    st.bar_chart(color_counts)
                else:
                    st.info("No color data available")
                
                # Shape distribution
                st.subheader("🔷 Shape Distribution")
                shape_counts = df['Shape'].value_counts()
                if not shape_counts.empty:
                    st.bar_chart(shape_counts)
                else:
                    st.info("No shape data available")
                
            else:
                st.warning("❌ No pills detected")
                st.info("""
                **Try these adjustments:**
                - Move the ROI to better cover the pills
                - Decrease the Minimum Pill Size
                - Ensure good lighting and contrast
                - Try a different image angle
                """)
                
                if hasattr(st.session_state, 'processed_img'):
                    st.image(st.session_state.processed_img, channels="BGR",
                           caption="Processing result", use_column_width=True)

else:
    st.info("👆 Please upload an image to begin analysis")

# Tips section
with st.expander("💡 Pro Tips for Best Results"):
    st.markdown("""
    **For best detection results:**
    
    🎯 **ROI Selection:**
    - Select an area that contains only pills
    - Avoid including background objects
    
    ⚡ **Lighting Conditions:**
    - Use even, diffused lighting
    - Avoid shadows and reflections
    - Ensure good contrast between pills and background
    
    🎨 **Background Tips:**
    - Plain, neutral backgrounds work best
    - Avoid patterns and textures
    - Ensure color contrast with pills
    
    ⚙️ **Parameter Tuning:**
    - Start with default settings
    - Adjust Minimum Pill Size based on actual pill dimensions
    - Use higher sensitivity for faint or small pills
    """)

# Footer
st.markdown("---")
st.caption("Advanced Pill Detection System v2.0 | Handles various backgrounds and lighting conditions")
