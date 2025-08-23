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
        pixels = image[mask > 0]
    else:
        pixels = image.reshape(-1, 3)
    
    if len(pixels) == 0:
        return [0, 0, 0]
    
    if len(pixels) < k:
        k = max(1, len(pixels))
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
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
    elif 169 < h <= 180: return "Red"  # Handle red wrap-around
    else: return "Other"

def get_shape_name(contour):
    """Improved shape detection with better accuracy"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(contour)
    
    if area < 200:  # Too small to classify reliably
        return "Small Object"
    
    # Calculate shape properties
    aspect_ratio = w / float(h)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Shape classification with better thresholds
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        if 0.85 <= aspect_ratio <= 1.15 and solidity > 0.9:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) > 6:
        if 0.85 <= aspect_ratio <= 1.15 and solidity > 0.85:
            return "Circle"
        else:
            return "Oval"
    else:
        return "Irregular"

# ====================================================================
# 2. Advanced Automatic Pill Detection
# ====================================================================

def auto_detect_pills(image, min_area=500, max_area=50000):
    """Automatically detect pills in the entire image using multiple strategies"""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Multiple detection strategies
    masks = []
    
    # 1. Edge detection with adaptive parameters
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    masks.append(edges)
    
    # 2. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    masks.append(adaptive)
    
    # 3. Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu)
    
    # 4. Color-based segmentation for various pill colors
    # Light pills (white, yellow, light colors)
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 80, 255])
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    
    # Dark pills (blue, green, red, dark colors)
    lower_dark = np.array([0, 40, 40])
    upper_dark = np.array([180, 255, 200])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Very dark pills (black, dark gray)
    lower_vdark = np.array([0, 0, 0])
    upper_vdark = np.array([180, 255, 50])
    mask_vdark = cv2.inRange(hsv, lower_vdark, upper_vdark)
    
    color_mask = cv2.bitwise_or(mask_light, mask_dark)
    color_mask = cv2.bitwise_or(color_mask, mask_vdark)
    masks.append(color_mask)
    
    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Advanced morphological operations
    kernel = np.ones((3,3), np.uint8)
    
    # Remove noise
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes and connect nearby regions
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    # Apply median blur to smooth edges
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_data = []
    output_image = original.copy()
    valid_contours = []
    
    # Filter and process contours
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            valid_contours.append(cnt)
            
            # Create mask for this pill
            pill_mask = np.zeros_like(gray)
            cv2.drawContours(pill_mask, [cnt], -1, 255, -1)
            
            # Get dominant color
            dominant_color = get_dominant_color(image, pill_mask)
            color_name = get_color_name(dominant_color)
            
            # Get shape
            shape = get_shape_name(cnt)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            pill_data.append({
                "Pill ID": len(pill_data) + 1,
                "Color": color_name,
                "Shape": shape,
                "Area (px)": int(area),
                "Width": w,
                "Height": h,
                "Position": f"({x},{y})"
            })
    
    # Draw all valid contours
    for i, cnt in enumerate(valid_contours):
        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(output_image, str(i + 1), (cX - 15, cY), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return output_image, pill_data, cleaned_mask

# ====================================================================
# 3. Streamlit UI with Automatic Detection
# ====================================================================

st.set_page_config(layout="wide", page_title="Auto Pill Detector")
st.title("💊 Automatic Pill Detection System")
st.write("Upload an image to automatically detect and analyze pills across the entire image")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.header("📷 Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file:
        st.header("⚙️ Detection Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            min_area = st.slider("Min Size", 100, 2000, 300, 
                               help="Minimum pill area in pixels")
        with col2:
            max_area = st.slider("Max Size", 1000, 100000, 20000,
                               help="Maximum pill area in pixels")
        
        st.header("🎯 Detection Mode")
        detection_mode = st.radio("Select detection mode:", 
                                ["Auto (Whole Image)", "Manual ROI"], 
                                help="Auto mode scans the entire image automatically")
        
        if detection_mode == "Manual ROI":
            image_pil = Image.open(uploaded_file)
            img_w, img_h = image_pil.size
            
            st.subheader("Manual ROI Selection")
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input("X", 0, img_w, 0)
                y = st.number_input("Y", 0, img_h, 0)
            with col2:
                w = st.number_input("Width", 100, img_w, min(400, img_w))
                h = st.number_input("Height", 100, img_h, min(400, img_h))
        
        if st.button("🚀 Detect Pills", type="primary", use_container_width=True):
            with st.spinner("Scanning image for pills..."):
                image_pil = Image.open(uploaded_file)
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                
                if detection_mode == "Auto (Whole Image)":
                    processed_img, results, detection_mask = auto_detect_pills(image_bgr, min_area, max_area)
                else:
                    # For manual ROI, crop the image first
                    roi = (x, y, w, h)
                    cropped = image_bgr[y:y+h, x:x+w]
                    if cropped.size == 0:
                        st.error("Invalid ROI selection!")
                    else:
                        processed_img, results, detection_mask = auto_detect_pills(cropped, min_area, max_area)
                
                st.session_state.processed = True
                st.session_state.results = results
                st.session_state.processed_img = processed_img
                st.session_state.detection_mask = detection_mask
                st.session_state.original_img = image_bgr

# Main content
if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(uploaded_file, use_column_width=True, caption="Uploaded Image")
    
    with col2:
        if st.session_state.processed:
            st.subheader("🔍 Detection Results")
            
            if st.session_state.results:
                st.success(f"✅ Found {len(st.session_state.results)} pills")
                st.image(st.session_state.processed_img, channels="BGR", 
                        use_column_width=True, caption="Detected Pills (Green outlines)")
                
                # Display detection mask for debugging
                with st.expander("🔧 Show Detection Mask"):
                    st.image(st.session_state.detection_mask, 
                            caption="Computer Vision Mask Used for Detection", 
                            use_column_width=True)
                
                # Display results
                df = pd.DataFrame(st.session_state.results)
                
                st.subheader("📊 Pill Analysis Summary")
                cols = st.columns(4)
                metrics = [
                    ("Total Pills", len(df)),
                    ("Unique Colors", df['Color'].nunique()),
                    ("Unique Shapes", df['Shape'].nunique()),
                    ("Avg Size", f"{df['Area (px)'].mean():.0f} px")
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)
                
                st.subheader("📋 Detailed Results")
                st.dataframe(df, use_container_width=True)
                
                # Distributions
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎨 Color Distribution")
                    color_counts = df['Color'].value_counts()
                    if not color_counts.empty:
                        st.bar_chart(color_counts)
                
                with col2:
                    st.subheader("🔷 Shape Distribution")
                    shape_counts = df['Shape'].value_counts()
                    if not shape_counts.empty:
                        st.bar_chart(shape_counts)
                
            else:
                st.warning("❌ No pills detected in the image")
                st.info("""
                **Try these adjustments:**
                - Decrease the Minimum Size setting
                - Increase the Maximum Size setting
                - Ensure good lighting and contrast
                - Try a different image with clearer pill boundaries
                """)
                
                if hasattr(st.session_state, 'detection_mask'):
                    with st.expander("🔧 Show Detection Analysis"):
                        st.image(st.session_state.detection_mask,
                               caption="What the computer sees",
                               use_column_width=True)
                        st.info("The detection mask shows what features were found. Adjust parameters to improve detection.")

else:
    st.info("👆 Please upload an image to begin automatic pill detection")

# Tips section
with st.expander("💡 Expert Tips for Best Results"):
    st.markdown("""
    **For optimal automatic detection:**
    
    🌟 **Image Quality:**
    - Use high-resolution images (minimum 1000x1000 pixels)
    - Ensure even lighting without shadows
    - Maintain good focus on all pills
    
    🎨 **Background Recommendations:**
    - Solid color backgrounds work best (white, black, gray)
    - Avoid patterns, textures, or busy backgrounds
    - Ensure color contrast between pills and background
    
    ⚡ **Lighting Conditions:**
    - Use diffused, even lighting from multiple angles
    - Avoid direct light that creates reflections
    - Remove shadows by using light tents or softboxes
    
    ⚙️ **Parameter Guidance:**
    - **Min Size**: Start with 300px, decrease if small pills are missed
    - **Max Size**: Start with 20000px, increase for very large pills
    - Use "Show Detection Mask" to understand what's being detected
    
    🔧 **Troubleshooting:**
    - If pills are missed: Decrease Min Size
    - If background is detected: Increase Min Size
    - If detection is poor: Try different background color
    """)

# Footer
st.markdown("---")
st.caption("Auto Pill Detection System v3.0 | Advanced computer vision for reliable pill detection across various backgrounds")
