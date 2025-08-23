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
    elif 169 < h <= 180: return "Red"
    else: return "Other"

def get_shape_name(contour):
    """Improved shape detection with better accuracy"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(contour)
    
    if area < 200:
        return "Small Object"
    
    aspect_ratio = w / float(h)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
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

def auto_detect_pills(image, min_area=300, max_area=20000):
    """Automatically detect pills in the entire image"""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Multiple detection strategies
    masks = []
    
    # 1. Edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    masks.append(edges)
    
    # 2. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 5)
    masks.append(adaptive)
    
    # 3. Color-based segmentation
    lower_light = np.array([0, 0, 150])
    upper_light = np.array([180, 80, 255])
    lower_dark = np.array([0, 40, 40])
    upper_dark = np.array([180, 255, 200])
    lower_vdark = np.array([0, 0, 0])
    upper_vdark = np.array([180, 255, 50])
    
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_vdark = cv2.inRange(hsv, lower_vdark, upper_vdark)
    
    color_mask = cv2.bitwise_or(mask_light, mask_dark)
    color_mask = cv2.bitwise_or(color_mask, mask_vdark)
    masks.append(color_mask)
    
    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_data = []
    output_image = original.copy()
    individual_pills = []  # Store individual pill images
    
    # Filter and process contours
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Extract individual pill
            pill_roi = original[y:y+h, x:x+w].copy()
            
            # Create mask for this pill
            pill_mask = np.zeros_like(gray)
            cv2.drawContours(pill_mask, [cnt], -1, 255, -1)
            pill_mask_roi = pill_mask[y:y+h, x:x+w]
            
            # Apply mask to get only the pill
            pill_only = cv2.bitwise_and(pill_roi, pill_roi, mask=pill_mask_roi)
            
            # Get dominant color
            dominant_color = get_dominant_color(pill_only)
            color_name = get_color_name(dominant_color)
            
            # Get shape
            shape = get_shape_name(cnt)
            
            pill_data.append({
                "Pill ID": len(pill_data) + 1,
                "Color": color_name,
                "Shape": shape,
                "Area (px)": int(area),
                "Width": w,
                "Height": h,
                "Position": f"({x},{y})"
            })
            
            # Draw on output image
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, str(len(pill_data)), (cX - 15, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Store individual pill image with annotation
            pill_with_text = pill_only.copy()
            cv2.putText(pill_with_text, f"#{len(pill_data)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            individual_pills.append(pill_with_text)
    
    return output_image, pill_data, cleaned_mask, individual_pills

# ====================================================================
# 3. Streamlit UI with Individual Pill Display
# ====================================================================

st.set_page_config(layout="wide", page_title="Pill Detector Pro")
st.title("💊 Advanced Pill Detection System")
st.write("Upload an image to automatically detect and analyze individual pills")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'individual_pills' not in st.session_state:
    st.session_state.individual_pills = None

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
            max_area = st.slider("Max Size", 1000, 50000, 20000,
                               help="Maximum pill area in pixels")
        
        if st.button("🚀 Detect Pills", type="primary", use_container_width=True):
            with st.spinner("Analyzing image for individual pills..."):
                image_pil = Image.open(uploaded_file)
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                
                processed_img, results, detection_mask, individual_pills = auto_detect_pills(
                    image_bgr, min_area, max_area
                )
                
                st.session_state.processed = True
                st.session_state.results = results
                st.session_state.processed_img = processed_img
                st.session_state.detection_mask = detection_mask
                st.session_state.individual_pills = individual_pills
                st.session_state.original_img = image_bgr

# Main content
if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(uploaded_file, use_container_width=True, caption="Uploaded Image")
    
    with col2:
        if st.session_state.processed:
            st.subheader("🔍 Detection Results")
            
            if st.session_state.results:
                st.success(f"✅ Found {len(st.session_state.results)} individual pills")
                st.image(st.session_state.processed_img, channels="BGR", 
                        use_container_width=True, caption="Detected Pills with Numbering")
                
                # Display individual pills in a grid
                st.subheader("🧪 Individual Pill Analysis")
                
                if st.session_state.individual_pills:
                    # Create columns for the pill grid
                    num_pills = len(st.session_state.individual_pills)
                    cols_per_row = 4
                    rows = (num_pills + cols_per_row - 1) // cols_per_row
                    
                    for row in range(rows):
                        cols = st.columns(cols_per_row)
                        for col_idx in range(cols_per_row):
                            pill_idx = row * cols_per_row + col_idx
                            if pill_idx < num_pills:
                                with cols[col_idx]:
                                    pill_img = st.session_state.individual_pills[pill_idx]
                                    pill_data = st.session_state.results[pill_idx]
                                    
                                    # Convert BGR to RGB for display
                                    pill_img_rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
                                    st.image(pill_img_rgb, 
                                            caption=f"Pill #{pill_idx + 1}: {pill_data['Color']} {pill_data['Shape']}",
                                            use_container_width=True)
                
                # Display results table
                st.subheader("📊 Analysis Summary")
                df = pd.DataFrame(st.session_state.results)
                
                summary_cols = st.columns(4)
                metrics = [
                    ("Total Pills", len(df)),
                    ("Unique Colors", df['Color'].nunique()),
                    ("Unique Shapes", df['Shape'].nunique()),
                    ("Avg Size", f"{df['Area (px)'].mean():.0f} px")
                ]
                
                for col, (label, value) in zip(summary_cols, metrics):
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
                
                # Debug view
                with st.expander("🔧 Detection Details"):
                    st.image(st.session_state.detection_mask, 
                            caption="Computer Vision Mask Used for Detection", 
                            use_container_width=True)
                    st.info("This shows the binary mask that was used to detect pill boundaries.")
                
            else:
                st.warning("❌ No pills detected in the image")
                st.info("""
                **Try these adjustments:**
                - Decrease the Minimum Size setting
                - Increase the Maximum Size setting
                - Ensure good lighting and contrast
                - Try a different image angle or background
                """)

else:
    st.info("👆 Please upload an image to begin pill detection")

# Tips section
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    **For optimal pill detection:**
    
    📸 **Image Quality:**
    - Use clear, high-resolution images
    - Ensure even lighting without shadows
    - Plain backgrounds work best (white, black, gray)
    
    ⚙️ **Settings Guidance:**
    - **Min Size**: Adjust if small pills are missed (decrease) or noise is detected (increase)
    - **Max Size**: Adjust for very large pills
    
    🎯 **Detection Features:**
    - Each pill is individually extracted and analyzed
    - Color and shape are determined for each pill
    - Pills are numbered for easy reference
    
    🔧 **Troubleshooting:**
    - If detection is poor, try different background colors
    - Ensure pills are well-separated in the image
    - Avoid overlapping pills for best results
    """)

st.markdown("---")
st.caption("Individual Pill Detection System | Advanced computer vision for precise pill analysis")
