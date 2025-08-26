import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from sklearn.cluster import KMeans

# ====================================================================
# 1. Enhanced Helper Functions with Debugging
# ====================================================================

def apply_morphological_operations(mask, operation_level=3):
    """Apply morphological operations with adjustable intensity"""
    # Create kernels
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    cleaned = mask.copy()
    
    # Adjust operations based on level
    if operation_level >= 1:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=2)
    if operation_level >= 2:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    if operation_level >= 3:
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)
        cleaned = cv2.medianBlur(cleaned, 5)
    if operation_level >= 4:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    return cleaned

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
    
    try:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]
        return dominant_color.astype(int)
    except:
        return [0, 0, 0]

def get_color_name(bgr_color):
    """Convert BGR color to descriptive name using HSV"""
    try:
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
    except:
        return "Unknown"

def get_shape_name(contour):
    """Improved shape detection"""
    try:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)
        
        if area < 100:
            return "Small"
        
        aspect_ratio = w / float(h)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if len(approx) == 3:
            return "Triangle"
        elif len(approx) == 4:
            if 0.9 <= aspect_ratio <= 1.1 and solidity > 0.85:
                return "Square"
            else:
                return "Rectangle"
        elif len(approx) > 6:
            if 0.85 <= aspect_ratio <= 1.15 and solidity > 0.8:
                return "Circle"
            else:
                return "Oval"
        else:
            return "Irregular"
    except:
        return "Unknown"

# ====================================================================
# 2. Advanced Pill Detection with Multiple Strategies
# ====================================================================

def detect_pills_advanced(image, min_area=300, max_area=20000, sensitivity=5, morph_level=3):
    """Advanced pill detection with adjustable parameters"""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjust parameters based on sensitivity
    canny_low = max(10, 50 - sensitivity * 5)
    canny_high = max(50, 150 - sensitivity * 10)
    adaptive_size = 11 + sensitivity * 2
    adaptive_c = 2 + sensitivity
    
    # Multiple detection strategies
    masks = []
    
    # Strategy 1: Edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    masks.append(edges)
    
    # Strategy 2: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, adaptive_size, adaptive_c)
    masks.append(adaptive)
    
    # Strategy 3: Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu)
    
    # Strategy 4: Color-based segmentation
    # Wider color ranges for better detection
    lower_light = np.array([0, 0, 100])
    upper_light = np.array([180, 80, 255])
    lower_color = np.array([0, 30, 30])
    upper_color = np.array([180, 255, 255])
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    
    mask_light = cv2.inRange(hsv, lower_light, upper_light)
    mask_color = cv2.inRange(hsv, lower_color, upper_color)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    color_mask = cv2.bitwise_or(mask_light, mask_color)
    color_mask = cv2.bitwise_or(color_mask, mask_dark)
    masks.append(color_mask)
    
    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Apply morphological operations
    final_mask = apply_morphological_operations(combined_mask, morph_level)
    
    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_data = []
    output_image = original.copy()
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
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
            
            # Draw on output image
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, str(len(pill_data)), (cX - 15, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return output_image, pill_data, final_mask, combined_mask

# ====================================================================
# 3. Streamlit UI with Advanced Controls
# ====================================================================

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'debug_masks' not in st.session_state:
    st.session_state.debug_masks = None

st.set_page_config(layout="wide", page_title="Pill Detector Pro")
st.title("💊 Advanced Pill Detection System")
st.write("Upload an image and adjust parameters for optimal detection")

# Main layout
col1, col2 = st.columns([1, 1])

with st.sidebar:
    st.header("📷 Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file:
        st.header("⚙️ Advanced Detection Settings")
        
        # Basic parameters
        min_area = st.slider("Minimum Pill Size", 50, 2000, 100, 
                           help="Minimum area in pixels for pill detection")
        max_area = st.slider("Maximum Pill Size", 500, 50000, 10000,
                           help="Maximum area in pixels for pill detection")
        
        # Advanced parameters
        st.subheader("Advanced Parameters")
        sensitivity = st.slider("Detection Sensitivity", 1, 10, 5,
                              help="Higher = more sensitive, may detect more noise")
        morph_level = st.slider("Morphological Level", 1, 5, 3,
                              help="Intensity of image cleaning operations")
        
        st.header("🎯 Detection Mode")
        detection_mode = st.radio("Select mode:", 
                                ["Full Image Analysis", "Manual ROI"])
        
        roi_coords = None
        if detection_mode == "Manual ROI":
            st.info("Enter coordinates for ROI")
            col1, col2 = st.columns(2)
            with col1:
                roi_x = st.number_input("X", 0, 2000, 0)
                roi_y = st.number_input("Y", 0, 2000, 0)
            with col2:
                roi_w = st.number_input("Width", 100, 2000, 300)
                roi_h = st.number_input("Height", 100, 2000, 300)
            roi_coords = (roi_x, roi_y, roi_w, roi_h)
        
        if st.button("🚀 Detect Pills", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with advanced algorithms..."):
                try:
                    # Load image
                    image_pil = Image.open(uploaded_file)
                    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.original_image = image_bgr
                    
                    # Apply ROI if selected
                    if detection_mode == "Manual ROI" and roi_coords:
                        x, y, w, h = roi_coords
                        roi_image = image_bgr[y:y+h, x:x+w].copy()
                        if roi_image.size == 0:
                            st.error("Invalid ROI selection!")
                        else:
                            processed_img, results, final_mask, combined_mask = detect_pills_advanced(
                                roi_image, min_area, max_area, sensitivity, morph_level
                            )
                            st.session_state.processed_image = processed_img
                            st.session_state.detection_results = results
                            st.session_state.debug_masks = {
                                'final_mask': final_mask,
                                'combined_mask': combined_mask
                            }
                    else:
                        # Full image analysis
                        processed_img, results, final_mask, combined_mask = detect_pills_advanced(
                            image_bgr, min_area, max_area, sensitivity, morph_level
                        )
                        st.session_state.processed_image = processed_img
                        st.session_state.detection_results = results
                        st.session_state.debug_masks = {
                            'final_mask': final_mask,
                            'combined_mask': combined_mask
                        }
                        
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

# Display results
if uploaded_file:
    with col1:
        st.subheader("📸 Original Image")
        if st.session_state.original_image is not None:
            st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), 
                    use_container_width=True, caption="Original Image")

    with col2:
        if st.session_state.processed_image is not None:
            st.subheader("🔍 Detection Results")
            
            if st.session_state.detection_results:
                total_pills = len(st.session_state.detection_results)
                st.success(f"✅ Found {total_pills} pills")
                
                # Show processed image
                st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), 
                        use_container_width=True, caption="Detected Pills")
                
                # Create categorized results
                df = pd.DataFrame(st.session_state.detection_results)
                
                # Group by color and shape
                summary_df = df.groupby(['Color', 'Shape']).size().reset_index(name='Quantity')
                summary_df = summary_df.sort_values('Quantity', ascending=False)
                
                st.subheader("📊 Categorized Results")
                st.dataframe(summary_df, use_container_width=True)
                
                # Show individual results
                with st.expander("📋 Detailed Results"):
                    st.dataframe(df, use_container_width=True)
                
                # Statistics
                st.subheader("📈 Statistics")
                cols = st.columns(4)
                metrics = [
                    ("Total Pills", total_pills),
                    ("Unique Colors", df['Color'].nunique()),
                    ("Unique Shapes", df['Shape'].nunique()),
                    ("Avg Size", f"{df['Area (px)'].mean():.0f} px")
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    col.metric(label, value)
                
            else:
                st.warning("❌ No pills detected")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.info("""
                    **Try these adjustments:**
                    1. ➕ Increase Sensitivity
                    2. ➖ Decrease Minimum Pill Size
                    3. 🔄 Adjust Morphological Level
                    4. 🎯 Try Manual ROI mode
                    5. 💡 Ensure good lighting in image
                    """)
                    
                    if st.session_state.debug_masks:
                        st.write("**Detection Masks:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(st.session_state.debug_masks['combined_mask'], 
                                    caption="Initial Combined Mask", use_container_width=True)
                        with col2:
                            st.image(st.session_state.debug_masks['final_mask'], 
                                    caption="Final Mask After Processing", use_container_width=True)

# Tips section
with st.expander("💡 Troubleshooting Guide"):
    st.markdown("""
    **If pills are not being detected:**
    
    1. **Adjust Sensitivity**: Increase to detect more objects
    2. **Size Settings**: 
       - Decrease Minimum Pill Size for small pills
       - Increase Maximum Pill Size for large pills
    3. **Morphological Level**: 
       - Increase for cleaner detection (good for noisy images)
       - Decrease for more sensitive detection
    4. **Try Manual ROI**: Select specific areas with pills
    5. **Image Quality**: Ensure good contrast and lighting
    
    **Common Issues:**
    - Pills too small → Decrease Minimum Pill Size
    - Background detected as pills → Increase Minimum Pill Size
    - Pills not detected → Increase Sensitivity
    - Over-detection → Decrease Sensitivity
    """)

st.markdown("---")
st.caption("Pill Detection System | Advanced computer vision with adjustable parameters")
