import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.cluster import KMeans

# ====================================================================
# 1. Enhanced Helper Functions with Morphological Operations
# ====================================================================

def apply_morphological_operations(mask):
    """Apply advanced morphological operations to clean the mask"""
    # Create different kernels for various operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((7, 7), np.uint8)
    
    # Step 1: Remove small noise (opening)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Step 2: Fill holes and connect regions (closing)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    
    # Step 3: Dilate to ensure pill boundaries are connected
    cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)
    
    # Step 4: Apply median blur to smooth edges
    cleaned = cv2.medianBlur(cleaned, 5)
    
    # Step 5: Final closing to ensure solid regions
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
    """Improved shape detection with morphological consistency"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    area = cv2.contourArea(contour)
    
    if area < 150:
        return "Small"
    
    aspect_ratio = w / float(h)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Shape classification with morphological consistency
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

# ====================================================================
# 2. Advanced Pill Detection with Multiple Strategies
# ====================================================================

def detect_pills_advanced(image, min_area=300, max_area=20000):
    """Advanced pill detection with multiple strategies and morphological operations"""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Multiple detection strategies
    masks = []
    
    # Strategy 1: Edge detection with morphological enhancement
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    edges = cv2.Canny(blurred, 20, 80)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    masks.append(edges)
    
    # Strategy 2: Advanced adaptive thresholding
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 7)
    masks.append(adaptive)
    
    # Strategy 3: Comprehensive color segmentation
    # For light pills (white, yellow, light colors)
    lower_light = np.array([0, 0, 120])
    upper_light = np.array([180, 60, 255])
    
    # For colored pills (blue, green, red, etc.)
    lower_color = np.array([0, 40, 40])
    upper_color = np.array([180, 255, 220])
    
    # For dark pills (black, dark colors)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    
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
    
    # Apply advanced morphological operations
    final_mask = apply_morphological_operations(combined_mask)
    
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
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return output_image, pill_data, final_mask

# ====================================================================
# 3. Streamlit UI with Manual ROI Drawing
# ====================================================================

def get_image_download_link(img, filename="detected_pills.png"):
    """Generate a download link for the image"""
    buffered = io.BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">Download Result Image</a>'
    return href

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None

st.set_page_config(layout="wide", page_title="Advanced Pill Detector")
st.title("💊 Advanced Pill Detection System")
st.write("Upload an image and detect pills with manual ROI or full image analysis")

# Main layout
col1, col2 = st.columns([1, 1])

with st.sidebar:
    st.header("📷 Image Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file:
        st.header("⚙️ Detection Settings")
        
        min_area = st.slider("Minimum Pill Size", 100, 2000, 300, 
                           help="Minimum area in pixels for pill detection")
        max_area = st.slider("Maximum Pill Size", 1000, 50000, 20000,
                           help="Maximum area in pixels for pill detection")
        
        st.header("🎯 Detection Mode")
        detection_mode = st.radio("Select detection mode:", 
                                ["Full Image Analysis", "Manual ROI Selection"])
        
        if detection_mode == "Manual ROI Selection":
            st.info("Draw a rectangle around the area containing pills")
            roi_x = st.number_input("ROI X coordinate", 0, 1000, 0)
            roi_y = st.number_input("ROI Y coordinate", 0, 1000, 0)
            roi_width = st.number_input("ROI Width", 100, 1000, 300)
            roi_height = st.number_input("ROI Height", 100, 1000, 300)
            st.session_state.roi_coords = (roi_x, roi_y, roi_width, roi_height)
        else:
            st.session_state.roi_coords = None
        
        if st.button("🚀 Detect Pills", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with advanced morphological operations..."):
                # Load and process image
                image_pil = Image.open(uploaded_file)
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                st.session_state.original_image = image_bgr
                
                # Apply ROI if selected
                if st.session_state.roi_coords:
                    x, y, w, h = st.session_state.roi_coords
                    roi_image = image_bgr[y:y+h, x:x+w].copy()
                    if roi_image.size == 0:
                        st.error("Invalid ROI selection! Please adjust coordinates.")
                    else:
                        processed_img, results, mask = detect_pills_advanced(roi_image, min_area, max_area)
                        # Convert back to full image coordinates
                        for result in results:
                            result['Position'] = f"({x + int(result["Position"][1:-1].split(",")[0])},{y + int(result["Position"][1:-1].split(",")[1])})"
                        st.session_state.processed_image = processed_img
                        st.session_state.detection_results = results
                else:
                    # Full image analysis
                    processed_img, results, mask = detect_pills_advanced(image_bgr, min_area, max_area)
                    st.session_state.processed_image = processed_img
                    st.session_state.detection_results = results

# Display results
if uploaded_file:
    with col1:
        st.subheader("📸 Original Image")
        if st.session_state.original_image is not None:
            st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), 
                    use_container_width=True, caption="Original Image")
            
            if st.session_state.roi_coords:
                x, y, w, h = st.session_state.roi_coords
                roi_img = st.session_state.original_image.copy()
                cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), 
                        use_container_width=True, caption="Selected ROI Area")

    with col2:
        if st.session_state.processed_image is not None and st.session_state.detection_results is not None:
            st.subheader("🔍 Detection Results")
            
            if st.session_state.detection_results:
                # Display summary
                total_pills = len(st.session_state.detection_results)
                st.success(f"✅ Found {total_pills} pills")
                
                # Show processed image
                st.image(cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB), 
                        use_container_width=True, caption="Detected Pills (Green outlines)")
                
                # Create categorized results
                df = pd.DataFrame(st.session_state.detection_results)
                
                # Group by color and shape for summary
                summary_df = df.groupby(['Color', 'Shape']).size().reset_index(name='Quantity')
                summary_df = summary_df.sort_values('Quantity', ascending=False)
                
                st.subheader("📊 Categorized Results")
                st.dataframe(summary_df, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Analysis")
                st.dataframe(df, use_container_width=True)
                
                # Statistics
                st.subheader("📈 Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pills", total_pills)
                with col2:
                    st.metric("Unique Colors", df['Color'].nunique())
                with col3:
                    st.metric("Unique Shapes", df['Shape'].nunique())
                with col4:
                    avg_size = df['Area (px)'].mean()
                    st.metric("Avg Size", f"{avg_size:.0f} px")
                
                # Visualizations
                st.subheader("🎨 Distribution Charts")
                
                tab1, tab2 = st.tabs(["Color Distribution", "Shape Distribution"])
                
                with tab1:
                    color_counts = df['Color'].value_counts()
                    if not color_counts.empty:
                        st.bar_chart(color_counts)
                    else:
                        st.info("No color data available")
                
                with tab2:
                    shape_counts = df['Shape'].value_counts()
                    if not shape_counts.empty:
                        st.bar_chart(shape_counts)
                    else:
                        st.info("No shape data available")
                
            else:
                st.warning("❌ No pills detected")
                st.info("""
                **Try these adjustments:**
                - Decrease Minimum Pill Size
                - Adjust ROI area if using manual selection
                - Ensure good lighting and contrast
                - Try different background
                """)
else:
    st.info("👆 Please upload an image to begin analysis")

# Footer with tips
with st.expander("💡 Expert Tips"):
    st.markdown("""
    **For best results:**
    
    🎯 **Detection Modes:**
    - **Full Image Analysis**: Automatically scans entire image
    - **Manual ROI**: Draw specific area for focused analysis
    
    ⚡ **Morphological Operations:**
    - Advanced cleaning removes noise
    - Handles overlapping pills better
    - Works with various backgrounds
    
    🎨 **Color & Shape Detection:**
    - Uses K-means clustering for accurate color identification
    - Advanced geometric analysis for shape classification
    - Handles irregular shapes and various pill types
    
    ⚙️ **Parameter Guidance:**
    - **Min Size**: Adjust based on smallest pill size
    - **Max Size**: Adjust for largest pills
    - Use manual ROI for crowded or complex images
    """)

st.markdown("---")
st.caption("Advanced Pill Detection System v4.0 | Morphological Operations + Robust Background Handling")
