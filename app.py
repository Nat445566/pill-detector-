import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# --- Core Image Processing Functions ---

def auto_tune_parameters(image):
    """
    Analyzes the image to automatically estimate the best min/max area for pills.
    """
    # Use a pre-scan to find candidate objects
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate areas of all reasonably sized contours
    areas = [cv2.contourArea(c) for c in contours if 100 < cv2.contourArea(c) < 100000]
    
    if not areas:
        # Return default values if no candidates are found
        return {'min_area': 500, 'max_area': 40000}
        
    # Estimate parameters based on the median size of detected objects
    median_area = np.median(areas)
    min_area_guess = int(median_area * 0.2)  # 20% of median
    max_area_guess = int(median_area * 5.0)   # 500% of median
    
    # Clamp values to a reasonable range
    min_area_guess = max(100, min_area_guess)
    max_area_guess = min(100000, max_area_guess)
    
    return {'min_area': min_area_guess, 'max_area': max_area_guess}

def get_pill_properties(image_bgr, contour):
    """Analyzes a single pill contour to determine its shape and color."""
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    shape = "Unknown"
    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.85: shape = "Round"
        elif circularity > 0.60: shape = "Oval"
        else: shape = "Capsule"

    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=mask)[:3]
    h, s, v = mean_hsv
    
    color = "Unknown"
    if s < 65 and v > 150: color = "White"
    elif 35 < h < 85 and s > 60: color = "Green"
    elif (h < 12 or h > 168) and s > 80: color = "Red"
    elif 10 < h < 35 and s > 80: color = "Brown/Orange"
    elif 85 < h < 130 and s > 70: color = "Blue"
    
    return shape, color

def detect_pills(image, min_area, max_area, circularity_threshold, solidity_threshold):
    """The main pill detection pipeline: finds, filters, and analyzes contours."""
    annotated_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pills_found_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
            
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < circularity_threshold:
            continue
            
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) == 0: continue
        solidity = float(area) / cv2.contourArea(hull)
        if solidity < solidity_threshold:
            continue
        
        pills_found_contours.append(c)

    for pill_contour in pills_found_contours:
        shape, color = get_pill_properties(image, pill_contour)
        x, y, w, h = cv2.boundingRect(pill_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label_text = f"{shape}, {color}"
        cv2.putText(annotated_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image, len(pills_found_contours)


# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Intelligent Pill Detector and Identifier")

# Initialize session state variables
if 'auto_params' not in st.session_state:
    st.session_state.auto_params = {'min_area': 500, 'max_area': 40000}
if 'processing_image' not in st.session_state:
    st.session_state.processing_image = None

# --- Sidebar for Controls ---
st.sidebar.title("Detection Controls")

# --- Main Page for Upload and Display ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    original_image = np.array(pil_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Store the image in session state to persist it
    st.session_state.processing_image = cv2.resize(original_image, (800, int(800 * original_image.shape[0] / original_image.shape[1])))
    
    # Auto-tune parameters on the first upload
    st.session_state.auto_params = auto_tune_parameters(st.session_state.processing_image)
    st.info("Parameters have been auto-tuned based on the image. Adjust sliders if needed.")

# Sliders are now populated with auto-tuned values
min_area = st.sidebar.slider("1. Minimum Pill Area", 100, 5000, st.session_state.auto_params['min_area'])
max_area = st.sidebar.slider("2. Maximum Pill Area", 5000, 100000, st.session_state.auto_params['max_area'])
circularity_threshold = st.sidebar.slider("3. Circularity (Higher = More Round)", 0.1, 1.0, 0.65, 0.01)
solidity_threshold = st.sidebar.slider("4. Solidity (Higher = More Solid)", 0.1, 1.0, 0.85, 0.01)

st.sidebar.markdown("---")
# Mode selection
mode = st.sidebar.radio("Select Mode:", ("Full Image Detection", "Manual ROI Selection"))

if st.session_state.processing_image is not None:
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.processing_image, channels="BGR")

    image_to_process = st.session_state.processing_image
    
    with col2:
        st.subheader("Detection Result")
        if mode == "Full Image Detection":
            if st.button("Detect Pills in Full Image"):
                annotated_image, pill_count = detect_pills(image_to_process, min_area, max_area, circularity_threshold, solidity_threshold)
                st.image(annotated_image, channels="BGR", caption=f"Found {pill_count} pill(s)")
                st.success(f"Detection complete. Found {pill_count} pill(s).")
        
        elif mode == "Manual ROI Selection":
            st.warning("Draw a box around the pills and click 'Crop'.")
            cropped_img = st_cropper(Image.fromarray(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)), realtime_update=True, box_color='green')
            
            if st.button("Detect Pills in Selected ROI"):
                # Convert cropped PIL image back to OpenCV format
                roi_image = np.array(cropped_img)
                roi_image = cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR)
                
                # Run detection ONLY on the cropped region
                annotated_roi, pill_count = detect_pills(roi_image, min_area, max_area, circularity_threshold, solidity_threshold)
                st.image(annotated_roi, channels="BGR", caption=f"Found {pill_count} pill(s) in ROI")
                st.success(f"Detection complete. Found {pill_count} pill(s) in the selected region.")

else:
    st.info("Please upload an image to begin.")
