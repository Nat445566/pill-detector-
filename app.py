import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# --- Core Image Processing & Analysis Functions ---

def auto_estimate_parameters(image):
    """
    Analyzes the image to automatically guess the best min/max area for pills,
    making the app work automatically for the user.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = [cv2.contourArea(c) for c in contours if 100 < cv2.contourArea(c) < 100000]
    
    if not areas:
        # Provide sensible defaults if no objects are found in the pre-scan
        return {'min_area': 500, 'max_area': 40000}
        
    median_area = np.median(areas)
    min_area_est = int(max(100, median_area * 0.2))
    max_area_est = int(min(100000, median_area * 5.0))
    
    return {'min_area': min_area_est, 'max_area': max_area_est}

def get_pill_properties(image_bgr, contour):
    """
    Analyzes a single pill contour to determine its shape and color.
    This version includes robust capsule detection.
    """
    # --- Shape Analysis ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"
    
    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Get the tightest-fitting rotated rectangle
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w,h) > 0 else 0
        
        if circularity > 0.82 and aspect_ratio < 1.2:
            shape = "Round"
        elif aspect_ratio > 1.8: # Capsules are significantly longer than they are wide
            shape = "Capsule"
        else:
            shape = "Oval"

    # --- Color Analysis ---
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

def detect_pills_pipeline(image, params):
    """The main pill detection pipeline using the provided parameters."""
    annotated_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pills_found = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue
            
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < params['circularity']:
            continue
            
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) == 0: continue
        solidity = float(area) / cv2.contourArea(hull)
        if solidity < params['solidity']:
            continue
        
        pills_found.append(c)

    for pill_contour in pills_found:
        shape, color = get_pill_properties(image, pill_contour)
        x, y, w, h = cv2.boundingRect(pill_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label_text = f"{shape}, {color}"
        cv2.putText(annotated_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image, len(pills_found)

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Intelligent Pill Detector and Identifier")
st.write("Upload an image, and the system will automatically detect the pills.")

# Initialize session state
if 'img' not in st.session_state:
    st.session_state.img = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    original_image = np.array(pil_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    st.session_state.img = cv2.resize(original_image, (800, int(800 * original_image.shape[0] / original_image.shape[1])))

# --- Main display area ---
col1, col2 = st.columns(2)
with col1:
    if st.session_state.img is not None:
        st.subheader("Original Image")
        st.image(st.session_state.img, channels="BGR")
    else:
        st.info("Awaiting image upload.")

# --- Sidebar with hidden advanced controls ---
with st.sidebar:
    st.title("Controls")
    mode = st.radio("Select Mode", ("Automatic Detection", "Manual ROI"))

    with st.expander("Manual Tuning & Advanced Options"):
        st.write("Only adjust these if the automatic detection is not perfect.")
        # Estimate parameters if an image is loaded, otherwise use defaults
        if st.session_state.img is not None:
            auto_params = auto_estimate_parameters(st.session_state.img)
        else:
            auto_params = {'min_area': 500, 'max_area': 40000}
            
        min_area = st.slider("Min Area", 100, 5000, auto_params['min_area'])
        max_area = st.slider("Max Area", 5000, 100000, auto_params['max_area'])
        # A lower circularity threshold is needed to allow capsules to pass the filter
        circularity = st.slider("Circularity", 0.1, 1.0, 0.5, 0.01)
        solidity = st.slider("Solidity", 0.1, 1.0, 0.85, 0.01)

        params = {
            'min_area': min_area,
            'max_area': max_area,
            'circularity': circularity,
            'solidity': solidity
        }

# --- Detection Logic ---
with col2:
    st.subheader("Detection Result")
    if st.session_state.img is not None:
        if mode == "Automatic Detection":
            # Auto-tune and run immediately
            auto_params_live = auto_estimate_parameters(st.session_state.img)
            # Use sensible defaults for shape, which are harder to auto-tune
            auto_params_live['circularity'] = 0.50 
            auto_params_live['solidity'] = 0.85
            
            annotated_image, pill_count = detect_pills_pipeline(st.session_state.img, auto_params_live)
            st.image(annotated_image, channels="BGR", caption=f"Automatically found {pill_count} pill(s)")
            st.success(f"Automatic detection complete. Found {pill_count} pill(s).")
        
        elif mode == "Manual ROI":
            st.warning("Draw a box on the original image and click 'Detect in ROI'.")
            cropped_img_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)), realtime_update=True, box_color='lime')
            
            if st.button("Detect Pills in Selected ROI"):
                cropped_img_cv = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)
                # Use the manually tuned parameters for the ROI detection
                annotated_roi, pill_count = detect_pills_pipeline(cropped_img_cv, params)
                st.image(annotated_roi, channels="BGR", caption=f"Found {pill_count} pill(s) in ROI")
                st.success(f"Manual ROI detection complete. Found {pill_count} pill(s).")
