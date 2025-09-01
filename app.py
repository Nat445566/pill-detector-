import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- Core Image Processing & Analysis Functions ---

def auto_estimate_parameters(image):
    """
    Analyzes the image to automatically guess the best min/max area for pills,
    making the app work automatically for the user.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Using adaptive thresholding can be more robust to lighting changes
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    This version includes more robust shape and color detection.
    """
    # --- Shape Analysis ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"

    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # Refined shape logic
        if circularity > 0.85 and aspect_ratio < 1.2:
            shape = "Round"
        elif aspect_ratio > 1.8:
            shape = "Capsule"
        else:
            shape = "Oval"

    # --- Color Analysis ---
    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=mask)[:3]
    h, s, v = mean_hsv

    # Refined HSV color ranges to fix misclassifications
    color = "Unknown"
    if s < 35 and v > 170: color = "White"
    elif s < 65 and v < 60: color = "Black" # Added for completeness
    elif (h >= 100 and h <= 130) and s > 70: color = "Blue"
    elif (h >= 35 and h <= 85) and s > 60: color = "Green"
    elif (h >= 8 and h <= 30) and s > 80: color = "Brown/Orange"
    elif (h < 10 or h > 170) and s > 80: color = "Red"
    elif (h > 20 and h < 35) and s > 60 : color = "Yellow" # Added for completeness

    return shape, color

def detect_pills_pipeline(image, params):
    """
    The main pill detection pipeline. Now returns the annotated image,
    the total count, and a list of detected pill properties.
    """
    annotated_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7) # Increased blur slightly for better noise reduction

    # Use adaptive thresholding for better contour detection in varied lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_pills = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue

        # Looser circularity to catch imperfect shapes, main logic is in get_pill_properties
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.4:
            continue

        shape, color = get_pill_properties(image, c)

        # Skip if color or shape is not confidently determined
        if color == "Unknown" or shape == "Unknown":
            continue

        detected_pills.append({'shape': shape, 'color': color, 'contour': c})

    # Draw results on the image
    for pill in detected_pills:
        x, y, w, h = cv2.boundingRect(pill['contour'])
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label_text = f"{pill['shape']}, {pill['color']}"
        cv2.putText(annotated_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image, len(detected_pills), detected_pills

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Intelligent Pill Detector and Identifier")
st.write("Upload an image to automatically detect pills, or use the manual ROI to find matching pills.")

# Initialize session state
if 'img' not in st.session_state:
    st.session_state.img = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    original_image = np.array(pil_image)
    # Ensure resizing maintains aspect ratio
    h, w, _ = original_image.shape
    scale = 800 / w
    new_h, new_w = int(h * scale), int(w * scale)
    st.session_state.img = cv2.resize(original_image, (new_w, new_h))
    st.session_state.img = cv2.cvtColor(st.session_state.img, cv2.COLOR_RGB2BGR)


# --- Sidebar with controls ---
with st.sidebar:
    st.title("Controls")
    mode = st.radio("Select Mode", ("Automatic Detection", "Manual ROI Matching"))

    with st.expander("Manual Tuning & Advanced Options"):
        st.write("Adjust these if the automatic detection is not perfect.")
        if st.session_state.img is not None:
            auto_params = auto_estimate_parameters(st.session_state.img)
        else:
            auto_params = {'min_area': 500, 'max_area': 40000}

        min_area = st.slider("Min Area", 100, 5000, auto_params['min_area'])
        max_area = st.slider("Max Area", 5000, 100000, auto_params['max_area'])

        params = {
            'min_area': min_area,
            'max_area': max_area
        }

# --- Main display area ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    if st.session_state.img is not None:
        if mode == "Manual ROI Matching":
            st.warning("Draw a box around a single pill to find its matches.")
            # Use st_cropper in the main column
            cropped_img_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)),
                                         realtime_update=True, box_color='lime', aspect_ratio=None)
            st.session_state.cropped_img = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB_BGR)
        else:
            st.image(st.session_state.img, channels="BGR")
    else:
        st.info("Awaiting image upload.")

# --- Detection Logic and Results Display ---
with col2:
    st.subheader("Detection Result")
    if st.session_state.img is not None:
        # Use the manually tuned parameters for both modes for consistency
        # Or you could use auto_params_live for automatic mode if you prefer
        
        if mode == "Automatic Detection":
            annotated_image, pill_count, detected_pills = detect_pills_pipeline(st.session_state.img, params)
            st.image(annotated_image, channels="BGR", caption=f"Found {pill_count} pill(s)")

            if detected_pills:
                st.write("---")
                st.subheader("Pill Summary")
                # Create a DataFrame for easy counting
                df = pd.DataFrame(detected_pills)
                summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                st.table(summary_df)

        elif mode == "Manual ROI Matching":
            if st.button("Find Matching Pills"):
                # 1. Analyze the cropped ROI to identify the target pill
                cropped_img_cv = st.session_state.get('cropped_img')
                if cropped_img_cv is None:
                    st.error("Please crop an image first.")
                else:
                    # Use tight area parameters for the single pill in the ROI
                    roi_params = {'min_area': 100, 'max_area': cropped_img_cv.shape[0] * cropped_img_cv.shape[1]}
                    _, _, pills_in_roi = detect_pills_pipeline(cropped_img_cv, roi_params)

                    if not pills_in_roi:
                        st.error("Could not detect a pill in the selected ROI. Try drawing a tighter box.")
                    else:
                        target_pill = pills_in_roi[0] # Assume first pill is the target
                        target_shape = target_pill['shape']
                        target_color = target_pill['color']

                        # 2. Analyze the full image to find all pills
                        _, _, all_pills = detect_pills_pipeline(st.session_state.img, params)

                        # 3. Find matches
                        matches = []
                        for pill in all_pills:
                            if pill['shape'] == target_shape and pill['color'] == target_color:
                                matches.append(pill)

                        # 4. Draw rectangles on the original image for visualization
                        match_image = st.session_state.img.copy()
                        for pill in matches:
                            x, y, w, h = cv2.boundingRect(pill['contour'])
                            cv2.rectangle(match_image, (x, y), (x+w, y+h), (0, 255, 255), 4) # Yellow highlight

                        st.image(match_image, channels="BGR", caption=f"Highlighted {len(matches)} matching pill(s)")
                        st.write("---")
                        st.subheader("Matching Results")
                        match_data = {
                            'Shape': [target_shape],
                            'Color': [target_color],
                            'Quantity Found': [len(matches)]
                        }
                        st.table(pd.DataFrame(match_data))
