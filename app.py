import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- Core Image Processing & Analysis Functions ---

def get_pill_properties(image_bgr, contour):
    """
    Analyzes a single pill contour to determine its shape and color.
    This version has reverted and refined shape/color logic.
    """
    # --- Shape Analysis ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"

    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # FIX: Reverted to a more balanced and forgiving shape logic
        if circularity > 0.82 and aspect_ratio < 1.4:
            shape = "Round"
        elif aspect_ratio > 2.0: # Reverted from the too-strict 2.5
            shape = "Capsule"
        else:
            shape = "Oval"

    # --- Color Analysis ---
    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=mask)[:3]
    h, s, v = mean_hsv

    # FIX: Adjusted color ranges to prevent misclassification
    color = "Unknown"
    if s < 25 and v > 180: color = "White"
    # Adjusted hue boundaries to better separate Yellow and Brown/Orange
    elif (h > 22 and h < 35) and s > 50: color = "Yellow"
    elif (h >= 8 and h <= 22) and s > 60: color = "Brown/Orange"
    elif (h >= 100 and h <= 130) and s > 60: color = "Blue"
    elif (h >= 35 and h <= 85) and s > 50: color = "Green"

    return shape, color

def detect_pills_pipeline(image, params):
    """
    Main pill detection pipeline using adaptive thresholding AND morphological closing.
    """
    annotated_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 1. Use Adaptive Thresholding to handle different brightness levels
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)

    # 2. THE KEY FIX: Use Morphological Closing to fill holes in the pills
    # This creates solid shapes and is crucial for reliable detection.
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Find contours on the cleaned-up image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_pills = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue

        # Solidity filter is still useful to remove any remaining noise
        hull = cv2.convexHull(c)
        solidity = float(area) / cv2.contourArea(hull)
        if solidity < 0.9:
            continue

        shape, color = get_pill_properties(image, c)
        if color == "Unknown" or shape == "Unknown":
            continue

        detected_pills.append({'shape': shape, 'color': color, 'contour': c})

    for pill in detected_pills:
        x, y, w, h = cv2.boundingRect(pill['contour'])
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label_text = f"{pill['shape']}, {pill['color']}"
        cv2.putText(annotated_image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image, len(detected_pills), detected_pills

# --- Streamlit Web App Interface (No changes needed below this line) ---

st.set_page_config(layout="wide")
st.title("Intelligent Pill Detector and Identifier")
st.write("Upload an image to automatically detect pills, or use the manual ROI to find matching pills.")

if 'img' not in st.session_state:
    st.session_state.img = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert('RGB')
    original_image = np.array(pil_image)
    h, w, _ = original_image.shape
    scale = 800 / w
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(original_image, (new_w, new_h))
    st.session_state.img = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

# --- Sidebar with controls ---
with st.sidebar:
    st.title("Controls")
    mode = st.radio("Select Mode", ("Automatic Detection", "Manual ROI Matching"))

    with st.expander("Manual Tuning & Advanced Options"):
        st.write("Adjust these if the automatic detection is not perfect.")
        # Auto-parameter estimation
        if st.session_state.img is not None:
            gray_for_params = cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2GRAY)
            blurred_for_params = cv2.GaussianBlur(gray_for_params, (7, 7), 0)
            thresh_for_params = cv2.adaptiveThreshold(blurred_for_params, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            contours_for_params, _ = cv2.findContours(thresh_for_params, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours_for_params if cv2.contourArea(c) > 50]
            if areas:
                median_area = np.median(areas)
                min_area_est = int(max(50, median_area * 0.2))
                max_area_est = int(min(100000, median_area * 5.0))
            else:
                min_area_est, max_area_est = 200, 40000
        else:
            min_area_est, max_area_est = 200, 40000

        min_area = st.slider("Min Area", 50, 5000, min_area_est)
        max_area = st.slider("Max Area", 5000, 100000, max_area_est)

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
            img_for_cropper = cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)
            cropped_img_pil = st_cropper(Image.fromarray(img_for_cropper),
                                         realtime_update=True, box_color='lime', aspect_ratio=None)
            st.session_state.cropped_img = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)
        else:
            st.image(st.session_state.img, channels="BGR")
    else:
        st.info("Awaiting image upload.")

# --- Detection Logic and Results Display ---
with col2:
    st.subheader("Detection Result")
    if st.session_state.img is not None:
        if mode == "Automatic Detection":
            annotated_image, pill_count, detected_pills = detect_pills_pipeline(st.session_state.img, params)
            st.image(annotated_image, channels="BGR", caption=f"Found {pill_count} pill(s)")

            if detected_pills:
                st.write("---")
                st.subheader("Pill Summary")
                df = pd.DataFrame([p for p in detected_pills if 'contour' in p])
                if not df.empty:
                    summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                    st.table(summary_df)

        elif mode == "Manual ROI Matching":
            if st.button("Find Matching Pills"):
                cropped_img_cv = st.session_state.get('cropped_img')
                if cropped_img_cv is None or cropped_img_cv.size == 0:
                    st.error("Please crop an image first.")
                else:
                    roi_params = {'min_area': 100, 'max_area': cropped_img_cv.shape[0] * cropped_img_cv.shape[1]}
                    _, _, pills_in_roi = detect_pills_pipeline(cropped_img_cv, roi_params)

                    if not pills_in_roi:
                        st.error("Could not detect a pill in the selected ROI. Try drawing a tighter box around one pill.")
                    else:
                        target_pill = pills_in_roi[0]
                        target_shape = target_pill['shape']
                        target_color = target_pill['color']

                        _, _, all_pills = detect_pills_pipeline(st.session_state.img, params)

                        matches = [p for p in all_pills if p['shape'] == target_shape and p['color'] == target_color]

                        match_image = st.session_state.img.copy()
                        for pill in matches:
                            x, y, w, h = cv2.boundingRect(pill['contour'])
                            cv2.rectangle(match_image, (x, y), (x+w, y+h), (0, 255, 255), 4)

                        st.image(match_image, channels="BGR", caption=f"Highlighted {len(matches)} matching pill(s)")
                        st.write("---")
                        st.subheader("Matching Results")
                        match_data = {
                            'Shape': [target_shape],
                            'Color': [target_color],
                            'Quantity Found': [len(matches)]
                        }
                        st.table(pd.DataFrame(match_data))
