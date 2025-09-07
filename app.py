import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- [UNCHANGED] Core Helper Function: Get Pill Properties ---
def get_pill_properties(image_bgr, contour):
    """A definitive, hierarchical classifier for shape and color."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"
    if perimeter > 0:
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if circularity > 0.82 and aspect_ratio < 1.4: shape = "Round"
        elif aspect_ratio > 2.0: shape = "Capsule"
        elif len(approx) == 4: shape = "Rectangular"
        else: shape = "Oval"

    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    eroded_mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=2)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=eroded_mask)[:3]
    h, s, v = mean_hsv
    if s < 45:
        if v > 150: color = "White"
        elif v < 70: color = "Black"
        else: color = "Gray"
    else:
        if (h <= 10 or h >= 165): color = "Red" if s > 120 else "Pink"
        elif h <= 25: color = "Brown" if v < 180 else "Orange"
        elif h <= 40: color = "Yellow"
        elif h <= 85: color = "Green"
        elif h <= 130: color = "Blue"
        else: color = "Unknown"
    return shape, color

# --- [NEW] Central Pill Filtering and Classification Function ---
def filter_and_classify_pills(image, contours, params):
    """
    Applies the original, robust filtering logic to any list of contours.
    This is the key to ensuring all detectors find ONLY pills.
    """
    detected_pills = []
    for c in contours:
        # Filter by area first
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue

        # CRITICAL: Filter by solidity to remove irregular shapes
        hull = cv2.convexHull(c)
        if hull.shape[0] < 3: continue
        solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        if solidity < 0.9:
            continue

        # Classify the remaining valid contours
        shape, color = get_pill_properties(image, c)
        if color == "Unknown" or shape == "Unknown":
            continue

        detected_pills.append({'shape': shape, 'color': color, 'contour': c})
    return detected_pills

# --- Detector Functions (Now only generate candidate contours) ---

# Detector 1: Your original adaptive color algorithm
def get_contours_adaptive_color(image, params):
    # Helper to check if background is light or dark
    def is_background_light(img):
        h, w, _ = img.shape
        corner_size = int(min(h, w) * 0.1)
        corners = [
            img[0:corner_size, 0:corner_size], img[0:corner_size, w-corner_size:w],
            img[h-corner_size:h, 0:corner_size], img[h-corner_size:h, w-corner_size:w]
        ]
        return np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() for c in corners]) > 120

    # Create mask based on background
    if is_background_light(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Simplified and more robust light background detection
        lower_bound = np.array([0, 40, 50])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        final_mask = cv2.bitwise_or(color_mask, white_mask)
    else: # Dark background detection
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        blurred = cv2.GaussianBlur(l, (5, 5), 0)
        _, final_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up the mask and find contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detector 2: Edge-Based (Canny)
def get_contours_canny(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, params['canny_thresh1'], params['canny_thresh2'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detector 3: Watershed Segmentation
def get_contours_watershed(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    markers = cv2.watershed(image, cv2.connectedComponents(np.uint8(sure_fg))[1] + 1)
    
    all_contours = []
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
    return all_contours

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("Pharmaceutical Tablet Analysis System")

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    detector_options = {
        "Contour-Based (Adaptive Color)": get_contours_adaptive_color,
        "Edge-Based (Canny)": get_contours_canny,
        "Watershed Segmentation": get_contours_watershed
    }
    detector_name = st.selectbox("1. Select Detector Algorithm", detector_options.keys())
    
    analysis_mode = st.radio("2. Select Analysis Mode", ("Full Image Detection", "Manual ROI (Matching Pills)"))

    with st.expander("🔬 Tuning & Advanced Options"):
        min_area = st.slider("Min Pill Area", 50, 5000, 500)
        max_area = st.slider("Max Pill Area", 5000, 100000, 50000)
        params = {'min_area': min_area, 'max_area': max_area}
        if detector_name == "Edge-Based (Canny)":
            params['canny_thresh1'] = st.slider("Canny Threshold 1", 0, 255, 30)
            params['canny_thresh2'] = st.slider("Canny Threshold 2", 0, 255, 150)

# --- Main Page Layout ---
_, main_col, _ = st.columns([1, 2, 1])
with main_col:
    st.write("Upload an image, then select your desired algorithm and analysis mode from the sidebar.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert('RGB')
        orig_img = np.array(pil_img)
        scale = 800 / orig_img.shape[1]
        new_size = (int(orig_img.shape[1] * scale), int(orig_img.shape[0] * scale))
        st.session_state.img = cv2.cvtColor(cv2.resize(orig_img, new_size), cv2.COLOR_RGB_BGR)

    if 'img' in st.session_state and st.session_state.img is not None:
        st.subheader("Image Analysis")
        
        # --- Full Image Detection Mode ---
        if analysis_mode == "Full Image Detection":
            st.image(st.session_state.img, channels="BGR", caption="Full image ready for analysis.")
            if st.button("Run Full Image Detection", use_container_width=True):
                with st.spinner("Analyzing..."):
                    contours = detector_options[detector_name](st.session_state.img, params)
                    detected_pills = filter_and_classify_pills(st.session_state.img, contours, params)
                    
                    annotated_image = st.session_state.img.copy()
                    for pill in detected_pills:
                        x, y, w, h = cv2.boundingRect(pill['contour'])
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{pill['shape']}, {pill['color']}"
                        cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    st.subheader("Detection Results")
                    st.metric("Total Pills Found", len(detected_pills))
                    st.image(annotated_image, channels="BGR", caption=f"Result from {detector_name}")
                    if detected_pills:
                        df = pd.DataFrame(detected_pills).drop(columns='contour')
                        summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                        st.dataframe(summary_df, use_container_width=True)

        # --- Manual ROI (Matching Pills) Mode ---
        elif analysis_mode == "Manual ROI (Matching Pills)":
            st.info("Draw a box around a single pill to define the target for matching.")
            cropped_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)), realtime_update=True, box_color='lime')
            
            if st.button("Find All Matching Pills", use_container_width=True):
                with st.spinner("Finding matches..."):
                    # Get the ROI from the cropper
                    box = cropped_pil.box
                    roi = st.session_state.img[box[1]:box[3], box[0]:box[2]]

                    if roi.size == 0:
                        st.error("Please draw a valid box on the image.")
                    else:
                        # 1. Find the target pill within the ROI
                        roi_contours = detector_options[detector_name](roi, params)
                        target_pills = filter_and_classify_pills(roi, roi_contours, params)

                        if not target_pills:
                            st.error("Could not identify a valid pill in the selected ROI. Please try drawing a tighter box.")
                        else:
                            target_pill = target_pills[0] # Use the first pill found in ROI
                            target_shape = target_pill['shape']
                            target_color = target_pill['color']

                            # 2. Find all pills in the full image
                            all_contours = detector_options[detector_name](st.session_state.img, params)
                            all_pills = filter_and_classify_pills(st.session_state.img, all_contours, params)
                            
                            # 3. Filter for matches
                            matches = [p for p in all_pills if p['shape'] == target_shape and p['color'] == target_color]

                            # 4. Display results
                            match_image = st.session_state.img.copy()
                            for pill in matches:
                                x, y, w, h = cv2.boundingRect(pill['contour'])
                                cv2.rectangle(match_image, (x, y), (x+w, y+h), (0, 255, 255), 3)

                            st.subheader("Matching Results")
                            st.metric(f"Found {len(matches)} pills matching the target", f"{target_shape}, {target_color}")
                            st.image(match_image, channels="BGR", caption=f"Found {len(matches)} matches for the selected pill.")
                            
                            match_data = {
                                'Target Shape': [target_shape],
                                'Target Color': [target_color],
                                'Quantity Found': [len(matches)]
                            }
                            st.dataframe(pd.DataFrame(match_data), use_container_width=True)

    elif not uploaded_file:
         st.info("Awaiting image upload to begin.")
