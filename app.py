import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- Core Image Processing & Analysis Functions ---

def get_pill_properties(image_bgr, contour):
    """
    A definitive, hierarchical classifier for shape and color, robust to lighting changes.
    """
    # --- Shape Analysis ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"
    if perimeter > 0:
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if circularity > 0.82 and aspect_ratio < 1.4: shape = "Round"
        elif aspect_ratio > 2.0: shape = "Capsule"
        elif num_vertices == 4: shape = "Rectangular"
        else: shape = "Oval"

    # --- HIERARCHICAL COLOR ANALYSIS ---
    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=eroded_mask)[:3]
    h, s, v = mean_hsv

    # 1. First, classify achromatic colors (White, Gray, Black).
    if s < 45:
        if v > 150: color = "White"
        elif v < 70: color = "Black"
        else: color = "Gray"
    # 2. If it's a chromatic color, then check the Hue.
    else:
        if (h <= 10 or h >= 165):
            if s > 120: color = "Red"
            else: color = "Pink"
        elif h > 10 and h <= 25:
            if v < 180: color = "Brown"
            else: color = "Orange"
        elif h > 25 and h <= 40: color = "Yellow"
        elif h > 40 and h <= 85: color = "Green"
        elif h > 85 and h <= 130: color = "Blue"
        else: color = "Unknown"

    return shape, color

def is_background_light(image):
    """Analyzes image corners to determine if the background is light or dark."""
    h, w, _ = image.shape
    corner_size = int(min(h, w) * 0.1)
    corners = [
        image[0:corner_size, 0:corner_size],
        image[0:corner_size, w-corner_size:w],
        image[h-corner_size:h, 0:corner_size],
        image[h-corner_size:h, w-corner_size:w]
    ]
    avg_brightness = np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() for c in corners])
    return avg_brightness > 120

def detect_on_dark_bg(image, params):
    """Pipeline optimized for dark backgrounds using LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    blurred_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    return opened_mask

def detect_on_light_bg(image, params):
    """Pipeline for light/complex backgrounds using a full color palette."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'VibrantRed1': [np.array([0, 120, 70]), np.array([10, 255, 255])],
        'VibrantRed2': [np.array([165, 120, 70]), np.array([180, 255, 255])],
        'PalePink1': [np.array([0, 40, 100]), np.array([10, 119, 255])],
        'PalePink2': [np.array([165, 40, 100]), np.array([180, 119, 255])],
        'BrownOrange': [np.array([11, 50, 50]), np.array([25, 255, 255])],
        'Yellow': [np.array([26, 40, 50]), np.array([40, 255, 255])],
        'Green': [np.array([41, 40, 50]), np.array([85, 255, 255])],
        'Blue': [np.array([86, 50, 50]), np.array([130, 255, 255])],
    }

    combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")
    for lower, upper in color_ranges.values():
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    dilated_white = cv2.dilate(white_thresh, np.ones((3,3), np.uint8), iterations=2)
    combined_mask = cv2.bitwise_or(combined_mask, dilated_white)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    return opened_mask

def detect_pills_pipeline(image, params):
    """Master adaptive pipeline."""
    annotated_image = image.copy()
    
    if is_background_light(image):
        final_mask = detect_on_light_bg(image, params)
    else:
        final_mask = detect_on_dark_bg(image, params)
        
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_pills = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (params['min_area'] < area < params['max_area']):
            continue

        hull = cv2.convexHull(c)
        if hull.shape[0] < 3: continue
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

# --- UI Helper Function ---
def resize_for_display(image, max_height=400): # <<< KEY CHANGE: Lowered from 500 to 400
    """
    Resizes an image to a maximum display height while maintaining aspect ratio.
    """
    h, w, _ = image.shape
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Automated Pharmaceutical Tablet Counting System")

if 'img' not in st.session_state:
    st.session_state.img = None

_, main_col, _ = st.columns([1, 2, 1])

with main_col:
    st.write("Upload an image to automatically detect pills.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
        st.write("Adjust these sliders if detection is not perfect.")
        min_area = st.slider("Min Area", 50, 5000, 100)
        max_area = st.slider("Max Area", 5000, 100000, 50000)

        params = {
            'min_area': min_area,
            'max_area': max_area
        }

# --- Centered, "Report Style" Layout ---

with main_col:
    if st.session_state.img is not None:
        st.subheader("Original Image")
        if mode == "Manual ROI Matching":
            st.warning("Draw a box around a single pill to find its matches.")
            display_img_resized = resize_for_display(st.session_state.img)
            img_for_cropper = cv2.cvtColor(display_img_resized, cv2.COLOR_BGR2RGB)
            cropped_img_pil = st_cropper(Image.fromarray(img_for_cropper),
                                         realtime_update=True, box_color='lime', aspect_ratio=None)
            st.session_state.cropped_img = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)
        else:
            display_img_resized = resize_for_display(st.session_state.img)
            st.image(display_img_resized, channels="BGR", use_container_width=True)

        st.divider()

        st.subheader("Detection Result")
        if mode == "Automatic Detection":
            annotated_image, pill_count, detected_pills = detect_pills_pipeline(st.session_state.img, params)
            display_annotated_resized = resize_for_display(annotated_image)

            st.metric(label="Total Pills Found", value=pill_count)
            st.image(display_annotated_resized, channels="BGR", use_container_width=True)

            if detected_pills:
                st.markdown("##### Pill Summary")
                df = pd.DataFrame([p for p in detected_pills if 'contour' in p])
                summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                
                total_quantity = summary_df['quantity'].sum()
                total_row = pd.DataFrame([{'shape': '**Total**', 'color': '', 'quantity': total_quantity}])
                summary_df = pd.concat([summary_df, total_row], ignore_index=True)
                
                st.dataframe(summary_df, use_container_width=True)

        elif mode == "Manual ROI Matching":
            if st.button("Find Matching Pills"):
                cropped_img_cv = st.session_state.get('cropped_img')
                if cropped_img_cv is None or cropped_img_cv.size == 0:
                    st.error("Please draw a box on the image above first.")
                else:
                    roi_params = {'min_area': 100, 'max_area': cropped_img_cv.shape[0] * cropped_img_cv.shape[1]}
                    _, _, pills_in_roi = detect_pills_pipeline(cropped_img_cv, roi_params)

                    if not pills_in_roi:
                        st.error("Could not detect a pill in the selected ROI.")
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
                        
                        display_match_resized = resize_for_display(match_image)

                        st.metric(label="Total Matches Found", value=len(matches))
                        st.image(display_match_resized, channels="BGR", use_container_width=True)
                        
                        st.markdown("##### Matching Results")
                        match_data = {
                            'Target Shape': [target_shape],
                            'Target Color': [target_color],
                            'Quantity Found': [len(matches)]
                        }
                        st.dataframe(pd.DataFrame(match_data), use_container_width=True)
    
    elif not uploaded_file:
         st.info("Awaiting image upload to display results.")
