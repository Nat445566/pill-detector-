import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# --- Core Feature Extraction and Analysis Functions ---

def get_template_features(roi_image):
    """
    Analyzes the user-selected ROI to extract the key features of the template pill.
    Returns a dictionary of features or None if no clear object is found.
    """
    # Pre-process the small ROI image
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find the largest contour in the ROI, assuming it's the pill
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    template_contour = max(contours, key=cv2.contourArea)
    
    # --- Feature Extraction ---
    # 1. Color Feature (in robust HSV space)
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_roi.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [template_contour], -1, 255, -1)
    mean_hsv = cv2.mean(hsv_roi, mask=mask)[:3]

    # 2. Shape Feature (Circularity)
    area = cv2.contourArea(template_contour)
    perimeter = cv2.arcLength(template_contour, True)
    circularity = 4 * np.pi * (area / (perimeter**2)) if perimeter > 0 else 0

    # 3. Size Feature (Area)
    return {
        "hsv_color": np.array(mean_hsv),
        "circularity": circularity,
        "area": area
    }

def find_candidate_pills(full_image):
    """
    Scans the entire image to find all potential objects (candidates) that could be pills.
    """
    gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Use Canny edge detection followed by morphology to find distinct objects
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Pill Detection by Example")
st.write("Instructions: **1.** Upload an image. **2.** Draw a box around **one** example pill. **3.** Click the button to find and count all similar pills.")

# --- Sidebar for Tolerance Controls ---
st.sidebar.title("Matching Sensitivity")
st.sidebar.write("Control how strict the matching is. Lower values are more strict.")
color_tolerance = st.sidebar.slider("Color Similarity Tolerance", 1, 100, 35)
shape_tolerance = st.sidebar.slider("Shape Similarity Tolerance", 0.01, 1.0, 0.25, 0.01)
size_tolerance = st.sidebar.slider("Size Similarity Tolerance (%)", 1, 100, 40)

# --- Main App Logic ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Select a Template Pill")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        # Resize for consistent display and processing
        pil_image.thumbnail((800, 800)) 
        
        # The cropper tool for user to select the ROI
        cropped_img = st_cropper(pil_image, realtime_update=True, box_color='lime', aspect_ratio=None, key='cropper')

        if st.button("Find and Count Similar Pills", type="primary"):
            # Convert the full image to OpenCV format
            full_image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert the user's crop to OpenCV format
            roi_image_cv = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing template pill..."):
                template_features = get_template_features(roi_image_cv)

            if template_features is None:
                st.error("Could not identify a clear pill in the selected region. Please draw a tighter box.")
            else:
                with st.spinner("Scanning image and matching pills..."):
                    candidate_contours = find_candidate_pills(full_image_cv)
                    
                    matching_pills = []
                    annotated_image = full_image_cv.copy()

                    for candidate in candidate_contours:
                        # Get features of the candidate pill
                        candidate_features = get_template_features(full_image_cv.copy())
                        if candidate_features is not None:
                             # Extract features for the current candidate contour
                            candidate_hsv_roi = cv2.cvtColor(full_image_cv, cv2.COLOR_BGR2HSV)
                            candidate_mask = np.zeros(candidate_hsv_roi.shape[:2], dtype="uint8")
                            cv2.drawContours(candidate_mask, [candidate], -1, 255, -1)
                            candidate_mean_hsv = cv2.mean(candidate_hsv_roi, mask=candidate_mask)[:3]
                            candidate_hsv_color = np.array(candidate_mean_hsv)
                            candidate_area = cv2.contourArea(candidate)
                            candidate_perimeter = cv2.arcLength(candidate, True)
                            candidate_circularity = 4 * np.pi * (candidate_area / (candidate_perimeter**2)) if candidate_perimeter > 0 else 0

                            # Compare features with tolerances
                            color_diff = cv2.norm(template_features["hsv_color"], candidate_hsv_color, cv2.NORM_L2)
                            shape_diff = abs(template_features["circularity"] - candidate_circularity)
                            size_diff = abs(1 - candidate_area / template_features["area"]) if template_features["area"] > 0 else 1

                            # Check if all features are within tolerance
                            if (color_diff < color_tolerance and 
                                shape_diff < shape_tolerance and 
                                size_diff < (size_tolerance / 100.0)):
                                matching_pills.append(candidate)
                    
                    # Draw results on the image
                    for pill_contour in matching_pills:
                        x, y, w, h = cv2.boundingRect(pill_contour)
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    with col2:
                        st.subheader("2. Detection Result")
                        st.image(annotated_image, channels="BGR", caption=f"Found {len(matching_pills)} matching pill(s).")
                        st.success(f"Found {len(matching_pills)} pills similar to your selection.")
