import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

# --- [Unchanged] Core Helper Function: Get Pill Properties ---
def get_pill_properties(image_bgr, contour):
    """A definitive, hierarchical classifier for shape and color."""
    # Shape Analysis
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

    # Color Analysis
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

# --- Detector 1: Contour-Based (Your Original Algorithm) ---
def detect_pills_adaptive_color(image, params):
    # Helper for background check
    def is_background_light(img):
        h, w, _ = img.shape
        corner_size = int(min(h, w) * 0.1)
        corners = [
            img[0:corner_size, 0:corner_size], img[0:corner_size, w-corner_size:w],
            img[h-corner_size:h, 0:corner_size], img[h-corner_size:h, w-corner_size:w]
        ]
        return np.mean([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean() for c in corners]) > 120

    # Main detection logic
    if is_background_light(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, final_mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)
        blurred = cv2.GaussianBlur(l, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=3)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --- Detector 2: Edge-Based (Canny) Detection ---
def detect_pills_canny(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, params['canny_thresh1'], params['canny_thresh2'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --- Detector 3: Watershed Segmentation ---
def detect_pills_watershed(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    markers = cv2.watershed(image, cv2.connectedComponents(np.uint8(sure_fg))[1] + 1)
    
    # Extract contours from the segmented regions
    all_contours = []
    for label in np.unique(markers):
        if label <= 1: continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
    return all_contours

# --- Detector 4: Template Matching ---
def find_template_matches(image, template, params):
    if template.size == 0: return []
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= params['match_threshold'])
    rects = [[int(pt[0]), int(pt[1]), int(w), int(h)] for pt in zip(*loc[::-1])]
    rects, _ = cv2.groupRectangles(rects * 2, 1, 0.2)
    return rects

# --- Detector 5: Feature-Based Matching ---
def find_feature_match(image, template):
    if template.size == 0: return None
    main_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(main_gray, None)
    if des1 is None or des2 is None: return None
    index_params= dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        return np.int32(cv2.perspectiveTransform(pts, M))
    return None

# --- UI Helper ---
def resize_for_display(image, max_height=400):
    h, w = image.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Pharmaceutical Tablet Analysis System")

# Initialize session state
if 'img' not in st.session_state:
    st.session_state.img = None

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    detector_options = {
        "Contour-Based (Adaptive Color)": detect_pills_adaptive_color,
        "Edge-Based (Canny)": detect_pills_canny,
        "Watershed Segmentation": detect_pills_watershed,
        "Template Matching": find_template_matches,
        "Feature-Based Matching": find_feature_match
    }
    detector_name = st.selectbox("1. Select Detector Algorithm", detector_options.keys())
    
    analysis_mode = st.radio("2. Select Analysis Mode", ("Full Image Detection", "Manual ROI Matching"))

    with st.expander("🔬 Tuning & Advanced Options"):
        min_area = st.slider("Min Pill Area", 50, 5000, 500)
        max_area = st.slider("Max Pill Area", 5000, 100000, 50000)
        params = {'min_area': min_area, 'max_area': max_area}
        if detector_name == "Edge-Based (Canny)":
            params['canny_thresh1'] = st.slider("Canny Threshold 1", 0, 255, 30)
            params['canny_thresh2'] = st.slider("Canny Threshold 2", 0, 255, 150)
        if detector_name == "Template Matching":
            params['match_threshold'] = st.slider("Match Confidence", 0.5, 1.0, 0.8)

# --- Main Page Layout ---
_, main_col, _ = st.columns([1, 2, 1])
with main_col:
    st.write("Upload an image, then select your desired algorithm and analysis mode from the sidebar.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert('RGB')
        # Standard preprocessing: resize to a consistent width
        orig_img = np.array(pil_img)
        scale = 800 / orig_img.shape[1]
        new_size = (int(orig_img.shape[1] * scale), int(orig_img.shape[0] * scale))
        st.session_state.img = cv2.cvtColor(cv2.resize(orig_img, new_size), cv2.COLOR_RGB2BGR)

    if st.session_state.img is not None:
        # --- Display Area ---
        st.subheader("Image Analysis")
        image_to_process = st.session_state.img
        
        # In ROI mode, activate the cropper
        if analysis_mode == "Manual ROI Matching":
            st.info("Draw a box on the image below to define your Region of Interest (ROI).")
            display_img_resized = resize_for_display(st.session_state.img)
            img_for_cropper = cv2.cvtColor(display_img_resized, cv2.COLOR_BGR2RGB)
            cropped_pil = st_cropper(Image.fromarray(img_for_cropper), realtime_update=True, box_color='lime')
            cropped_cv = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
            # Rescale cropped coordinates back to original image size
            h_disp, w_disp = display_img_resized.shape[:2]
            h_orig, w_orig = st.session_state.img.shape[:2]
            scale_h, scale_w = h_orig / h_disp, w_orig / w_disp
            (x,y,w,h) = (cropped_pil.box[0]*scale_w, cropped_pil.box[1]*scale_h, cropped_pil.box[2]*scale_w, cropped_pil.box[3]*scale_h)
            image_to_process = st.session_state.img[int(y):int(y+h), int(x):int(x+w)]
        else:
            st.image(resize_for_display(st.session_state.img), channels="BGR", caption="Full image ready for analysis.")

        st.divider()

        # --- Execution and Results ---
        button_label = "Run Analysis"
        if analysis_mode == "Manual ROI Matching":
             button_label = "Analyze Selected ROI"
        
        is_template_detector = detector_name in ["Template Matching", "Feature-Based Matching"]
        if is_template_detector:
             button_label = f"Find Matches Using ROI as Template"

        if st.button(button_label, use_container_width=True):
            with st.spinner("Processing..."):
                annotated_image = image_to_process.copy()
                pill_count = 0
                
                # Logic for contour-based detectors
                if detector_name in ["Contour-Based (Adaptive Color)", "Edge-Based (Canny)", "Watershed Segmentation"]:
                    contours = detector_options[detector_name](image_to_process, params)
                    detected_pills = []
                    for c in contours:
                        area = cv2.contourArea(c)
                        if params['min_area'] < area < params['max_area']:
                             shape, color = get_pill_properties(image_to_process, c)
                             detected_pills.append({'shape': shape, 'color': color, 'contour': c})
                             x, y, w, h = cv2.boundingRect(c)
                             cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    pill_count = len(detected_pills)
                    st.metric(f"Pills Found in {'ROI' if analysis_mode == 'Manual ROI Matching' else 'Full Image'}", pill_count)
                    st.image(resize_for_display(annotated_image), channels="BGR", caption=f"Result from {detector_name}")
                    if detected_pills:
                        df = pd.DataFrame(detected_pills).drop(columns='contour')
                        summary_df = df.groupby(['shape', 'color']).size().reset_index(name='quantity')
                        st.dataframe(summary_df, use_container_width=True)

                # Logic for template-based detectors (always use ROI as template, search full image)
                elif is_template_detector:
                    if analysis_mode != "Manual ROI Matching" or image_to_process.size == 0:
                        st.error("Please select 'Manual ROI Matching' mode and draw a box to define a template.")
                    else:
                        template = image_to_process
                        annotated_image = st.session_state.img.copy() # Search on the full image
                        if detector_name == "Template Matching":
                            matches = find_template_matches(st.session_state.img, template, params)
                            pill_count = len(matches)
                            for (x, y, w, h) in matches:
                                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        elif detector_name == "Feature-Based Matching":
                            match_poly = find_feature_match(st.session_state.img, template)
                            if match_poly is not None:
                                cv2.polylines(annotated_image, [match_poly], True, (255, 0, 255), 3)
                                pill_count = 1
                        
                        st.metric(f"Matches Found in Full Image", pill_count)
                        st.image(resize_for_display(annotated_image), channels="BGR", caption=f"Result from {detector_name}")

    elif not uploaded_file:
         st.info("Awaiting image upload to begin.")
