import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ====================================================================
# 1. Helper Functions for Color and Shape Detection
# ====================================================================

def get_color_name(hsv_color):
    """Takes an HSV color value and returns a descriptive color name."""
    h, s, v = hsv_color
    if s < 35 and v > 190: return "White"
    if s < 35 and v < 80: return "Black"
    if s < 45 and 80 <= v <= 190: return "Gray"
    if (0 <= h <= 10) or (170 <= h <= 180): return "Red"
    elif 11 <= h <= 25: return "Orange"
    elif 26 <= h <= 35: return "Yellow"
    elif 36 <= h <= 85: return "Green"
    elif 86 <= h <= 125: return "Blue"
    elif 126 <= h <= 145: return "Purple"
    elif 146 <= h <= 169: return "Pink"
    else: return "Brown/Other"

def get_shape_name(contour):
    """Takes a contour and returns a shape name based on its geometry."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    if len(approx) > 6:
        return "Circle" if 0.9 <= aspect_ratio <= 1.1 else "Oval/Capsule"
    elif len(approx) == 4:
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    else:
        return "Irregular"

# ====================================================================
# 2. Main Image Processing Function
# ====================================================================

def analyze_pills(image, roi, bg_threshold, min_area):
    """The main pipeline to detect, count, and analyze pills."""
    x, y, w, h = roi
    cropped_image = image[y:y+h, x:x+w]
    
    corner_h, corner_w, _ = cropped_image.shape
    if corner_h < 20 or corner_w < 20:
        return cropped_image, [] # ROI is too small, return empty
        
    corners = np.array([
        cropped_image[5:15, 5:15].mean(axis=(0,1)),
        cropped_image[5:15, corner_w-15:corner_w-5].mean(axis=(0,1)),
        cropped_image[corner_h-15:corner_h-5, 5:15].mean(axis=(0,1)),
        cropped_image[corner_h-15:corner_h-5, corner_w-15:corner_w-5].mean(axis=(0,1))
    ])
    std_dev = np.std(corners)

    if std_dev < bg_threshold: # Uniform Background
        blurred = cv2.GaussianBlur(cropped_image, (7, 7), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        avg_hsv_bg = np.mean(corners, axis=0)
        h_range, s_range, v_range = 15, 60, 60
        lower_bg = np.array([max(0, avg_hsv_bg[0]-h_range), max(0, avg_hsv_bg[1]-s_range), max(0, avg_hsv_bg[2]-v_range)])
        upper_bg = np.array([min(179, avg_hsv_bg[0]+h_range), min(255, avg_hsv_bg[1]+s_range), min(255, avg_hsv_bg[2]+v_range)])
        background_mask = cv2.inRange(hsv_image, lower_bg, upper_bg)
        mask = cv2.bitwise_not(background_mask)
    else: # Complex Background
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8); cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((5, 5), np.uint8); cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pill_data = []
    output_image = cropped_image.copy()
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > min_area:
            single_pill_mask = np.zeros(cleaned_mask.shape, dtype="uint8"); cv2.drawContours(single_pill_mask, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(cropped_image, mask=single_pill_mask)[:3]; mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            pill_data.append({ "Pill ID": i + 1, "Color": get_color_name(mean_hsv), "Shape": get_shape_name(cnt), "Area (px)": int(cv2.contourArea(cnt)) })
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, str(i + 1), (cX - 15, cY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # --- THIS IS THE CORRECTED LINE ---
    return output_image, pill_data

# ====================================================================
# 3. Streamlit User Interface
# ====================================================================

st.set_page_config(layout="wide")
st.title("💊 Pill Detector Pro")
st.write("Upload an image to count pills and analyze their color and shape.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.subheader("Detection Parameters")
    min_pill_area = st.slider("1. Minimum Pill Area (pixels)", 50, 2000, 200, help="Filters out small noise. Increase if pills are large.")
    bg_std_threshold = st.slider("2. Background Uniformity", 5.0, 50.0, 15.0, help="Lower if the background is very uniform (like a solid color), higher if it has texture or shadows.")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB_BGR)
    img_h, img_w, _ = image_bgr.shape

    with st.sidebar:
        st.subheader("Region of Interest (ROI)")
        roi_x = st.slider("ROI X start", 0, img_w, 0)
        roi_y = st.slider("ROI Y start", 0, img_h, 0)
        roi_w = st.slider("ROI Width", 1, img_w, img_w)
        roi_h = st.slider("ROI Height", 1, img_h, img_h)
        
    roi = (roi_x, roi_y, roi_w, roi_h)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image with ROI")
        display_img_with_roi = image_bgr.copy()
        cv2.rectangle(display_img_with_roi, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 5)
        st.image(display_img_with_roi, channels="BGR", caption="Adjust ROI and parameters in the sidebar.")

    if st.sidebar.button("Analyze Pills"):
        processed_image, pill_results = analyze_pills(image_bgr, roi, bg_std_threshold, min_pill_area)
        
        with col2:
            st.subheader("Processed Image")
            if pill_results:
                st.image(processed_image, channels="BGR", caption=f"Detected {len(pill_results)} pills.")
            else:
                st.warning("No pills detected. Try adjusting the parameters or ROI.")

        st.subheader("📊 Detection Results")
        if pill_results:
            st.dataframe(pd.DataFrame(pill_results))
        else:
            st.info("No data to display.")
else:
    st.info("Awaiting image upload...")
