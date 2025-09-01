import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# --- Core Image Processing & Analysis Functions ---

def get_pill_properties(image_bgr, contour):
    """
    Analyzes a single, confirmed pill contour to determine its shape and color.
    This version includes robust capsule detection.
    """
    # --- Shape Analysis ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    shape = "Unknown"
    
    if perimeter > 0:
        # Use a rotated rectangle for accurate aspect ratio
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Use circularity for round/oval distinction
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if aspect_ratio > 1.8: # Capsules are significantly longer than they are wide
            shape = "Capsule"
        elif circularity > 0.82:
            shape = "Round"
        else:
            shape = "Oval"

    # --- Color Analysis ---
    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=mask)[:3]
    h, s, v = mean_hsv
    
    color = "Unknown"
    # Tuned HSV ranges for better accuracy
    if s < 40 and v > 160: color = "White"
    elif 35 < h < 85 and s > 60: color = "Green"
    elif (h < 12 or h > 168) and s > 80: color = "Red/Brown"
    elif 15 < h < 35 and s > 80: color = "Yellow"
    elif 85 < h < 130 and s > 70: color = "Blue"
    
    return shape, color

def detect_pills_by_background_subtraction(image):
    """
    The main pill detection pipeline. This is highly effective for images with
    plain, consistent backgrounds.
    """
    annotated_image = image.copy()
    
    # --- Step 1: Segmentation via Background Subtraction ---
    # Convert to HSV, which is great for color-based segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for the light gray/white background
    # This captures low saturation (colorfulness) and high value (brightness)
    lower_background = np.array([0, 0, 150])
    upper_background = np.array([180, 60, 255])
    
    # Create a mask of the background
    background_mask = cv2.inRange(hsv, lower_background, upper_background)
    
    # Invert the mask to get the foreground (the pills)
    pills_mask = cv2.bitwise_not(background_mask)
    
    # --- Step 2: Clean the Pill Mask ---
    # Use Morphological Closing to fill small holes inside the pills (e.g., from text)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(pills_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # --- Step 3: Find and Analyze Pill Contours ---
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pill_count = 0
    for c in contours:
        # Filter out contours that are too small to be pills
        if cv2.contourArea(c) < 500:
            continue
            
        pill_count += 1
        shape, color = get_pill_properties(image, c)
        
        # Draw bounding box and label
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{shape}, {color}"
        cv2.putText(annotated_image, label_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated_image, pill_count

# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Intelligent Pill Detector")
st.write("Upload an image. The system will automatically detect and classify all pills against a plain background.")

# Initialize session state
if 'img' not in st.session_state:
    st.session_state.img = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    original_image = np.array(pil_image.convert('RGB'))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    st.session_state.img = cv2.resize(original_image, (800, int(800 * original_image.shape[0] / original_image.shape[1])))

# Main display area
col1, col2 = st.columns(2)
with col1:
    if st.session_state.img is not None:
        st.subheader("Your Image")
        st.image(st.session_state.img, channels="BGR")
    else:
        st.info("Please upload an image to begin.")

# Sidebar controls
st.sidebar.title("Controls")
mode = st.sidebar.radio("Select Mode", ("Automatic Full Image", "Manual ROI Selection"))

with col2:
    st.subheader("Detection Result")
    if st.session_state.img is not None:
        if mode == "Automatic Full Image":
            # The app works automatically now
            annotated_image, pill_count = detect_pills_by_background_subtraction(st.session_state.img)
            st.image(annotated_image, channels="BGR", caption=f"Automatically found {pill_count} pill(s)")
            st.success(f"Automatic detection complete. Found {pill_count} pill(s).")
        
        elif mode == "Manual ROI Selection":
            st.warning("Draw a box on the image below and click the button.")
            # Use a different key for the cropper to avoid state issues
            cropped_img_pil = st_cropper(Image.fromarray(cv2.cvtColor(st.session_state.img, cv2.COLOR_BGR2RGB)), key="cropper")
            
            if st.button("Detect Pills in Selected Region"):
                cropped_img_cv = cv2.cvtColor(np.array(cropped_img_pil), cv2.COLOR_RGB2BGR)
                annotated_roi, pill_count = detect_pills_by_background_subtraction(cropped_img_cv)
                st.image(annotated_roi, channels="BGR", caption=f"Found {pill_count} pill(s) in ROI")
                st.success(f"Manual ROI detection complete. Found {pill_count} pill(s).")
