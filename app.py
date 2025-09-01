import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Core Image Processing Functions ---

def get_pill_properties(image_bgr, contour):
    """
    Analyzes a single, confirmed pill contour to determine its shape and color.
    """
    # 1. Shape Analysis using a robust circularity calculation
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    shape = "Unknown"
    if perimeter > 0:
        # A perfect circle has a circularity of 1.0
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.85:
            shape = "Round"
        elif circularity > 0.60:
            shape = "Oval"
        else:
            shape = "Capsule" # Long, thin shapes have low circularity

    # 2. Color Analysis using the average color inside the contour
    mask = np.zeros(image_bgr.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Convert to HSV for robust color detection
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(image_hsv, mask=mask)[:3]
    h, s, v = mean_hsv
    
    color = "Unknown"
    # Tuned HSV ranges for better accuracy in various lighting
    if s < 65 and v > 160: color = "White"
    elif 35 < h < 85 and s > 60: color = "Green"
    elif (h < 12 or h > 168) and s > 80: color = "Red"
    elif 10 < h < 35 and s > 80: color = "Brown/Orange"
    elif 85 < h < 130 and s > 70: color = "Blue"
    
    return shape, color

def detect_pills(image, min_area, max_area, circularity_threshold, solidity_threshold):
    """
    The main pill detection pipeline. Finds, filters, and analyzes contours
    to identify objects that are exclusively pills.
    """
    # Create a copy to draw annotations on
    annotated_image = image.copy()
    
    # --- 1. Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Median Blur is excellent for removing "salt-and-pepper" noise while keeping object edges sharp
    blurred = cv2.medianBlur(gray, 5)

    # --- 2. Segmentation using Canny Edge Detection ---
    # This is much more robust for complex backgrounds than simple thresholding
    edges = cv2.Canny(blurred, 50, 150)
    
    # --- 3. Find All Potential Objects (Contours) ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pills_found_contours = []
    
    # --- 4. The Critical "Pill-ness" Filter Pipeline ---
    # Each contour is tested. Only those passing all tests are considered pills.
    for c in contours:
        # a) Filter by Area: Is the object the right size?
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue
            
        # b) Filter by Circularity: Is the object round or oval? 
        # (This is the MOST IMPORTANT filter for ignoring background patterns)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < circularity_threshold:
            continue
            
        # c) Filter by Solidity: Is the object a solid shape (not hollow)?
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < solidity_threshold:
            continue
        
        # If a contour passes all checks, it's a pill!
        pills_found_contours.append(c)

    # --- 5. Analyze and Annotate the Confirmed Pills ---
    for pill_contour in pills_found_contours:
        shape, color = get_pill_properties(image, pill_contour)
        
        # Draw bounding box and the final classification label
        x, y, w, h = cv2.boundingRect(pill_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        label_text = f"{shape}, {color}"
        cv2.putText(annotated_image, label_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated_image, len(pills_found_contours)


# --- Streamlit Web App Interface ---

st.set_page_config(layout="wide")
st.title("Accurate Pill Detector and Identifier")

st.write("""
Upload an image containing pills. This application uses a multi-stage filtering pipeline to detect **only pill-like objects** and ignore complex backgrounds. 
Use the sliders in the sidebar to **fine-tune the detection sensitivity** for your specific image.
""")

# --- Sidebar for User-Adjustable Parameters ---
st.sidebar.title("Detection Controls")
min_area = st.sidebar.slider("1. Minimum Pill Area", 100, 5000, 500, help="Filters out small noise. Increase if small pills are missed.")
max_area = st.sidebar.slider("2. Maximum Pill Area", 5000, 100000, 40000, help="Filters out objects that are too large.")
circularity_threshold = st.sidebar.slider("3. Circularity Threshold", 0.1, 1.0, 0.65, 0.01, help="The most important filter. Higher values mean 'more round'. Lower this to detect very long capsules.")
solidity_threshold = st.sidebar.slider("4. Solidity Threshold", 0.1, 1.0, 0.85, 0.01, help="Filters out objects with concave shapes or holes. Pills should be very solid (close to 1.0).")

# --- Main Page for Upload and Display ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file).convert('RGB')
    original_image = np.array(pil_image)
    # Convert RGB (from PIL) to BGR (for OpenCV)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Resize for consistent processing and display
    processing_image = cv2.resize(original_image, (800, int(800 * original_image.shape[0] / original_image.shape[1])))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(processing_image, channels="BGR", caption="Uploaded Image")
    
    # Perform detection
    annotated_image, pill_count = detect_pills(
        processing_image, 
        min_area, 
        max_area, 
        circularity_threshold,
        solidity_threshold
    )
    
    with col2:
        st.image(annotated_image, channels="BGR", caption=f"Detection Result: {pill_count} pill(s) found")
    
    if pill_count > 0:
        st.success(f"Successfully detected {pill_count} pill(s)!")
    else:
        st.warning("No objects matching the filter criteria were found. Please try adjusting the sliders in the sidebar.")
