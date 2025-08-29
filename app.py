import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Smart Pill Counter", page_icon="💊", layout="wide")

# --------------------------------------------------------------------------------
# Core Image Processing Function (Based on BMDS2133 Handbook)
# --------------------------------------------------------------------------------
def get_pill_contours(image_cv):
    """
    Takes an OpenCV image and returns the contours of the pills using a
    pipeline based on the practical handbook.
    """
    # Convert to Grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Adaptive Thresholding for better results on varied backgrounds
    binary_mask = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, C=4
    )
    # Clean up the mask with Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find and return contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Upload an image, draw a box around the area you want to analyze, and click 'Count Pills'.")

# --- Sidebar for file upload and settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

# --- Main app logic ---
# Initialize session state for the background image
if 'bg_image' not in st.session_state:
    st.session_state.bg_image = None

# If a new file is uploaded, update the session state
if uploaded_file is not None:
    # Read the file bytes and convert to a PIL Image
    image = Image.open(uploaded_file)
    
    # Resize image for a better UI experience
    if image.width > 800:
        ratio = 800 / image.width
        new_height = int(image.height * ratio)
        image = image.resize((800, new_height), Image.Resampling.LANCZOS)
    
    # This is the key: store the image in session_state
    st.session_state.bg_image = image

# Only display the canvas and the rest of the app if an image has been loaded into the state
if st.session_state.bg_image is not None:
    st.subheader("Step 1: Draw a box to define the counting area")
    
    # The canvas now reliably gets its background image from the session state
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=st.session_state.bg_image,
        height=st.session_state.bg_image.height,
        width=st.session_state.bg_image.width,
        drawing_mode="rect",
        key="canvas"
    )

    # If a rectangle has been drawn, show the button
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        st.subheader("Step 2: Process the selected area")
        if st.button("Count Pills in Selected Area"):
            with st.spinner("Analyzing..."):
                rect = canvas_result.json_data["objects"][0]
                x, y, w, h = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])

                if w > 0 and h > 0:
                    # Get the full image from session state and convert for processing
                    full_image_pil = st.session_state.bg_image
                    full_image_cv = cv2.cvtColor(np.array(full_image_pil), cv2.COLOR_RGB2BGR)
                    
                    # Crop the image to the selected ROI
                    cropped_image = full_image_cv[y:y+h, x:x+w]
                    
                    # Run contour detection ONLY on the cropped image
                    contours_in_crop = get_pill_contours(cropped_image)
                    
                    pill_count = len(contours_in_crop)
                    
                    # Create a copy of the full image to draw the results on
                    output_image = np.array(full_image_pil).copy()
                    
                    # Draw the detected contours, making sure to add the (x, y) offset
                    # so they appear in the correct location on the full image.
                    cv2.drawContours(output_image, contours_in_crop, -1, (0, 255, 0), 3, offset=(x, y))
                    
                    st.subheader("Results")
                    st.image(output_image, caption=f"Found {pill_count} pills inside the selected area.")
                    st.success(f"**Total Pills Counted: {pill_count}**")
else:
    st.info("Please upload an image to get started.")
