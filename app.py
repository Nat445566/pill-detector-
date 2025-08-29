import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Pill Counter",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------------------------------------
# Image Processing Pipeline (Based on BMDS2133 Handbook)
# --------------------------------------------------------------------------------
def get_pill_contours(image_cv):
    """
    Takes an OpenCV image and returns the contours of the pills. This function
    is now run ONLY on the cropped section of the main image.
    """
    # Practical 1 & 3: Grayscale Conversion
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Handbook p. 38: Gaussian Blurring for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Handbook p. 32: Adaptive Thresholding for robust segmentation
    binary_mask = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=15, C=4
    )
    # Practical 7: Morphological Operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find and return contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --------------------------------------------------------------------------------
# Streamlit UI and State Management
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Upload an image, draw a box around the area you want to analyze, and click 'Count Pills'.")

# Initialize session state for the image
if 'display_image' not in st.session_state:
    st.session_state.display_image = None

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"], key="uploader")

    # This logic processes and stores the image in session_state ONCE per upload
    if uploaded_file is not None:
        if "current_file_id" not in st.session_state or st.session_state.current_file_id != uploaded_file.id:
            st.session_state.current_file_id = uploaded_file.id
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Resize for a better UI experience
            if pil_image.width > 800:
                ratio = 800 / pil_image.width
                new_height = int(pil_image.height * ratio)
                pil_image = pil_image.resize((800, new_height), Image.Resampling.LANCZOS)
            
            st.session_state.display_image = pil_image

# Main app logic: Renders only if an image is in the session state
if st.session_state.display_image is not None:
    display_image = st.session_state.display_image
    
    st.subheader("Step 1: Draw a box to define the counting area")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=display_image,
        height=display_image.height,
        width=display_image.width,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        st.subheader("Step 2: Process the selected area")
        if st.button("Count Pills in Selected Area"):
            with st.spinner("Analyzing..."):
                rect = canvas_result.json_data["objects"][0]
                x, y, w, h = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])

                if w > 0 and h > 0:
                    # Convert the full display image to OpenCV format
                    full_image_cv = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                    
                    # CROP the image to the selected ROI
                    cropped_image = full_image_cv[y:y+h, x:x+w]
                    
                    # Run the contour detection ONLY on the cropped image
                    contours_in_crop = get_pill_contours(cropped_image)
                    
                    pill_count = len(contours_in_crop)
                    
                    # Create a copy of the full image to draw the results on
                    output_image = np.array(display_image).copy()
                    
                    # Draw the detected contours, making sure to add the (x, y) offset
                    # so they appear in the correct location on the full image.
                    cv2.drawContours(output_image, contours_in_crop, -1, (0, 255, 0), 3, offset=(x, y))
                    
                    st.subheader("Results")
                    st.image(output_image, caption=f"Found {pill_count} pills inside the selected area.")
                    st.success(f"**Total Pills Counted: {pill_count}**")

else:
    st.info("Please upload an image to get started.")
