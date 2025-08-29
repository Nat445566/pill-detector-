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
# Core Image Processing Pipeline (Based on BMDS2133 Handbook)
# --------------------------------------------------------------------------------
def get_pill_contours(image_cv):
    """
    Takes an OpenCV image and performs operations based on the practical handbook
    to return the contours of the pills.
    """
    # Grayscale Conversion, Gaussian Blurring, and Otsu's Binarization
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, binary_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological Operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find and return contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Based on the **BMDS2133 Image Processing Handbook**. This tool allows you to count pills from an image using two different methods.")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Using a key is important for state management
    uploaded_file = st.file_uploader("1. Upload your image", type=["jpg", "jpeg", "png"], key="uploader")
    
    analysis_mode = st.radio(
        "2. Choose Analysis Mode",
        ('Count with ROI (Select a Sample)', 'Count All Pills (Full Image Analysis)'),
        key="analysis_mode"
    )

    # Sliders for Full Analysis mode
    if analysis_mode == 'Count All Pills (Full Image Analysis)':
        st.subheader("Full Analysis Settings")
        min_area = st.slider("Minimum Pill Area", 100, 5000, 500)
        max_area = st.slider("Maximum Pill Area", 500, 20000, 10000)

# --- Logic to handle image loading and state correctly ---
# This block runs only when a new file is uploaded
if uploaded_file is not None:
    # Check if this is a new file. If so, process and store it.
    if 'current_file_id' not in st.session_state or st.session_state.current_file_id != uploaded_file.file_id:
        st.session_state.current_file_id = uploaded_file.file_id
        
        # Read the file bytes and process the image ONCE
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Resize for display
        if pil_image.width > 700:
            ratio = 700 / pil_image.width
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((700, new_height), Image.Resampling.LANCZOS)
        
        # Store the final, displayable image in session state. THIS IS THE KEY FIX.
        st.session_state.display_image = pil_image

# --- Main App Logic: Renders based on what's in session_state ---
# This part of the code is now safe from re-runs because it uses the saved state.
if 'display_image' in st.session_state:
    display_image = st.session_state.display_image
    
    if analysis_mode == 'Count with ROI (Select a Sample)':
        st.subheader("Step 1: Draw a box around a sample pill")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            background_image=display_image,  # Always uses the saved image
            height=display_image.height,
            width=display_image.width,
            drawing_mode="rect",
            key="canvas_roi"
        )

        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            st.subheader("Step 2: Process the image")
            if st.button("Count Pills using ROI"):
                with st.spinner("Analyzing..."):
                    rect = canvas_result.json_data["objects"][0]
                    roi_w, roi_h = rect['width'], rect['height']
                    
                    if roi_w > 0 and roi_h > 0:
                        roi_area = roi_w * roi_h
                        image_to_process = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                        contours = get_pill_contours(image_to_process)
                        
                        pill_count = 0
                        output_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
                        
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if (roi_area * 0.5) < area < (roi_area * 1.5):
                                pill_count += 1
                                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)

                        st.subheader("Results")
                        st.image(output_image, caption=f"Found {pill_count} pills similar to the sample.")
                        st.success(f"**Total Pills Counted: {pill_count}**")

    elif analysis_mode == 'Count All Pills (Full Image Analysis)':
        st.subheader("Step 1: Review Image and Settings")
        st.image(display_image, caption="Uploaded Image") # Works because it uses the saved image
        st.markdown(f"The app will count pills with an area between **{min_area}** and **{max_area}** pixels.")
        
        st.subheader("Step 2: Process the image")
        if st.button("Count All Pills"):
            with st.spinner("Analyzing..."):
                image_to_process = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                contours = get_pill_contours(image_to_process)
                
                pill_count = 0
                output_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_area < area < max_area:
                        pill_count += 1
                        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)
                
                st.subheader("Results")
                st.image(output_image, caption=f"Found {pill_count} pills within the specified size range.")
                st.success(f"**Total Pills Counted: {pill_count}**")
else:
    st.info("Please upload an image to get started.")
