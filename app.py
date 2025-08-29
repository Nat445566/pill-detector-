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
# ADVANCED Image Processing Pipeline (Using Adaptive Thresholding from Handbook p. 32)
# --------------------------------------------------------------------------------
def get_pill_contours(image_cv, show_steps=False):
    """
    This function uses a more robust pipeline based on the BMDS2133 Handbook
    to handle images with complex backgrounds and lighting.
    """
    # Practical 1 & 3: Grayscale Conversion
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Handbook p. 38: Gaussian Blurring for noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Handbook p. 32: USE ADAPTIVE THRESHOLDING for uneven lighting
    # This is the key change for better accuracy on complex backgrounds.
    # It calculates a threshold for small regions, making it robust to shadows.
    binary_mask = cv2.adaptiveThreshold(
        blurred_image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # Use INV since pills are often darker than background
        blockSize=15, C=4 
    )

    # Practical 7 (p. 64-67): Morphological Operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    # Closing fills small holes inside the pills
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Opening removes small noise specks around the pills
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    # Find and return the final contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Optional: Display intermediate steps for debugging
    if show_steps:
        with st.expander("Show Processing Steps"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(gray_image, caption="1. Grayscale", use_column_width=True)
            with col2:
                st.image(binary_mask, caption="2. After Adaptive Thresholding", use_column_width=True)
            with col3:
                st.image(cleaned_mask, caption="3. Cleaned Mask", use_column_width=True)
            
    return contours

# --------------------------------------------------------------------------------
# Streamlit UI and State Management
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Based on the **BMDS2133 Image Processing Handbook**. This tool allows you to count pills from an image using two different methods.")

# Initialize session state variables if they don't exist
if 'display_image' not in st.session_state:
    st.session_state.display_image = None
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_file = st.file_uploader("1. Upload your image", type=["jpg", "jpeg", "png"], key="uploader")
    
    # This is the key logic: Process the image only when a new file is uploaded
    if uploaded_file is not None and uploaded_file.file_id != st.session_state.current_file_id:
        st.session_state.current_file_id = uploaded_file.file_id
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Resize for display
        if pil_image.width > 700:
            ratio = 700 / pil_image.width
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((700, new_height), Image.Resampling.LANCZOS)
        
        # Save the processed, displayable image to the session state.
        st.session_state.display_image = pil_image

    analysis_mode = st.radio(
        "2. Choose Analysis Mode",
        ('Count with ROI (Select a Sample)', 'Count All Pills (Full Image Analysis)')
    )

    if analysis_mode == 'Count All Pills (Full Image Analysis)':
        st.subheader("Full Analysis Settings")
        min_area = st.slider("Minimum Pill Area (pixels)", 100, 5000, 500)
        max_area = st.slider("Maximum Pill Area (pixels)", 500, 20000, 10000)

# --- Main app logic: Renders using the image from session_state ---
if st.session_state.display_image is None:
    st.info("Please upload an image to get started.")
else:
    display_image = st.session_state.display_image
    
    # --- ROI ANALYSIS MODE ---
    if analysis_mode == 'Count with ROI (Select a Sample)':
        st.subheader("Step 1: Draw a box around a sample pill")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            background_image=display_image, # This now correctly and persistently uses the image
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
                        contours = get_pill_contours(image_to_process, show_steps=True)
                        
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
    
    # --- FULL IMAGE ANALYSIS MODE ---
    elif analysis_mode == 'Count All Pills (Full Image Analysis)':
        st.subheader("Step 1: Review Image and Settings")
        st.image(display_image, caption="Uploaded Image")
        st.markdown(f"The app will count pills with an area between **{min_area}** and **{max_area}** pixels.")
        
        st.subheader("Step 2: Process the image")
        if st.button("Count All Pills"):
            with st.spinner("Analyzing..."):
                image_to_process = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                contours = get_pill_contours(image_to_process, show_steps=True)
                
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
