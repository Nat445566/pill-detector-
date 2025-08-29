import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Pill Counter (OpenCV)",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------------------------------------
# Image Processing Pipeline (Based on BMDS2133 Handbook)
# --------------------------------------------------------------------------------
def get_pill_contours(image_cv, show_steps=False):
    """
    Takes an OpenCV image and returns the contours of the pills using a
    pipeline based on the practical handbook.
    """
    # Practical 1 & 3: Grayscale Conversion
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Handbook p. 38: Gaussian Blurring for noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Handbook p. 32: USE ADAPTIVE THRESHOLDING for uneven lighting
    # This is the key change for better accuracy on complex backgrounds.
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
            col1.image(gray_image, caption="1. Grayscale", use_column_width=True)
            col2.image(binary_mask, caption="2. After Adaptive Thresholding", use_column_width=True)
            col3.image(cleaned_mask, caption="3. Cleaned Mask", use_column_width=True)
            
    return contours

# --------------------------------------------------------------------------------
# Streamlit UI and State Management
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System (OpenCV Edition)")
st.markdown("Built using the **BMDS2133 Image Processing Handbook**. This tool allows you to count pills from an image using two different methods.")

# Initialize session state variables
if 'display_image' not in st.session_state:
    st.session_state.display_image = None
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("1. Upload your image", type=["jpg", "jpeg", "png"])
    
    # Process and store the image in session_state ONCE per new upload
    if uploaded_file and (st.session_state.current_file_id != uploaded_file.file_id):
        st.session_state.current_file_id = uploaded_file.file_id
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        if pil_image.width > 700:
            ratio = 700 / pil_image.width
            new_height = int(pil_image.height * ratio)
            pil_image = pil_image.resize((700, new_height), Image.Resampling.LANCZOS)
        
        st.session_state.display_image = pil_image

    analysis_mode = st.radio(
        "2. Choose Analysis Mode",
        ('Count in a Selected Area (ROI)', 'Count All Pills (Full Image)')
    )

    if analysis_mode == 'Count All Pills (Full Image)':
        st.subheader("Full Analysis Settings")
        min_area = st.slider("Minimum Pill Area (pixels)", 100, 5000, 500)
        max_area = st.slider("Maximum Pill Area (pixels)", 500, 20000, 10000)

# --- Main app logic ---
if st.session_state.display_image is not None:
    display_image = st.session_state.display_image
    
    if analysis_mode == 'Count in a Selected Area (ROI)':
        st.subheader("Step 1: Draw a box to define the counting area")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            background_image=display_image,
            height=display_image.height,
            width=display_image.width,
            drawing_mode="rect",
            key="canvas_roi"
        )

        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            st.subheader("Step 2: Process the selected area")
            if st.button("Count Pills in Selected Area"):
                with st.spinner("Analyzing..."):
                    rect = canvas_result.json_data["objects"][0]
                    x, y, w, h = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])
                    
                    if w > 0 and h > 0:
                        full_image_cv = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                        cropped_image = full_image_cv[y:y+h, x:x+w]
                        contours_in_crop = get_pill_contours(cropped_image, show_steps=True)
                        pill_count = len(contours_in_crop)
                        
                        output_image = np.array(display_image).copy()
                        cv2.drawContours(output_image, contours_in_crop, -1, (0, 255, 0), 2, offset=(x, y))
                        
                        st.subheader("Results")
                        st.image(output_image, caption=f"Found {pill_count} pills inside the selected area.")
                        st.success(f"**Total Pills Counted: {pill_count}**")

    elif analysis_mode == 'Count All Pills (Full Image)':
        st.subheader("Step 1: Review Full Image")
        st.image(display_image, caption="Uploaded Image")
        st.markdown(f"The app will count pills with an area between **{min_area}** and **{max_area}** pixels.")
        
        if st.button("Count All Pills"):
            with st.spinner("Analyzing..."):
                image_to_process = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                contours = get_pill_contours(image_to_process, show_steps=True)
                
                pill_count = 0
                output_image = np.array(display_image).copy()
                
                for cnt in contours:
                    if min_area < cv2.contourArea(cnt) < max_area:
                        pill_count += 1
                        cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
                
                st.subheader("Results")
                st.image(output_image, caption=f"Found {pill_count} pills within the specified size range.")
                st.success(f"**Total Pills Counted: {pill_count}**")
else:
    st.info("Please upload an image to get started.")
