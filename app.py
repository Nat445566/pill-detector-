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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Caching the image processing to prevent re-computation
# This function takes the raw bytes of the image as input
# --------------------------------------------------------------------------------
@st.cache_data
def process_and_resize_image(file_bytes, max_width=700):
    """
    Loads image bytes, decodes with OpenCV, converts to RGB, resizes,
    and returns a PIL Image. This is cached so it only runs once per upload.
    """
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Decode the image using OpenCV
    opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert from BGR (OpenCV) to RGB (Pillow)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(rgb_image)
    
    # Resize the image while maintaining aspect ratio
    if pil_image.width > max_width:
        ratio = max_width / float(pil_image.width)
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    return pil_image

# --------------------------------------------------------------------------------
# Core Image Processing Function for Counting
# --------------------------------------------------------------------------------
def count_pills(image_cv, roi_coords):
    """Processes an OpenCV image to count pills based on a selected ROI."""
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, binary_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = roi_coords
    if w <= 0 or h <= 0:
        return cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), 0, "Error: The drawn box has no area."

    roi_area = w * h
    min_area = roi_area * 0.5
    max_area = roi_area * 1.5

    pill_count = 0
    output_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            pill_count += 1
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)

    return output_image, pill_count, None

# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Upload an image, draw a box around a single sample pill, and press 'Count Pills' to get the total count.")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
        1.  **Upload Image:** Select an image of pills.
        2.  **Select Sample Pill:** Draw a rectangle around ONE complete pill.
        3.  **Count Pills:** Click the 'Count Pills' button.
    """)
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    stroke_width = st.slider("Box Stroke Width: ", 1, 25, 3)
    stroke_color = st.color_picker("Box Stroke Color: ", "#00FF00")

if uploaded_file is not None:
    # Read the file bytes ONCE
    file_bytes = uploaded_file.getvalue()
    
    # Call the cached function to process the bytes and get a displayable PIL image
    display_image = process_and_resize_image(file_bytes)
    
    st.subheader("Step 1: Draw a box around a sample pill")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=display_image,  # Use the processed image for the canvas
        update_streamlit=True,
        height=display_image.height,
        width=display_image.width,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][0]
        roi_coords = (int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"]))
        
        if roi_coords[2] > 0 and roi_coords[3] > 0:
            st.subheader("Step 2: Process the image")
            if st.button('Count Pills'):
                with st.spinner('Analyzing image...'):
                    # Convert the display image to OpenCV format for processing
                    image_to_process = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                    
                    # Run the counting function
                    result_image, count, error_msg = count_pills(image_to_process, roi_coords)
                    
                    if error_msg:
                        st.error(error_msg)
                    else:
                        st.subheader("Results")
                        st.image(result_image, caption="Processed Image", use_column_width=True)
                        st.success(f"**Total Pills Counted: {count}**")
else:
    st.info("Please upload an image to begin.")
