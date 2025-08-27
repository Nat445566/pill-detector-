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
# Core Image Processing Function
# --------------------------------------------------------------------------------
def count_pills(image, roi_coords):
    """
    Processes the uploaded image to count pills based on a selected ROI.
    
    Args:
        image (numpy.ndarray): The full image uploaded by the user.
        roi_coords (tuple): A tuple (x, y, w, h) defining the ROI.

    Returns:
        tuple: A tuple containing the final image with drawings and the pill count.
    """
    # --- 1. Preprocessing ---
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # --- 2. Segmentation using Thresholding ---
    _, binary_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- 3. Cleaning with Morphological Operations ---
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 4. Contour Detection and Filtering ---
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 5. Calculate ROI area for filtering ---
    x, y, w, h = roi_coords
    roi_area = w * h
    min_area = roi_area * 0.5  # Allow for 50% smaller
    max_area = roi_area * 1.5  # Allow for 50% larger

    pill_count = 0
    # Convert original image to RGB for displaying with Streamlit
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 6. Loop, Filter, and Draw ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            pill_count += 1
            # Draw green bounding boxes around counted pills
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 3)

    return output_image, pill_count


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.title("💊 Smart Pill Counting System")
st.markdown("Upload an image, draw a box around a single sample pill, and press 'Count Pills' to get the total count.")

# --- Sidebar for Instructions and Upload ---
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
        1.  **Upload Image:** Use the file uploader to select an image of pills.
        2.  **Select Sample Pill:** A canvas will appear. Draw a rectangle around ONE complete pill to use it as a sample.
        3.  **Count Pills:** Click the 'Count Pills' button to start the analysis.
    """)
    
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Drawing canvas parameters
    stroke_width = st.slider("Box Stroke Width: ", 1, 25, 3)
    stroke_color = st.color_picker("Box Stroke Color: ", "#00FF00")
    drawing_mode = "rect"

# --- Main Content Area ---
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file)
    st_image = np.array(pil_image)
    
    # Check if the image has an alpha channel and remove it
    if st_image.shape[2] == 4:
        st_image = st_image[:, :, :3]
        
    # Convert RGB from PIL to BGR for OpenCV
    cv_image = cv2.cvtColor(st_image, cv2.COLOR_RGB2BGR)

    st.subheader("Step 1: Draw a box around a sample pill")
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=pil_image,
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Check if the user has drawn a rectangle
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        # Extract ROI coordinates from the drawn rectangle
        rect = canvas_result.json_data["objects"][0]
        left, top, width, height = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])
        
        # Ensure coordinates are valid
        if width > 0 and height > 0:
            roi_coords = (left, top, width, height)
            
            st.subheader("Step 2: Process the image")
            if st.button('Count Pills'):
                with st.spinner('Processing...'):
                    # Perform the pill counting
                    result_image, count = count_pills(cv_image, roi_coords)

                    # Display the results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(result_image, caption="Processed Image", use_column_width=True)
                    with col2:
                        st.metric(label="Total Pills Counted", value=count)
                        st.info("Note: The count is based on objects similar in size to your selection.")

else:
    st.info("Please upload an image to begin.")
