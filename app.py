import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 Pill Counter", page_icon="💊", layout="wide")

# --- Load Your Custom Trained YOLOv8 Model ---
# This uses a try-except block to handle errors if the model file is missing.
try:
    # Place your trained model file 'best.pt' in the same folder as this app.py file.
    model = YOLO('best.pt') 
except Exception as e:
    st.error(f"Error loading YOLOv8 model: {e}")
    st.error("Please make sure you have a trained model file named 'best.pt' in your repository.")
    st.stop()

st.title("💊 Smart Pill Counting System (Powered by YOLOv8)")
st.markdown("This AI-powered tool can count pills in an entire image or just within a selected area.")

# --- Initialize Session State ---
if 'display_image' not in st.session_state:
    st.session_state.display_image = None
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# --- Sidebar for Uploads and Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    uploaded_file = st.file_uploader("1. Upload your image", type=["jpg", "jpeg", "png"])
    
    analysis_mode = st.radio(
        "2. Choose Analysis Mode",
        ('Count All Pills (Full Image)', 'Count in a Selected Area (ROI)')
    )

# --- Logic to handle image loading and state ---
if uploaded_file is not None:
    if uploaded_file.file_id != st.session_state.current_file_id:
        st.session_state.current_file_id = uploaded_file.file_id
        
        image = Image.open(uploaded_file)
        if image.width > 800:
            ratio = 800 / image.width
            new_height = int(image.height * ratio)
            image = image.resize((800, new_height), Image.Resampling.LANCZOS)
        
        st.session_state.display_image = image

# --- Main app logic based on selected mode ---
if st.session_state.display_image is None:
    st.info("Please upload an image to get started.")
else:
    display_image = st.session_state.display_image
    
    # --- ROI ANALYSIS MODE ---
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
                with st.spinner("AI is analyzing the selected area..."):
                    rect = canvas_result.json_data["objects"][0]
                    x, y, w, h = int(rect["left"]), int(rect["top"]), int(rect["width"]), int(rect["height"])
                    
                    if w > 0 and h > 0:
                        # Crop the PIL image
                        cropped_pil_image = display_image.crop((x, y, x + w, y + h))
                        
                        # Run YOLOv8 detection ONLY on the cropped image
                        results = model(cropped_pil_image)
                        pill_count = len(results[0].boxes)
                        
                        # Get the cropped image with detections drawn on it
                        cropped_result_img = results[0].plot() # This is a NumPy array
                        
                        # Create a final output image from the original display image
                        output_image = np.array(display_image).copy()
                        
                        # Paste the processed crop back onto the full image
                        output_image[y:y+h, x:x+w] = cv2.cvtColor(cropped_result_img, cv2.COLOR_BGR2RGB)

                        st.subheader("Results")
                        st.image(output_image, caption=f"Detected {pill_count} pills inside the selected area.")
                        st.success(f"**Total Pills Counted: {pill_count}**")

    # --- FULL IMAGE ANALYSIS MODE ---
    elif analysis_mode == 'Count All Pills (Full Image)':
        st.subheader("Step 1: Review Full Image")
        st.image(display_image, caption="Uploaded Image")
        
        st.subheader("Step 2: Process the image")
        if st.button("Count All Pills"):
            with st.spinner("AI is analyzing the full image..."):
                # Run YOLOv8 detection on the entire image
                results = model(display_image)
                pill_count = len(results[0].boxes)
                
                # Get the image with bounding boxes drawn
                output_image_np = results[0].plot()
                output_image_rgb = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2RGB)
                
                st.subheader("Results")
                st.image(output_image_rgb, caption=f"Detected {pill_count} pills in the image.")
                st.success(f"**Total Pills Counted: {pill_count}**")
