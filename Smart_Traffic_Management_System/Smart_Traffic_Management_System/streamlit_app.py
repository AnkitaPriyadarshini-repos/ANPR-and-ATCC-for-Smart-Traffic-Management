"""Smart Traffic Management System - Streamlit Application

A YOLO-based detection system for ANPR and ATCC with real-time analytics.
License: MIT
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os
import pandas as pd
from pathlib import Path
import time
import cv2
try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_OPTIONS = {
    "ANPR (License Plate Detection)": "yolo_ANPR.pt",
    "ATCC (Traffic Classification)": "yolo_ATCC.pt",
}

ATCC_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

ANPR_CLASSES = ["license plate"]

# Dataset paths (using Path for cross-platform compatibility)
# Navigate from script location: streamlit_app.py -> Smart_Traffic_Management_System -> Smart_Traffic_Management_System -> STMS root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Fallback check: if ANPR not found at expected location, try current working directory
if not (PROJECT_ROOT / "anpr").exists():
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "anpr").exists():
        # Last resort: search parent directories
        for parent in Path.cwd().parents:
            if (parent / "anpr").exists():
                PROJECT_ROOT = parent
                break

ANPR_IMAGES_PATH = PROJECT_ROOT / "anpr" / "images"
ATCC_IMAGES_PATH = PROJECT_ROOT / "ATCC" / "bdd100k" / "bdd100k" / "images" / "100k"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_sample_images(model_choice, max_samples=200):
    """Get sample images from dataset based on model choice.
    
    Args:
        model_choice: Selected model name
        max_samples: Maximum number of samples to return
        
    Returns:
        List of image file paths
    """
    base_path = ANPR_IMAGES_PATH if "ANPR" in model_choice else ATCC_IMAGES_PATH
    
    image_files = []
    for folder in ['train', 'val', 'test']:
        folder_path = base_path / folder
        if folder_path.exists():
            # Get jpg and png files - increased from 20 to 100 per type
            jpg_files = list(folder_path.glob("*.jpg"))[:100]
            png_files = list(folder_path.glob("*.png"))[:100]
            image_files.extend([str(f) for f in jpg_files + png_files])
    
    return image_files[:max_samples]

@st.cache_data(ttl=3600)
def get_dataset_stats():
    """Get dataset statistics for display."""
    stats = {}
    
    # ANPR stats
    if ANPR_IMAGES_PATH.exists():
        train_count = len(list((ANPR_IMAGES_PATH / "train").glob("*.jpg")))
        val_count = len(list((ANPR_IMAGES_PATH / "val").glob("*.jpg")))
        test_count = len(list((ANPR_IMAGES_PATH / "test").glob("*.jpg")))
        stats['anpr'] = {
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'total': train_count + val_count + test_count
        }
    
    # ATCC stats
    if ATCC_IMAGES_PATH.exists():
        train_count = len(list((ATCC_IMAGES_PATH / "train").glob("*.jpg")))
        val_count = len(list((ATCC_IMAGES_PATH / "val").glob("*.jpg")))
        stats['atcc'] = {
            'train': train_count,
            'val': val_count,
            'total': train_count + val_count
        }
    
    return stats

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path):
    """Load and cache YOLO model for efficient reuse.
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        Loaded YOLO model or None if error
    """
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource(show_spinner="Initializing OCR...")
def load_ocr():
    """Load and cache OCR reader for text extraction."""
    if HAS_OCR:
        try:
            return easyocr.Reader(['en'])
        except Exception as e:
            st.warning(f"OCR not available: {e}")
            return None
    return None

# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .stAlert {border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("Smart Traffic Management System")
st.markdown("### YOLO Detection - ANPR and ATCC Technology")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("Configuration")
    
    # Model Selection
    model_choice = st.selectbox(
        "Select Detection Model",
        list(MODEL_OPTIONS.keys()),
        help="Choose between license plate detection or traffic classification"
    )
    
    model_filename = MODEL_OPTIONS[model_choice]
    # Get the full path to the model in the same directory as this script
    model_path = Path(__file__).parent / model_filename
    model = load_model(str(model_path))
    
    if model is None:
        st.error("Failed to load model. Please check if model file exists.")
        st.stop()
    
    st.success(f"Model loaded: {os.path.basename(str(model_path))}")
    
    # Detection Settings
    st.divider()
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Image Source Selection
    st.divider()
    st.header("Image Source")
    image_source = st.radio(
        "Choose image source:",
        ["Dataset Samples", "Upload Your Own"],
        help="Select images from dataset or upload your own"
    )
    
    # Model Info
    st.divider()
    st.header("Model Info")
    if "ANPR" in model_choice:
        st.info(f"**Classes:** {len(ANPR_CLASSES)}\n\n{', '.join(ANPR_CLASSES)}")
    else:
        st.info(f"**Classes:** {len(ATCC_CLASSES)}\n\n{', '.join(ATCC_CLASSES[:5])}...")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

selected_image = None
image_path = None

if "Dataset" in image_source:
    st.header("Dataset Sample Selection")
    
    # Get sample images based on model choice
    sample_images = get_sample_images(model_choice)
    
    if sample_images:
        # Create dropdown with just filenames
        image_names = [os.path.basename(img) for img in sample_images]
        selected_name = st.selectbox(
            "Select an image from the dataset:",
            ["Select an image..."] + image_names
        )
        
        if selected_name != "Select an image...":
            # Find the full path
            image_path = next(img for img in sample_images if os.path.basename(img) == selected_name)
            
            try:
                selected_image = Image.open(image_path)
                
                # Display image info
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Image Information")
                    st.write(f"**Filename:** {selected_name}")
                    st.write(f"**Size:** {selected_image.size[0]} × {selected_image.size[1]} px")
                    st.write(f"**Mode:** {selected_image.mode}")
                    st.write(f"**Format:** {selected_image.format}")
                
                with col2:
                    st.subheader("Original Image")
                    st.image(selected_image, width="stretch")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
    else:
        st.warning(f"No sample images found for {model_choice}.")
        st.info(f"Expected location: {ANPR_IMAGES_PATH if 'ANPR' in model_choice else ATCC_IMAGES_PATH}")

else:  # Upload Your Own
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file)
            st.image(selected_image, caption="Uploaded Image", width="stretch")
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")

# ============================================================================
# DETECTION & RESULTS
# ============================================================================

if selected_image is not None:
    with st.spinner(f"🔍 Running {model_choice} detection..."):
        try:
            # Start timer for performance metrics
            start_time = time.time()
            
            # Run YOLO detection
            results = model.predict(
                source=selected_image,
                conf=confidence,
                save=False,
                verbose=False
            )
            
            detection_time = time.time() - start_time
            
            # Plot results
            res_plotted = results[0].plot()[:, :, ::-1]
            result_image = Image.fromarray(res_plotted)
            
            # Display result
            st.header("Detection Results")
            st.image(result_image, caption="Detected Objects", width="stretch")
            
            # OCR processing for ANPR
            ocr_results = []
            if "ANPR" in model_choice and HAS_OCR:
                ocr_reader = load_ocr()
                if ocr_reader is not None:
                    img_array = np.array(selected_image)
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        st.subheader("License Plate Text Recognition")
                        
                        for idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            plate_region = img_cv[y1:y2, x1:x2]
                            
                            if plate_region.size > 0:
                                try:
                                    ocr_result = ocr_reader.readtext(plate_region)
                                    plate_text = ''.join([text[1] for text in ocr_result])
                                    text_confidence = np.mean([text[2] for text in ocr_result]) if ocr_result else 0
                                    
                                    ocr_results.append({
                                        "Index": idx + 1,
                                        "Detected Text": plate_text if plate_text.strip() else "Unreadable",
                                        "Text Confidence": f"{text_confidence:.2%}"
                                    })
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.text(f"Plate {idx + 1}: {plate_text}")
                                    with col2:
                                        st.metric("Confidence", f"{text_confidence:.1%}")
                                except Exception as e:
                                    st.warning(f"Could not extract text from plate {idx + 1}")
            
            # Extract detection data
            boxes = results[0].boxes
            detections = []
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "ID": i + 1,
                    "Class": model.names[cls_id],
                    "Confidence": f"{conf:.2%}",
                    "Box": f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                })
            
            # Analytics Section
            if detections:
                st.subheader("Detection Analytics")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Detections", len(detections))
                with col2:
                    unique_classes = len(set(d["Class"] for d in detections))
                    st.metric("Unique Classes", unique_classes)
                with col3:
                    avg_conf = sum(float(d["Confidence"].strip('%')) for d in detections) / len(detections)
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                with col4:
                    st.metric("Detection Time", f"{detection_time:.2f}s")
                
                # Detection table
                st.subheader("Detection Details")
                df = pd.DataFrame(detections)
                st.dataframe(df, width="stretch")
                
                # Class distribution chart
                st.subheader("Class Distribution")
                class_counts = df['Class'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                st.bar_chart(class_counts.set_index('Class'))
                
                # Confidence statistics
                st.subheader("Confidence Statistics")
                confidences = [float(d["Confidence"].strip('%')) for d in detections]
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                with conf_col1:
                    st.metric("Min Confidence", f"{min(confidences):.1f}%")
                with conf_col2:
                    st.metric("Max Confidence", f"{max(confidences):.1f}%")
                with conf_col3:
                    st.metric("Std Dev", f"{np.std(confidences):.1f}%")
                
                # Download button
                img_byte_arr = io.BytesIO()
                result_image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                st.download_button(
                    label="Download Detection Result",
                    data=img_bytes,
                    file_name=f"detection_{model_choice.split()[0]}_{time.strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    type="primary"
                )
            else:
                st.warning("No objects detected in the image. Try:")
                st.markdown("- Lowering the confidence threshold\n- Using a different image\n- Checking if the image contains relevant objects")
        
        except Exception as e:
            st.error(f"An error occurred during detection: {e}")
            st.info("Try reloading the page or selecting a different image.")

else:
    # Show dataset info when no image is selected
    if "Dataset" in image_source:
        st.info("Select an image from the dropdown above to start detection")
        
        # Dataset statistics
        st.subheader("Dataset Information")
        
        stats = get_dataset_stats()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ANPR Dataset (License Plates)**")
            if 'anpr' in stats:
                st.write(f"- Training: {stats['anpr']['train']:,} images")
                st.write(f"- Validation: {stats['anpr']['val']:,} images")
                st.write(f"- Test: {stats['anpr']['test']:,} images")
                st.write(f"- **Total: {stats['anpr']['total']:,} images**")
            else:
                st.write("Dataset not found")
        
        with col2:
            st.markdown("**ATCC Dataset (Traffic Objects)**")
            if 'atcc' in stats:
                st.write(f"- Training: {stats['atcc']['train']:,} images")
                st.write(f"- Validation: {stats['atcc']['val']:,} images")
                st.write(f"- **Total: {stats['atcc']['total']:,} images**")
            else:
                st.write("Dataset not found")

# ============================================================================
# FOOTER - MODEL INFORMATION
# ============================================================================

st.markdown("---")
with st.expander("View Model Class Information", expanded=False):
    class_col1, class_col2 = st.columns(2)
    
    with class_col1:
        st.markdown("##### ANPR Model (`yolo_ANPR.pt`)")
        st.markdown(f"**Classes:** {len(ANPR_CLASSES)}")
        st.code(', '.join(ANPR_CLASSES))
    
    with class_col2:
        st.markdown("##### ATCC Model (`yolo_ATCC.pt`)")
        st.markdown(f"**Classes:** {len(ATCC_CLASSES)}")
        st.code(', '.join(ATCC_CLASSES))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit and YOLOv8"
    "</div>",
    unsafe_allow_html=True
)