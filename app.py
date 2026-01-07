import streamlit as st
import supervision as sv
from ultralytics import YOLOE
import cv2
import numpy as np
import tempfile
import os
import urllib.request
from pathlib import Path
import re
import gdown

# Set Streamlit page config
st.set_page_config(page_title="PPE Detection System", layout="wide")
st.title("PPE DETECTION SYSTEM 🦺🪖🧤🥽🥾")

# Model download and setup code
@st.cache_resource
def download_and_setup_model():
    # Check if the final model file already exists
    if os.path.exists("yoloe-11l-seg-ppe.pt"):
        return
    
    url = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
    urllib.request.urlretrieve(url, "mobileclip_blt.pt")
    from ultralytics import YOLOE
    # Initialize a YOLOE model
    model = YOLOE("yoloe-11l-seg.pt")  
    names = ["person","safety vest","helmet","gloves","goggles","shoes"]
    model.set_classes(names, model.get_text_pe(names))
    # save the model
    model.save("yoloe-11l-seg-ppe.pt")

class PPEDetectionSystem:
    def __init__(self, model_path: str = "yoloe-11l-seg-ppe.pt"):
        self.model = YOLOE(model_path)
        self.ppe_items = ["safety vest", "helmet", "gloves", "goggles", "shoes"]
        self.ppe_icons = {
            "helmet": "🪖",
            "gloves": "🧤", 
            "safety vest": "🦺",
            "goggles": "🥽",
            "shoes": "🥾"
        }
        
    def calculate_overlap_percentage(self, person_box: np.ndarray, ppe_box: np.ndarray) -> float:
        """Calculate what percentage of PPE box is inside person box"""
        px1, py1, px2, py2 = person_box
        ox1, oy1, ox2, oy2 = ppe_box
        
        # Calculate intersection
        ix1 = max(px1, ox1)
        iy1 = max(py1, oy1)
        ix2 = min(px2, ox2)
        iy2 = min(py2, oy2)
        
        # No intersection
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
            
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        ppe_area = (ox2 - ox1) * (oy2 - oy1)
        
        return intersection_area / ppe_area if ppe_area > 0 else 0.0
    
    def process_frame(self, frame: np.ndarray):
        # Run inference
        results = self.model.predict(frame, conf=0.1, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        if len(detections) == 0:
            return frame, []
            
        # Get class names
        class_names = [self.model.names[class_id] for class_id in detections.class_id]
        
        # Find persons and sort by x-coordinate (left to right)
        persons = []
        for i, class_name in enumerate(class_names):
            if class_name == "person":
                bbox = detections.xyxy[i]
                center_x = (bbox[0] + bbox[2]) / 2
                persons.append((center_x, i, bbox))
        
        # Sort persons by x-coordinate
        persons.sort(key=lambda x: x[0])
        
        # Use supervision library for annotations
        annotated_frame = frame.copy()
        annotated_frame = sv.ColorAnnotator().annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.BoxAnnotator().annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.LabelAnnotator().annotate(scene=annotated_frame, detections=detections)
        
        # Add person IDs (keeping your original logic)
        for person_id, (_, detection_idx, person_bbox) in enumerate(persons, 1):
            # Draw person ID with smaller font in red
            cv2.putText(annotated_frame, f"Person {person_id}", 
                       (int(person_bbox[0]), int(person_bbox[1]) - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Process each person for PPE analysis (keeping your original logic)
        analysis_results = []
        for person_id, (_, detection_idx, person_bbox) in enumerate(persons, 1):
            # Check each PPE item
            ppe_status = {}
            for ppe_item in self.ppe_items:
                found = False
                for i, class_name in enumerate(class_names):
                    if class_name == ppe_item:
                        ppe_bbox = detections.xyxy[i]
                        overlap = self.calculate_overlap_percentage(person_bbox, ppe_bbox)
                        if overlap >= 0.75:
                            found = True
                            break
                ppe_status[ppe_item] = found
            
            analysis_results.append({
                'person_id': person_id,
                'ppe_status': ppe_status
            })
        
        return annotated_frame, analysis_results

def setup_sample_videos():
    """Download and setup sample videos from Google Drive"""
    video_dir = Path("sample_videos")
    gdrive_folder_id = "1qY_3KLuOiuwV-73tsSQIlL7V1IjcC_e6"
    
    if not video_dir.exists():
        st.info("📥 Downloading sample videos from Google Drive...")
        try:
            gdown.download_folder(id=gdrive_folder_id, output=str(video_dir), quiet=False, use_cookies=False)
            st.success("✅ Sample videos downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download sample videos: {e}")
            return None
    
    # Collect video files from the directory
    video_files = sorted(video_dir.glob("sample_video_ppe_*.mp4"))
    
    # Prepare base video names (without _annotated)
    video_options = sorted(set(
        re.sub(r"_annotated", "", f.stem) for f in video_files if "_annotated" not in f.stem
    ))
    
    return video_dir, video_options

def main():
    # Setup sample videos
    sample_setup = setup_sample_videos()
    
    # Video source selection
    st.subheader("📹 Video Source Selection")
    video_source = st.radio(
        "Choose video source:",
        ["Sample Videos", "Upload Video"],
        horizontal=True
    )
    
    video_path = None
    show_annotated = False
    annotated_video_path = None
    
    if video_source == "Sample Videos" and sample_setup:
        video_dir, video_options = sample_setup
        
        if video_options:
            selected_video = st.selectbox(
                "Select a demo video:",
                video_options,
                format_func=lambda x: f"🎥 {x.replace('_', ' ').title()}"
            )
            
            # Construct file paths
            original_path = video_dir / f"{selected_video}.mp4"
            annotated_path = video_dir / f"{selected_video}_annotated.mp4"
            
            if original_path.exists():
                video_path = str(original_path.resolve())
                show_annotated = True
                if annotated_path.exists():
                    annotated_video_path = str(annotated_path.resolve())
            else:
                st.error("Selected video not found.")
        else:
            st.warning("⚠️ No sample videos found.")
    
    elif video_source == "Upload Video":
        uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
    
    # Display videos if available
    if video_path:
        if show_annotated and annotated_video_path:
            # Side-by-side video layout for sample videos
            st.subheader("📺 Video Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📹 Original Video**")
                st.video(video_path)
            
            with col2:
                st.markdown("**🎯 Detection Video**")
                st.video(annotated_video_path)
        else:
            # Single video display for uploaded videos
            st.subheader("📹 Video Preview")
            st.video(video_path)
        
        # Process Video Button and Real-time Analysis
        st.subheader("🔍 Real-time PPE Analysis")
        
        if st.button("🚀 Process Video", type="primary"):
            # Download and setup model
            download_and_setup_model()
            
            # Initialize model
            if 'model' not in st.session_state:
                st.session_state.model = PPEDetectionSystem()
            
            try:
                # Get video info
                video_info = sv.VideoInfo.from_video_path(video_path)
                fps = video_info.fps
                total_frames = video_info.total_frames
                frame_interval = int(fps // 2)  # 2 frames per second
                
                st.info(f"📊 Processing video: {fps} FPS, {total_frames} total frames")
                
                # Create containers for real-time display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    video_container = st.empty()
                
                with col2:
                    analysis_container = st.empty()
                
                # Process video frames
                frame_generator = sv.get_video_frames_generator(video_path)
                
                progress_bar = st.progress(0)
                
                for frame_index, frame in enumerate(frame_generator):
                    if frame_index % frame_interval == 0:
                        # Process frame
                        annotated_frame, analysis_results = st.session_state.model.process_frame(frame)
                        
                        # Display annotated frame
                        video_container.image(annotated_frame, channels="BGR")
                        
                        # Display analysis results with better spacing (same format as app1.py)
                        if analysis_results:
                            analysis_text = ""
                            for result in analysis_results:
                                person_id = result['person_id']
                                ppe_status = result['ppe_status']
                                
                                analysis_text += f"**Person {person_id}:** "
                                ppe_items = []
                                for ppe_item, detected in ppe_status.items():
                                    icon = st.session_state.model.ppe_icons[ppe_item]
                                    status_icon = "**✓**" if detected else "**✘**"
                                    ppe_items.append(f"{icon} {ppe_item}: {status_icon}")
                                
                                # Join PPE items with more spacing
                                analysis_text += "  |  ".join(ppe_items)
                                analysis_text += "\n\n"
                            
                            analysis_container.markdown(analysis_text)
                        else:
                            analysis_container.markdown("❌ No persons detected in current frame")
                    
                    # Update progress
                    progress = (frame_index + 1) / total_frames
                    progress_bar.progress(progress)
                
                st.success("✅ Video processing completed!")
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
            
            finally:
                # Clean up uploaded file
                if video_source == "Upload Video" and os.path.exists(video_path):
                    os.unlink(video_path)
    
    # Demo warning notice
    st.markdown("""
    ---  
    ⚠️ **Demo Version Notice**

    This is a simplified demonstration model with basic PPE detection capabilities.
    """)

if __name__ == "__main__":
    main()