import base64
import io
import os
import time
from typing import List, Optional
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(page_title="Gemini - Labels", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def encode_frame_to_jpeg(frame: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return bytes(encoded)


def get_video_capture(input_type: str, connection_string: str):
    """
    Create a VideoCapture object based on input type.
    
    Args:
        input_type: One of 'Webcam', 'UDP', 'TCP', 'RTSP', 'HTTP', 'File'
        connection_string: Connection details (device index, URL, or file path)
    
    Returns:
        cv2.VideoCapture object or None if failed
    """
    try:
        if input_type == "Webcam":
            # Parse device index (default 0)
            device_index = int(connection_string) if connection_string.strip() else 0
            return cv2.VideoCapture(device_index)
        
        elif input_type == "UDP":
            # UDP stream format: udp://ip:port or udp://@:port (for listening)
            if not connection_string.startswith("udp://"):
                connection_string = f"udp://{connection_string}"
            cap = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)
            return cap
        
        elif input_type == "TCP":
            # TCP stream format: tcp://ip:port
            if not connection_string.startswith("tcp://"):
                connection_string = f"tcp://{connection_string}"
            cap = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)
            return cap
        
        elif input_type == "RTSP":
            # RTSP stream format: rtsp://username:password@ip:port/path or rtsp://ip:port/path
            if not connection_string.startswith("rtsp://"):
                connection_string = f"rtsp://{connection_string}"
            cap = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)
            return cap
        
        elif input_type == "HTTP/MJPEG":
            # HTTP MJPEG stream format: http://ip:port/stream
            if not connection_string.startswith("http://") and not connection_string.startswith("https://"):
                connection_string = f"http://{connection_string}"
            cap = cv2.VideoCapture(connection_string)
            return cap
        
        elif input_type == "Video File":
            # Local video file path
            return cv2.VideoCapture(connection_string)
        
        else:
            return None
            
    except Exception as e:
        st.error(f"Failed to create video capture: {e}")
        return None


def send_frame(frame_bytes: bytes, session_id: Optional[str] = None, frame_number: int = 0, is_first_frame: bool = False):
    """
    Send frame to backend for analysis.
    If session_id is provided, uses contextual endpoint with conversation history.
    """
    try:
        if session_id:
            # Use contextual endpoint with session tracking
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            data_payload = {
                "session_id": session_id,
                "frame_number": frame_number,
                "is_first_frame": is_first_frame
            }
            resp = requests.post(
                f"{BACKEND_URL}/analyze_frame_contextual",
                files=files,
                data=data_payload,
                timeout=30
            )
        else:
            # Legacy endpoint without context
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            resp = requests.post(f"{BACKEND_URL}/analyze_frame", files=files, timeout=20)
        
        if resp.status_code != 200:
            return {"labels": [], "points": [], "bboxes": [], "objects": []}
        
        data = resp.json()
        return {
            "labels": data.get("labels", []) or [],
            "points": data.get("points", []) or [],
            "bboxes": data.get("bboxes", []) or [],
            "objects": data.get("objects", []) or [],
            "unique_objects_count": data.get("unique_objects_count", 0),
        }
    except Exception as e:
        return {"labels": [], "points": [], "bboxes": [], "objects": []}


@dataclass
class AnalysisState:
    latest_jpg: Optional[bytes] = None
    last_result: dict = field(default_factory=lambda: {"labels": [], "points": [], "bboxes": []})
    last_latency_ms: Optional[float] = None
    running: bool = False
    thread: Optional[threading.Thread] = None
    interval_sec: float = 0.3
    recording: bool = False
    recorded_objects: dict = field(default_factory=dict)  # {label: count} - OLD method
    session_id: Optional[str] = None  # Session ID for contextual tracking
    frame_number: int = 0  # Current frame number in recording session
    unique_objects: dict = field(default_factory=dict)  # {object_id: {label, first_seen, last_seen}}


def _ensure_analysis_state() -> AnalysisState:
    if "analysis_state" not in st.session_state:
        st.session_state["analysis_state"] = AnalysisState()
    return st.session_state["analysis_state"]


def start_analysis_worker():
    state = _ensure_analysis_state()
    if state.running:
        return

    state.running = True

    def _worker():
        last_ts = 0.0
        while state.running:
            try:
                now = time.time()
                # Only call API when recording is active
                if state.recording and state.latest_jpg is not None and (now - last_ts) >= max(0.05, state.interval_sec):
                    t0 = time.time()
                    
                    # Determine if this is the first frame of the session
                    is_first = (state.frame_number == 0)
                    
                    # Call API with session context
                    result = send_frame(
                        state.latest_jpg,
                        session_id=state.session_id,
                        frame_number=state.frame_number,
                        is_first_frame=is_first
                    )
                    state.last_latency_ms = (time.time() - t0) * 1000.0
                    state.last_result = result
                    state.frame_number += 1
                    last_ts = now
                    
                    # Update unique objects from contextual response
                    if state.recording and result and isinstance(result, dict):
                        objects = result.get("objects", [])
                        for obj in objects:
                            if isinstance(obj, dict):
                                obj_id = obj.get("object_id")
                                label = obj.get("label")
                                is_new = obj.get("is_new", False)
                                
                                if obj_id and label:
                                    if obj_id not in state.unique_objects:
                                        state.unique_objects[obj_id] = {
                                            "label": label,
                                            "first_seen": state.frame_number,
                                            "last_seen": state.frame_number,
                                            "is_new": is_new
                                        }
                                    else:
                                        state.unique_objects[obj_id]["last_seen"] = state.frame_number
                        
                        # Also maintain old method for backward compatibility
                        labels = result.get("labels", [])
                        for label in labels:
                            if label and isinstance(label, str):
                                label_lower = label.lower().strip()
                                state.recorded_objects[label_lower] = state.recorded_objects.get(label_lower, 0) + 1
            except Exception:
                pass
            time.sleep(0.005)

    t = threading.Thread(target=_worker, daemon=True)
    state.thread = t
    t.start()


def stop_analysis_worker():
    state = _ensure_analysis_state()
    state.running = False


st.title("Gemini Object Labels")
st.caption("Legacy mode with smooth rendering: camera renders continuously; analysis runs in background.")

# Video input configuration (also in sidebar for convenience)
st.subheader("📹 Video Input Configuration")
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    input_type = st.selectbox(
        "Input Type",
        ["Webcam", "UDP", "TCP", "RTSP", "HTTP/MJPEG", "Video File"],
        index=0,
        help="Select the type of video input source"
    )

# Also add to sidebar for convenience
st.sidebar.header("Video Input Configuration")
st.sidebar.write(f"**Selected:** {input_type}")

# Dynamic connection string based on input type
with col_input2:
    if input_type == "Webcam":
        connection_string = st.text_input(
            "Device Index",
            value="0",
            help="Camera device index (usually 0 for default webcam)"
        )
        st.caption("Example: 0, 1, 2")
        
    elif input_type == "UDP":
        connection_string = st.text_input(
            "UDP Address",
            value="udp://@:5000",
            help="UDP stream address (server mode: udp://@:port, client: udp://ip:port)"
        )
        st.caption("Examples: udp://@:5000 (listen) or udp://192.168.1.100:5000")
        
    elif input_type == "TCP":
        connection_string = st.text_input(
            "TCP Address",
            value="tcp://192.168.1.100:5000",
            help="TCP stream address"
        )
        st.caption("Example: tcp://192.168.1.100:5000")
        
    elif input_type == "RTSP":
        connection_string = st.text_input(
            "RTSP URL",
            value="rtsp://192.168.1.100:554/stream",
            help="RTSP stream URL (supports authentication)"
        )
        st.caption("Example: rtsp://192.168.1.100:554/stream or rtsp://user:pass@ip:port/stream")
        
    elif input_type == "HTTP/MJPEG":
        connection_string = st.text_input(
            "HTTP URL",
            value="http://192.168.1.100:8080/video",
            help="HTTP MJPEG stream URL"
        )
        st.caption("Example: http://192.168.1.100:8080/video")
        
    elif input_type == "Video File":
        connection_string = st.text_input(
            "File Path",
            value="",
            help="Path to video file"
        )
        st.caption("Example: /path/to/video.mp4")

st.divider()

# Control buttons and status
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    debug_enabled = st.checkbox("Show debug info", value=False)
with col_ctrl2:
    status_placeholder = st.empty()
with col_ctrl3:
    record_button_placeholder = st.empty()

# Additional info
with st.expander("ℹ️ Connection Help & Examples"):
    st.markdown("""
    **Webcam**: Use device index (0, 1, 2...)
    
    **UDP**: Stream video over UDP protocol
    - Server mode: `udp://@:port` (listen)
    - Client mode: `udp://ip:port`
    
    **TCP**: Stream video over TCP protocol
    - Format: `tcp://ip:port`
    
    **RTSP**: IP camera or RTSP stream
    - Basic: `rtsp://ip:port/path`
    - With auth: `rtsp://user:pass@ip:port/path`
    
    **HTTP/MJPEG**: Motion JPEG over HTTP
    - Format: `http://ip:port/stream`
    
    **Video File**: Local video file playback
    - Provide full path to file
    
    **Note**: Network streams require proper firewall configuration and the stream source must be actively transmitting.
    """)

debug_placeholder = st.empty()

# Also keep sidebar info
st.sidebar.divider()
st.sidebar.toggle("Show debug (sidebar)", value=False, key="_sidebar_debug", disabled=True, help="Use main page toggle")
sidebar_status = st.sidebar.empty()

col_annotated, col_side = st.columns([3, 2])
with col_annotated:
    st.subheader("Annotated")
    annotated_placeholder = st.empty()
with col_side:
    st.subheader("Raw Feed")
    cam_placeholder = st.empty()
    st.subheader("Labels")
    labels_placeholder = st.empty()

# Recording summary section
st.divider()
summary_container = st.container()
with summary_container:
    summary_placeholder = st.empty()

# Initialize recording state outside the loop
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False

# Recording button with callback
def toggle_recording():
    import uuid
    
    st.session_state["is_recording"] = not st.session_state["is_recording"]
    state = _ensure_analysis_state()
    state.recording = st.session_state["is_recording"]
    
    if st.session_state["is_recording"]:
        # Starting new recording - generate session ID
        state.session_id = str(uuid.uuid4())
        state.frame_number = 0
        state.recorded_objects = {}
        state.unique_objects = {}
        logger_msg = f"Started recording session: {state.session_id}"
        print(logger_msg)
    else:
        # Stopped recording - cleanup session on backend
        if state.session_id:
            try:
                requests.delete(f"{BACKEND_URL}/session/{state.session_id}", timeout=5)
            except Exception:
                pass
        state.session_id = None
        state.frame_number = 0

# Show recording button
if st.session_state.get("is_recording", False):
    record_button_placeholder.button("⏹️ Stop Recording", on_click=toggle_recording, type="primary", use_container_width=True)
else:
    record_button_placeholder.button("🔴 Start Recording", on_click=toggle_recording, use_container_width=True)

# Display recording summary if we just stopped recording and have data
if not st.session_state.get("is_recording", False):
    state = _ensure_analysis_state()
    if state.unique_objects:
        # Show unique objects tracked via context
        summary_text = "### 📊 Recording Summary (Context-Aware Tracking)\n\n**Unique Objects Detected:**\n\n"
        
        # Group by label
        label_groups = {}
        for obj_id, obj_data in state.unique_objects.items():
            label = obj_data["label"]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(obj_data)
        
        for label, objs in sorted(label_groups.items()):
            count = len(objs)
            if count == 1:
                summary_text += f"- **{label.title()}**: 1 unique instance\n"
            else:
                summary_text += f"- **{label.title()}**: {count} unique instances\n"
        
        total_unique = len(state.unique_objects)
        total_appearances = sum(state.recorded_objects.values()) if state.recorded_objects else 0
        
        summary_text += f"\n**Summary:**\n"
        summary_text += f"- Total unique objects: **{total_unique}**\n"
        if total_appearances > 0:
            summary_text += f"- Total frame appearances: {total_appearances}\n"
            summary_text += f"- Deduplication saved: **{total_appearances - total_unique}** duplicate counts!\n"
        
        summary_placeholder.success(summary_text)
    elif state.recorded_objects:
        # Fallback to old method if contextual tracking didn't work
        summary_text = "### 📊 Recording Summary (Legacy Mode)\n\n**Objects Detected:**\n\n"
        sorted_objects = sorted(state.recorded_objects.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_objects:
            summary_text += f"- **{label.title()}**: {count} appearances\n"
        summary_text += f"\n**Note**: This shows frame appearances, not unique objects.\n"
        summary_text += f"**Total detections**: {sum(state.recorded_objects.values())}"
        summary_placeholder.warning(summary_text)
    else:
        summary_placeholder.empty()

run = st.toggle("Start Stream", value=False, key="_toggle_start_webcam")

if run:
    start_analysis_worker()
    
    # Show connection attempt status
    status_placeholder.info(f"🔄 Connecting to {input_type}...")
    sidebar_status.info(f"🔄 Connecting...")
    
    # Create video capture based on selected input type
    cap = get_video_capture(input_type, connection_string)
    
    if cap is None or not cap.isOpened():
        st.error(f"Unable to open {input_type} stream. Please check your connection settings.")
        status_placeholder.error(f"❌ Connection failed")
        sidebar_status.error(f"❌ Failed")
        stop_analysis_worker()
    else:
        status_placeholder.success(f"✅ Connected to {input_type}")
        sidebar_status.success(f"✅ Connected")
        try:
            # Cache last results to only update UI when new details arrive
            last_labels: List[str] = []
            last_bboxes = []
            state = _ensure_analysis_state()
            
            # Initialize labels display
            labels_placeholder.write("Click 'Start Recording' to begin detection...")
            
            # Sync session state with analysis state
            state.recording = st.session_state.get("is_recording", False)
            
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("Failed to read frame")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(rgb, channels="RGB")

                # Hand off JPEG to background worker; it will update state.last_result
                try:
                    state.latest_jpg = encode_frame_to_jpeg(frame)
                except Exception:
                    pass

                result = state.last_result if isinstance(state.last_result, dict) else {}
                labels: List[str] = result.get("labels", [])
                points = result.get("points", [])
                bboxes = result.get("bboxes", [])

                if debug_enabled:
                    debug_placeholder.code({
                        "latency_ms": round(state.last_latency_ms, 1) if state.last_latency_ms else None,
                        "labels_count": len(labels),
                        "points_count": len(points) if isinstance(points, list) else 0,
                        "bboxes_count": len(bboxes) if isinstance(bboxes, list) else 0,
                    })

                # Update labels only if they changed
                details_changed = (
                    labels != last_labels or bboxes != last_bboxes
                )
                if details_changed:
                    last_labels = labels
                    last_bboxes = bboxes
                    # Update label display
                    if state.recording:
                        if labels:
                            labels_text = "**🔴 Recording**\n\n"
                            labels_text += "\n".join([f"• {label}" for label in labels])
                            labels_placeholder.markdown(labels_text)
                        else:
                            labels_placeholder.markdown("**🔴 Recording**\n\nNo labels detected")
                    else:
                        labels_placeholder.write("Click 'Start Recording' to begin detection...")
                    
                    # Only update annotated frame when detection results change (not every frame)
                    # This makes it a "snapshot" of the last analyzed frame
                    if state.recording and (labels or bboxes):
                        try:
                            annotated = rgb.copy()
                            h, w, _ = annotated.shape
                            
                            def denorm_x(x):
                                return int(max(0, min(w - 1, round((x / 1000.0) * w))))
                            def denorm_y(y):
                                return int(max(0, min(h - 1, round((y / 1000.0) * h))))

                            for idx, bbox in enumerate(bboxes):
                                if isinstance(bbox, list) and len(bbox) == 4:
                                    y1, x1, y2, x2 = bbox
                                    p1 = (denorm_x(x1), denorm_y(y1))
                                    p2 = (denorm_x(x2), denorm_y(y2))
                                    cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                                    if idx < len(labels):
                                        text = labels[idx]
                                        # Font scale relative to image height for readability; thicker stroke
                                        font_scale = max(0.7, min(2.0, h / 480.0))
                                        cv2.putText(
                                            annotated,
                                            text,
                                            (p1[0], max(0, p1[1] - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale,
                                            (0, 255, 0),
                                            2,
                                        )
                            # Points intentionally not rendered per requirement

                            # Add timestamp/frame number overlay
                            timestamp_text = f"Frame: {state.frame_number} | Objects: {len(labels)}"
                            cv2.putText(
                                annotated,
                                timestamp_text,
                                (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            
                            annotated_placeholder.image(annotated, channels="RGB", use_column_width=True)
                        except Exception as e:
                            # Show error if rendering fails
                            annotated_placeholder.error(f"Rendering error: {e}")

                run = st.session_state.get("_toggle_start_webcam", run)
                time.sleep(0.01)
        finally:
            cap.release()
            stop_analysis_worker()
            # Reset recording state
            if "is_recording" in st.session_state:
                st.session_state["is_recording"] = False
            state.recording = False
            status_placeholder.warning("⏸️ Disconnected")
            sidebar_status.warning("⏸️ Disconnected")
else:
    st.caption("Idle - Select input type and click 'Start Stream'")
    status_placeholder.info("⏸️ Ready to connect")
    sidebar_status.info("⏸️ Ready")
    summary_placeholder.empty()


