### Gemini Object Detection GCS (Streamlit + FastAPI)

Real-time object detection system using Google Gemini Robotics API with support for multiple video input sources including webcam, UDP, TCP, RTSP streams, and video files.

## Features

- **Multiple Video Input Sources**:

  - Webcam (USB/Built-in cameras)
  - UDP streams (server/client modes)
  - TCP streams
  - RTSP streams (IP cameras with authentication)
  - HTTP/MJPEG streams
  - Local video files

- **Real-time Object Detection**:

  - Powered by Google Gemini Robotics API
  - Bounding box visualization with labels
  - Adjustable label font size
  - Smooth video rendering without lag
  - On-demand recording to control API costs

- **Context-Aware Object Tracking**:

  - Maintains conversation history across frames
  - Prevents duplicate object counting
  - Tracks unique objects throughout recording session
  - AI-powered object matching by position and appearance
  - Accurate statistics: unique objects vs frame appearances

- **Smart Architecture**:
  - Background analysis thread for non-blocking video feed
  - Session-based conversation context with Gemini
  - Configurable analysis interval
  - Live connection status monitoring
  - Automatic session cleanup

## Prerequisites

- Python 3.9+
- A Google AI Studio API key with access to Gemini Robotics models
- OpenCV with FFmpeg support (for network streams)

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file at project root with:

```bash
GEMINI_API_KEY=your_api_key_here
ROBOTICS_MODEL=gemini-robotics-er-1.5-preview
BACKEND_URL=http://127.0.0.1:8000
```

3. (Optional) Configure logging level:

```bash
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
GEMINI_DEBUG=false
```

## Running the Application

### Start Backend

```bash
uvicorn backend.main:app --reload
```

The backend will start on `http://127.0.0.1:8000`

### Start Frontend

```bash
streamlit run frontend/app.py
```

The frontend will open in your browser at `http://localhost:8501`

## Usage

### Basic Usage (Webcam with Recording)

1. Open the Streamlit app
2. Select **"Webcam"** as Input Type
3. Enter device index (usually `0` for default webcam)
4. Click **"Start Stream"** toggle
5. Click **"🔴 Start Recording"** button to begin object detection
6. View results:
   - **Left**: Annotated snapshot of last analyzed frame with bounding boxes
   - **Right Top**: Live webcam feed
   - **Right Bottom**: Detected labels list
7. Click **"⏹️ Stop Recording"** to end session
8. View recording summary with unique object counts

### Network Streams

#### UDP Stream (Listening Mode)

```
Input Type: UDP
UDP Address: udp://@:5000
```

#### UDP Stream (Client Mode)

```
Input Type: UDP
UDP Address: udp://192.168.1.100:5000
```

#### TCP Stream

```
Input Type: TCP
TCP Address: tcp://192.168.1.100:5000
```

#### RTSP Camera (No Auth)

```
Input Type: RTSP
RTSP URL: rtsp://192.168.1.100:554/stream
```

#### RTSP Camera (With Authentication)

```
Input Type: RTSP
RTSP URL: rtsp://username:password@192.168.1.100:554/stream1
```

#### HTTP MJPEG Stream

```
Input Type: HTTP/MJPEG
HTTP URL: http://192.168.1.100:8080/video
```

#### Video File

```
Input Type: Video File
File Path: /path/to/your/video.mp4
```

## Features Details

### Video Display

- **Annotated Frame**: Large view (left side) showing snapshot of last analyzed frame with bounding boxes and labels
  - Only updates when new detection completes
  - Shows frame number and object count overlay
  - Static display between detections
- **Raw Feed**: Smaller view (right side) showing continuous live video stream
- **Labels List**: Detected object labels displayed as bulleted list with recording status

### Recording & Object Tracking

- **On-Demand Detection**: Click "Start Recording" to begin API calls (save costs!)
- **Context-Aware Tracking**: Gemini maintains conversation history across frames
- **Unique Object Counting**: AI tracks same object across multiple frames
- **Session Management**: Each recording gets unique UUID for context isolation
- **Smart Deduplication**: Summary shows unique objects vs total appearances

### Recording Summary

After stopping recording, view detailed statistics:

- **Unique Objects**: Count of distinct objects detected (e.g., "Person: 1 unique instance")
- **Instance Tracking**: Multiple instances of same object type counted separately
- **Deduplication Stats**: Shows how many duplicate counts were prevented
- **Comparison**: Total unique objects vs total frame appearances

Example Output:

```
📊 Recording Summary (Context-Aware Tracking)

Unique Objects Detected:
- Person: 1 unique instance
- Laptop: 1 unique instance
- Cup: 2 unique instances

Summary:
- Total unique objects: 4
- Total frame appearances: 87
- Deduplication saved: 83 duplicate counts!
```

### Detection Settings

- Analysis runs in background thread (default: every 0.3 seconds)
- Configurable via `interval_sec` in `AnalysisState` class
- Video rendering at full frame rate regardless of backend processing
- API only called during active recording (cost-effective)

### Debug Mode

- Enable "Show debug info" to see:
  - API latency in milliseconds
  - Number of detected labels
  - Number of bounding boxes
  - Frame processing details

## Architecture

### Backend (`backend/main.py`)

- FastAPI server with two endpoints:
  - `/analyze_frame`: Legacy stateless detection
  - `/analyze_frame_contextual`: Session-based contextual tracking
- Session management with in-memory storage
- Conversation history maintenance (last 5 exchanges)
- Gemini Robotics API integration with context
- Returns labels, bounding boxes, objects with tracking IDs
- Normalized coordinates (0-1000 range)
- Automatic session cleanup after 1 hour

### Frontend (`frontend/app.py`)

- Streamlit UI with video input selection
- Background worker thread for non-blocking analysis
- Real-time video rendering (live feed)
- Snapshot-based annotated frame display
- Session ID generation (UUID) for recording sessions
- Bounding box visualization with adaptive font sizing
- Context-aware object tracking and deduplication
- Comprehensive recording summary statistics

## Configuration

### Environment Variables

| Variable         | Description              | Default                          |
| ---------------- | ------------------------ | -------------------------------- |
| `GEMINI_API_KEY` | Google AI Studio API key | Required                         |
| `ROBOTICS_MODEL` | Gemini model name        | `gemini-robotics-er-1.5-preview` |
| `BACKEND_URL`    | Backend API URL          | `http://127.0.0.1:8000`          |
| `LOG_LEVEL`      | Logging level            | `INFO`                           |
| `GEMINI_DEBUG`   | Enable debug logging     | `false`                          |

## Troubleshooting

### Network Streams Not Connecting

- Ensure firewall allows incoming/outgoing connections on specified ports
- Verify the stream source is actively transmitting
- Check that OpenCV is built with FFmpeg support: `cv2.getBuildInformation()`
- For UDP server mode, use `udp://@:port` format
- For RTSP, verify credentials and stream path

### Video Lag

- Adjust `interval_sec` in `AnalysisState` (line 52 in `frontend/app.py`)
- Lower values = more frequent API calls (higher cost, more detections)
- Higher values = less frequent updates (lower cost, smoother video)

### No Labels Detected

- Check backend logs for API errors
- Verify `GEMINI_API_KEY` is correct
- Ensure sufficient lighting and object visibility
- Backend returns up to 10 objects per frame
- Make sure recording is active (click Start Recording button)
- Check that backend is running on correct port

### Recording Summary Shows Legacy Mode

- Context-aware tracking requires active recording session
- Ensure backend `/analyze_frame_contextual` endpoint is working
- Check backend logs for session creation messages
- Verify session_id is being passed in requests

### Duplicate Object Counting

- If seeing duplicate counts, verify context-aware endpoint is being used
- Check that session_id is maintained throughout recording
- Backend may fall back to legacy mode if context fails
- Review backend logs for conversation history errors

## Notes

- All configuration is done through UI or environment variables (no hardcoding)
- Supports Windows, macOS, and Linux
- Network streams require proper network configuration
- **Gemini API costs apply per request** - Use recording feature to control when API is called
- Background thread ensures smooth video playback
- Context-aware tracking maintains conversation history (limited to last 5 exchanges)
- Sessions auto-expire after 1 hour of inactivity
- Annotated frame shows snapshot of last analyzed frame, not continuous feed
- Recording button must be active for detection to run (cost-saving feature)

## How Context-Aware Tracking Works

1. **Start Recording**: Generates unique session ID (UUID)
2. **First Frame**: Gemini detects objects and assigns tracking IDs
3. **Subsequent Frames**: Gemini receives conversation history
   - Sees previous frames and objects
   - Matches objects by position and appearance
   - Assigns same ID to persistent objects
   - Creates new ID for new objects
4. **Stop Recording**: Backend session cleaned up
5. **Summary**: Shows unique objects (not frame appearances)

This approach prevents counting the same person/object multiple times across frames!
