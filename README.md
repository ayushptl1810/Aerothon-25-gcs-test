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

- **Smart Architecture**:
  - Background analysis thread for non-blocking video feed
  - Configurable analysis interval
  - Live connection status monitoring

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

### Basic Usage (Webcam)

1. Open the Streamlit app
2. Select **"Webcam"** as Input Type
3. Enter device index (usually `0` for default webcam)
4. Click **"Start Stream"** toggle
5. View annotated frames with detected objects on the left
6. Labels appear on the right side

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

- **Annotated Frame**: Large view (left side) showing detected objects with bounding boxes and labels
- **Raw Feed**: Smaller view (right side) showing original video stream
- **Labels List**: Detected object labels displayed as bulleted list

### Detection Settings

- Analysis runs in background thread (default: every 0.3 seconds)
- Configurable via `interval_sec` in `AnalysisState` class
- Video rendering at full frame rate regardless of backend processing

### Debug Mode

- Enable "Show debug info" to see:
  - API latency in milliseconds
  - Number of detected labels
  - Number of bounding boxes
  - Number of points

## Architecture

### Backend (`backend/main.py`)

- FastAPI server handling frame analysis requests
- Gemini Robotics API integration
- Returns labels, bounding boxes, and point coordinates
- Normalized coordinates (0-1000 range)

### Frontend (`frontend/app.py`)

- Streamlit UI with video input selection
- Background worker thread for non-blocking analysis
- Real-time video rendering
- Bounding box visualization with adaptive font sizing

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

## Notes

- All configuration is done through UI or environment variables (no hardcoding)
- Supports Windows, macOS, and Linux
- Network streams require proper network configuration
- Gemini API costs apply per request
- Background thread ensures smooth video playback
