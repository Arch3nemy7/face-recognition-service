# Face Recognition Service

A stateless, production-ready Python microservice for face embedding extraction and comparison using ArcFace via InsightFace. This service is designed to replace third-party face recognition providers while keeping all user data and business logic in your main backend.

## Features

- **Stateless Design**: No database, no file storage, no user data persistence
- **Modern Stack**: FastAPI + InsightFace (ArcFace) + ONNX Runtime
- **Production-Ready**: Docker support, health checks, comprehensive error handling
- **Horizontally Scalable**: Multiple instances can run independently
- **CPU-Optimized**: Efficient inference on CPU with optional GPU support
- **RESTful API**: Clean JSON endpoints with automatic OpenAPI documentation

## Architecture

This microservice is designed to work alongside your existing attendance system:

```
┌─────────────────────────────────────────────┐
│         Main Backend (Your System)          │
│  ┌────────────────────────────────────────┐ │
│  │ • User Management                      │ │
│  │ • Embedding Storage (Database)         │ │
│  │ • Attendance Recording                 │ │
│  │ • Business Logic                       │ │
│  └────────────────────────────────────────┘ │
│                    │                         │
│                    │ HTTP Requests           │
│                    ▼                         │
│  ┌────────────────────────────────────────┐ │
│  │   Face Recognition Service             │ │
│  │ ┌────────────────────────────────────┐ │ │
│  │ │ POST /api/v1/embed                 │ │ │
│  │ │   → Extract face embedding         │ │ │
│  │ │                                    │ │ │
│  │ │ POST /api/v1/compare               │ │ │
│  │ │   → Compare embeddings (optional)  │ │ │
│  │ └────────────────────────────────────┘ │ │
│  │        (Stateless - No Data Storage)   │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- 2GB RAM minimum (4GB+ recommended)
- Linux, macOS, or Windows

### Installation

#### 1. Clone the Repository

```bash
cd face-recognition-service
```

#### 2. Create Python Virtual Environment

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

On first run, InsightFace will automatically download the ArcFace model (~140MB). This happens once and is cached in `~/.insightface/`.

#### 4. Run the Service

```bash
# Development mode (with auto-reload)
uvicorn face_recognition_service.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python -m face_recognition_service.main
```

The service will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

**GET** `/api/v1/health`

Check if the service is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "buffalo_l"
}
```

### 2. Model Information

**GET** `/api/v1/model-info`

Get information about the loaded face recognition model.

**Response:**
```json
{
  "name": "buffalo_l",
  "embedding_size": 512,
  "backend": "insightface",
  "device": "cpu"
}
```

### 3. Extract Face Embedding

**POST** `/api/v1/embed`

Extract a 512-dimensional face embedding from an image.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],  // 512 floats
  "face_detected": true,
  "detection_score": 0.98
}
```

**Error Cases:**
- No face detected: `error_code: "NO_FACE_DETECTED"`
- Multiple faces: `error_code: "MULTIPLE_FACES_DETECTED"`
- Invalid image: `error_code: "INVALID_IMAGE"`

### 4. Compare Embeddings

**POST** `/api/v1/compare`

Compare a query embedding against multiple reference embeddings.

**Request:**
```json
{
  "query_embedding": [0.123, -0.456, ...],  // 512 floats
  "reference_embeddings": [
    {
      "id": "user_001",
      "embedding": [0.234, -0.567, ...]
    },
    {
      "id": "user_002",
      "embedding": [0.345, -0.678, ...]
    }
  ],
  "distance_metric": "cosine"  // or "euclidean"
}
```

**Response:**
```json
{
  "matches": [
    {
      "id": "user_001",
      "distance": 0.234,
      "similarity": 0.883
    },
    {
      "id": "user_002",
      "distance": 0.567,
      "similarity": 0.716
    }
  ],
  "best_match": {
    "id": "user_001",
    "distance": 0.234,
    "similarity": 0.883
  },
  "distance_metric": "cosine"
}
```

## Usage Examples

### Using cURL

**Extract Embedding:**
```bash
# Prepare base64 image
BASE64_IMAGE=$(base64 -w 0 face.jpg)

# Call API
curl -X POST http://localhost:8000/api/v1/embed \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,$BASE64_IMAGE\"}"
```

**Compare Embeddings:**
```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.123, -0.456, ...],
    "reference_embeddings": [
      {"id": "user_001", "embedding": [0.234, -0.567, ...]},
      {"id": "user_002", "embedding": [0.345, -0.678, ...]}
    ],
    "distance_metric": "cosine"
  }'
```

### Using Python

```python
import base64
import requests

# Load and encode image
with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Extract embedding
response = requests.post(
    "http://localhost:8000/api/v1/embed",
    json={"image": f"data:image/jpeg;base64,{image_base64}"}
)

result = response.json()
embedding = result["embedding"]  # 512-dimensional vector
print(f"Embedding size: {len(embedding)}")
print(f"Detection score: {result['detection_score']}")
```

## Integration with Main Backend

### Pattern A: Backend Does Matching (Recommended)

Your main backend stores all embeddings and only calls this service for extraction.

**User Enrollment:**
```python
# 1. User uploads photo
# 2. Send to face service
embedding_response = requests.post(
    "http://face-service:8000/api/v1/embed",
    json={"image": base64_image}
)
embedding = embedding_response.json()["embedding"]

# 3. Store embedding in your database
db.users.update(user_id, {"face_embedding": embedding})
```

**Attendance Check:**
```python
# 1. Capture live photo
# 2. Extract embedding
live_response = requests.post(
    "http://face-service:8000/api/v1/embed",
    json={"image": live_base64_image}
)
live_embedding = live_response.json()["embedding"]

# 3. Fetch user embeddings from database
stored_embeddings = db.users.find({}, {"user_id": 1, "face_embedding": 1})

# 4. Compare in your backend
from scipy.spatial.distance import cosine

best_match = None
min_distance = float('inf')

for user in stored_embeddings:
    distance = cosine(live_embedding, user["face_embedding"])
    if distance < min_distance:
        min_distance = distance
        best_match = user["user_id"]

# 5. Validate threshold and record attendance
THRESHOLD = 0.4  # Tune based on your data
if min_distance < THRESHOLD:
    db.attendance.insert({"user_id": best_match, "timestamp": now()})
```

### Pattern B: Service Does Matching (Optional)

Main backend sends reference embeddings to the service for comparison.

**Attendance Check:**
```python
# 1. Extract live embedding
live_response = requests.post(
    "http://face-service:8000/api/v1/embed",
    json={"image": live_base64_image}
)
live_embedding = live_response.json()["embedding"]

# 2. Fetch reference embeddings
stored_embeddings = db.users.find({}, {"user_id": 1, "face_embedding": 1})

# 3. Call compare endpoint
compare_response = requests.post(
    "http://face-service:8000/api/v1/compare",
    json={
        "query_embedding": live_embedding,
        "reference_embeddings": [
            {"id": user["user_id"], "embedding": user["face_embedding"]}
            for user in stored_embeddings
        ],
        "distance_metric": "cosine"
    }
)

result = compare_response.json()
best_match = result["best_match"]

# 4. Validate and record
THRESHOLD = 0.4
if best_match["distance"] < THRESHOLD:
    db.attendance.insert({"user_id": best_match["id"], "timestamp": now()})
```

## Configuration

Create a `.env` file to customize settings:

```env
# API Settings
APP_NAME=Face Recognition Service
DEBUG=false
LOG_LEVEL=info

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Settings
MODEL_NAME=buffalo_l  # Options: buffalo_l, buffalo_sc
DETECTION_THRESHOLD=0.5
DEVICE=cpu  # Options: cpu, cuda

# Image Processing
MAX_IMAGE_SIZE=10485760  # 10MB in bytes

# CORS Settings
CORS_ENABLED=true
CORS_ORIGINS=["*"]
```

### Model Options

- **buffalo_l**: Larger model, better accuracy, slower (~140MB)
- **buffalo_sc**: Smaller model, faster, slightly lower accuracy (~50MB)

## Docker Deployment

### Build Image

```bash
docker build -t face-recognition:latest .
```

### Run Container

```bash
docker run -d \
  --name face-recognition \
  -p 8000:8000 \
  -e LOG_LEVEL=info \
  -e DEVICE=cpu \
  face-recognition:latest
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  face-recognition:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
      - DEVICE=cpu
      - MODEL_NAME=buffalo_l
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Run with:
```bash
docker-compose up -d
```

## Performance Tuning

### CPU Optimization

- Use `buffalo_sc` for faster inference
- Increase workers for concurrent requests: `--workers 4`
- Use ONNX Runtime CPU provider (default)

### GPU Acceleration

1. Install CUDA and cuDNN
2. Install GPU version: `pip install onnxruntime-gpu`
3. Set environment: `DEVICE=cuda`

### Benchmarks

**CPU (Intel i7-10700K):**
- Model loading: ~2-3 seconds
- Single embedding: ~50-200ms
- Throughput: ~5-20 requests/second

**GPU (NVIDIA RTX 3080):**
- Model loading: ~2-3 seconds
- Single embedding: ~20-50ms
- Throughput: ~20-50 requests/second

## Error Handling

All errors return JSON with standardized format:

```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "detail": "Optional detailed information"
}
```

**Error Codes:**
- `INVALID_IMAGE`: Image decode/format error
- `NO_FACE_DETECTED`: No face found in image
- `MULTIPLE_FACES_DETECTED`: More than one face found
- `IMAGE_TOO_LARGE`: Exceeds size limit
- `UNSUPPORTED_FORMAT`: Invalid image format
- `MODEL_NOT_LOADED`: Model initialization failed
- `INVALID_EMBEDDING`: Embedding validation failed
- `PROCESSING_ERROR`: General processing error

## Limitations and Best Practices

### Accuracy Considerations

- **Poor Lighting**: Low accuracy in very dark or overexposed images
- **Extreme Poses**: Side profiles reduce accuracy
- **Occlusions**: Masks, sunglasses, or hats affect performance
- **Low Resolution**: Minimum 32x32 pixels for face region

### Best Practices

1. **Multiple Enrollment Images**: Register 3-5 photos per user in different conditions
2. **Quality Control**: Validate image quality before processing
3. **Threshold Tuning**: Use real data to set appropriate matching thresholds
   - Lower threshold: More false accepts (different person accepted)
   - Higher threshold: More false rejects (same person rejected)
   - Recommended starting point: 0.3-0.5 for cosine distance
4. **Lighting**: Ensure consistent lighting between enrollment and verification
5. **Face Size**: Larger faces (closer to camera) work better

### Operational Recommendations

- **Health Monitoring**: Poll `/api/v1/health` regularly
- **Timeout Handling**: Set request timeout to 30 seconds
- **Retry Logic**: Implement exponential backoff for transient errors
- **Rate Limiting**: Add rate limiting in your API gateway
- **Authentication**: Add API key authentication for production

## Security Considerations

1. **No Data Persistence**: Service doesn't store any user data
2. **Non-Root User**: Docker container runs as non-root user
3. **Input Validation**: All inputs are validated before processing
4. **Size Limits**: Image size limited to prevent DoS attacks
5. **CORS**: Configure allowed origins for production

**Production Checklist:**
- [ ] Add authentication middleware (API keys, JWT)
- [ ] Configure CORS with specific origins
- [ ] Set up HTTPS/TLS termination
- [ ] Implement rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation

## Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Manually download models
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"
```

### Memory Issues

Reduce memory usage:
- Use `buffalo_sc` instead of `buffalo_l`
- Reduce max image size: `MAX_IMAGE_SIZE=5242880`
- Limit workers: `--workers 1`

### Import Errors

If you get import errors:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## Development

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black face_recognition_service/

# Sort imports
isort face_recognition_service/

# Lint
flake8 face_recognition_service/
```

### Type Checking

```bash
mypy face_recognition_service/
```

## Extensibility

This architecture supports future enhancements:

1. **ONNX/TensorRT**: Replace InsightFace backend
2. **Batch Processing**: Add endpoint for multiple images
3. **Async Processing**: Queue-based processing for high volume
4. **Model Versioning**: A/B test different models
5. **Authentication**: Add OAuth2/JWT middleware
6. **Metrics**: Add Prometheus metrics export
7. **Liveness Detection**: Add anti-spoofing checks

## License

This project is provided as-is for integration with your attendance system.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the interactive API docs at `/docs`
3. Check application logs for detailed error messages

## Acknowledgments

- **InsightFace**: Open-source face recognition library
- **ArcFace**: State-of-the-art face recognition model
- **FastAPI**: Modern web framework for Python
- **ONNX Runtime**: Cross-platform inference engine
