# Face Recognition Service

A stateless, production-ready Python microservice for face embedding extraction and comparison using ArcFace via InsightFace. This service is designed to replace third-party face recognition providers while keeping all user data and business logic in your own backend.

## Features

- **Stateless Design**: No database, no file storage, no user data persistence
- **Modern Stack**: FastAPI + InsightFace (ArcFace) + ONNX Runtime
- **Production-Ready**: Docker support, health checks, comprehensive error handling
- **Horizontally Scalable**: Multiple instances can run independently
- **CPU-Optimized**: Efficient inference on CPU with optional GPU support
- **RESTful API**: Clean JSON endpoints with automatic OpenAPI documentation

## Architecture

This microservice is designed to work alongside your existing backend:

```
┌─────────────────────────────────────────────┐
│         Your Backend / Calling App          │
│  ┌────────────────────────────────────────┐ │
│  │ • User Management                      │ │
│  │ • Embedding Storage (Database)         │ │
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
│  │ │ POST /api/v1/compare-photos        │ │ │
│  │ │   → Fetch+compare two photos       │ │ │
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
pip install -r requirements.txt -c constraints.txt
```

On first run, InsightFace will automatically download the model pack (`MODEL_NAME`, default `antelopev2`, ~350MB) if it isn't already cached under `~/.insightface/`.

#### 4. Run the Service

```bash
# Development mode (with auto-reload)
uvicorn face_recognition_service.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python (no auto-reload, even with DEBUG=true -- see below)
python -m face_recognition_service.main
```

`python -m face_recognition_service.main` always runs with `reload=False`,
regardless of `DEBUG`. This is also how the container's `CMD` starts the
service, and it runs as PID 1 there: uvicorn's reloader spawns a supervisor
subprocess that doesn't forward signals the same way a single process does,
which breaks graceful shutdown under `docker stop`/SIGTERM. For local
auto-reload during development, run `uvicorn ... --reload` directly as
shown above instead.

The service will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

**GET** `/api/v1/health`

Check if the service is running and the model is loaded. Does not require
authentication.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "antelopev2"
}
```

Returns HTTP 503 (still valid JSON, `"status": "unhealthy"`) while the model
hasn't finished loading yet -- expected for a short window right after
startup, not a failure by itself.

### 2. Model Information

**GET** `/api/v1/model-info`

Get information about the loaded face recognition model. Requires the
bearer token.

**Response:**
```json
{
  "name": "antelopev2",
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
  "detection_score": 0.98,
  "quality": {
    "face_size_px": 210.0,
    "interocular_px": 68.4,
    "roll_deg": 1.2,
    "yaw_proxy": 0.03,
    "blur_variance": 812.5,
    "embedding_norm": 24.1,
    "faces_considered": 1,
    "second_face_ratio": 0.0
  }
}
```

The embedding is L2-normalised (unit length); compare embeddings with cosine distance, or euclidean distance on unit vectors.

`quality` is always **informational** -- it is reported whether or not any quality gate is enabled, and every gate defaults off (see "Configuration" below), so its presence never changes whether a request succeeds. Each metric, one line:
- `face_size_px`: shorter side of the detected face's bounding box, in pixels -- bigger is generally better.
- `interocular_px`: distance between the two eye landmarks, in pixels; `null` if no landmarks were found.
- `roll_deg`: in-plane head tilt in degrees (0 = level); `null` without landmarks.
- `yaw_proxy`: left/right head-turn proxy, ~0 when frontal, sign gives direction; `null` without landmarks.
- `blur_variance`: variance of the Laplacian on the aligned face crop -- lower means blurrier; `null` without landmarks.
- `embedding_norm`: L2 norm of the raw embedding before it's unit-normalised; very low values often mean a poor-quality face.
- `faces_considered`: how many detected faces met the detector's own quality threshold.
- `second_face_ratio`: area of the next-largest qualifying face relative to the chosen face's area; `0` if there was only one.

**Error Cases:**
- No face detected: `error_code: "NO_FACE_DETECTED"`
- Face(s) found but too low quality: `error_code: "FACE_LOW_QUALITY"` (only when a quality gate is enabled -- all gates default off, see "Configuration" below)
- Multiple faces: `error_code: "MULTIPLE_FACES_DETECTED"` (opt-in only, via `MULTI_FACE_POLICY=reject_ambiguous`; with the default `largest` policy the dominant face is always used and this code is never returned -- see the warning in "Configuration")
- Invalid image: `error_code: "INVALID_IMAGE"`

### 4. Compare Photos (URL + upload)

**POST** `/api/v1/compare-photos`

Fetches `image1` from a URL and compares it against an uploaded `image2`
file. Both `multipart/form-data` fields.

**Request (multipart form):**
- `image1`: `https://example.com/reference.jpg` (a URL, plain text field)
- `image2`: uploaded file
- `distance_metric`: `"cosine"` or `"euclidean"` (optional, default `"cosine"`)

**Response:**
```json
{
  "match": true,
  "similarity": 0.91,
  "distance": 0.18,
  "distance_metric": "cosine",
  "threshold": 0.5,
  "cosine_similarity": 0.91,
  "image1_detection_score": 0.99,
  "image2_detection_score": 0.97,
  "image1_quality": { "face_size_px": 240.0, "interocular_px": 78.2, "...": "..." },
  "image2_quality": { "face_size_px": 205.0, "interocular_px": 66.1, "...": "..." }
}
```

`image1_quality`/`image2_quality` have the same shape and meaning as
`quality` on `/embed` above -- purely informational. `image1` is analysed
with `role="reference"` and `image2` with `role="selfie"`; quality gates and
the multi-face policy are opt-in (every threshold defaults off) and, even
when turned on, only apply to the selfie role unless
`QUALITY_GATES_APPLY_TO_REFERENCE=true` -- a rejected reference photo can
lock someone out of matching entirely, so it never happens by surprise.

### 5. Compare Photos (upload + upload)

**POST** `/api/v1/compare-photos-upload`

Same response shape as `/compare-photos`, but both `image1` and `image2`
are uploaded files (no reference-URL fetch). Convenient for testing via the
interactive docs at `/docs` without a publicly reachable reference URL.

### 6. Compare Embeddings

**POST** `/api/v1/compare`

Compare a query embedding against multiple reference embeddings, when your
backend already stores embeddings itself rather than photos/URLs.

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
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/jpeg;base64,$BASE64_IMAGE\"}"
```

**Compare Photos (reference URL + selfie upload):**
```bash
curl -X POST http://localhost:8000/api/v1/compare-photos \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "image1=https://example.com/reference.jpg" \
  -F "image2=@selfie.jpg" \
  -F "distance_metric=cosine"
```

### Using Python

```python
import base64
import requests

API_TOKEN = "..."

# Load and encode image
with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Extract embedding
response = requests.post(
    "http://localhost:8000/api/v1/embed",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    json={"image": f"data:image/jpeg;base64,{image_base64}"}
)

result = response.json()
embedding = result["embedding"]  # 512-dimensional vector
print(f"Embedding size: {len(embedding)}")
print(f"Detection score: {result['detection_score']}")
```

## Integration Patterns

### Pattern A: Backend Does Matching (Recommended)

Your backend stores all embeddings and only calls this service for
extraction (`/embed`).

```python
# 1. User uploads photo
# 2. Send to face service
embedding_response = requests.post(
    "http://face-service:8000/api/v1/embed",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    json={"image": base64_image}
)
embedding = embedding_response.json()["embedding"]

# 3. Store embedding in your own database
db.users.update(user_id, {"face_embedding": embedding})

# --- later, at verification time ---

live_response = requests.post(
    "http://face-service:8000/api/v1/embed",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    json={"image": live_base64_image}
)
live_embedding = live_response.json()["embedding"]

# 4. Compare in your own backend
from scipy.spatial.distance import cosine

best_match, min_distance = None, float("inf")
for user in db.users.find({}, {"user_id": 1, "face_embedding": 1}):
    distance = cosine(live_embedding, user["face_embedding"])
    if distance < min_distance:
        min_distance, best_match = distance, user["user_id"]

THRESHOLD = 0.5  # tune against your own data -- see "Threshold guidance"
if min_distance < THRESHOLD:
    record_match(best_match)
```

### Pattern B: Service Fetches and Compares Photos Directly

Useful when your backend already has a stable reference-photo URL per user
and doesn't want to manage embeddings at all -- see `/compare-photos` above.

```python
response = requests.post(
    "http://face-service:8000/api/v1/compare-photos",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    data={"image1": reference_photo_url, "distance_metric": "cosine"},
    files={"image2": open("selfie.jpg", "rb")},
)
result = response.json()
if result["match"]:
    record_match(user_id)
```

## Configuration

Copy `.env.example` to `.env` and customize it -- `.env.example` is the source
of truth for every setting (grouped, with defaults and warnings) and is kept
in sync with `face_recognition_service/config.py` by `tests/test_config_drift.py`.
The table below is a quick-reference summary; see `.env.example` for the full
commentary.

| Variable | Default | Meaning |
|---|---|---|
| `APP_NAME` | `Face Recognition Service` | Display name, e.g. in `/health`. |
| `APP_VERSION` | `0.1.0` | Reported version string. |
| `API_V1_PREFIX` | `/api/v1` | URL prefix for versioned endpoints. |
| `DEBUG` | `false` | FastAPI debug mode. |
| `API_TOKEN` | *(required, no default)* | Bearer token every request must present. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Bind port. |
| `WORKERS` | `1` | Uvicorn worker processes. |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error`, `critical`. |
| `CORS_ENABLED` | `false` | Enable CORS middleware. Off by default: the intended caller is server-to-server (bearer token, not a browser). When enabled, credentials are never allowed (`allow_credentials=False`) regardless of `CORS_ORIGINS`. |
| `CORS_ORIGINS` | `["*"]` | JSON array of allowed origins. |
| `CORS_METHODS` | `["*"]` | JSON array of allowed methods. |
| `CORS_HEADERS` | `["*"]` | JSON array of allowed headers. |
| `MODEL_NAME` | `antelopev2` | `antelopev2` (default, matches what the Dockerfile bakes into the image), `buffalo_l`, `buffalo_sc`. |
| `DETECTION_THRESHOLD` | `0.5` (`0.1` under docker-compose.yml/`.env.example`) | InsightFace `det_thresh`: minimum score to count as a detected face. `0.5` is `config.py`'s conservative library default; `0.1` is the deployment starting point shipped in compose/`.env.example` -- re-measure per `docs/evaluation.md` before trusting it for your own traffic. |
| `MIN_FACE_QUALITY` | `0.7` (`0.1` under docker-compose.yml/`.env.example`) | Minimum `det_score` to accept the detection at all. Same split and caveat as `DETECTION_THRESHOLD` above. |
| `EMBEDDING_SIZE` | `512` | ArcFace embedding dimensionality. |
| `COSINE_MATCH_THRESHOLD` | `0.5` | Match when cosine distance is below this. See "Threshold guidance" below before changing. |
| `DET_SIZE` | `640` | Detector input size (square). |
| `PAD_RETRY_RATIO` | `0.5` | On zero detections, retry once on a copy padded by this fraction of the longer side (`0` = off). |
| `ORT_INTRA_OP_THREADS` | `0` (`2` under docker-compose.yml) | ONNX Runtime intra-op threads (`0` = runtime default; set to the container's CPU quota). docker-compose.yml passes `2`, matching its own `cpus: 2` limit. |
| `ORT_INTER_OP_THREADS` | `0` (`1` under docker-compose.yml) | ONNX Runtime inter-op threads (`0` = runtime default). |
| `DEVICE` | `cpu` | `cpu` or `cuda`. |
| `PROVIDERS` | *(derived from `DEVICE`)* | JSON array of ONNX Runtime providers; overrides `DEVICE`-derived defaults when set. |
| `MAX_IMAGE_SIZE` | `10485760` | Max encoded image size in bytes (10 MB). |
| `MAX_REQUEST_BODY_BYTES` | `26214400` | Hard cap on the whole request body (25 MB), enforced by ASGI middleware before any route runs. Must exceed two images plus base64/multipart overhead. Enforced before authentication, so an oversize request is rejected without needing a valid token -- only the size limit itself is revealed to an unauthenticated caller, nothing about the image/model. `nginx/nginx.conf`'s `client_max_body_size` is set to `32M`, strictly above this default (both reject on strictly-greater-than, so an equal cap would let nginx's opaque 413 fire first) -- keep nginx's cap strictly larger if you raise this. |
| `MAX_IMAGE_PIXELS` | `50000000` | Max decoded width×height, checked from the header before decoding. |
| `MAX_IMAGE_SIDE` | `2048` | Downscale so the longer side is at most this many pixels (`0` = keep original size). |
| `ALLOWED_IMAGE_FORMATS` | `["jpg","jpeg","png","bmp","webp","mpo"]` | JSON array of accepted image formats. |
| `ENHANCE_IMAGE` | *(unset)* | **Deprecated**, use `ENHANCE_MODE`. `true`/`false` maps to `ENHANCE_MODE=always`/`off`. Do not set both. Under the provided `docker-compose.yml`, this variable is never passed to the container (absent from the compose file's `environment:` list) -- setting it in the host `.env` has no effect there; set `ENHANCE_MODE` in that `.env` instead. |
| `ENHANCE_MODE` | *(unset; effective default `detect_fallback`)* | `always` (always enhance before detecting/embedding), `off` (never enhance), `detect_fallback` (detect on the original first, only retry on an enhanced copy if nothing is found; always embed from the original). |
| `MIN_FACE_SIZE_PX` | `0` (off) | Reject if the bbox's shorter side is below this. |
| `MIN_INTEROCULAR_PX` | `0` (off) | Reject if eye-to-eye distance is below this. |
| `MIN_BLUR_VARIANCE` | `0` (off) | Reject if the aligned crop's Laplacian variance is below this. |
| `MAX_ABS_YAW_PROXY` | `0` (off) | Reject if `|yaw_proxy|` exceeds this. |
| `MAX_ABS_ROLL_DEG` | `0` (off) | Reject if `|roll_deg|` exceeds this. |
| `MIN_EMBEDDING_NORM` | `0` (off) | Reject if the raw embedding norm is below this. |
| `QUALITY_GATES_APPLY_TO_REFERENCE` | `false` | Opt-in: apply the gates above to the reference photo too, not just the selfie. |
| `MULTI_FACE_POLICY` | `largest` | `largest` (always accept the dominant face) or `reject_ambiguous` (raise `MULTIPLE_FACES_DETECTED` when a second qualifying face is close in score/size). |
| `MULTI_FACE_AREA_RATIO` | `0.5` | Second face must be at least this fraction of the chosen face's area to count as ambiguous. |
| `MULTI_FACE_MIN_SCORE` | `0.5` | Second face must have at least this `det_score` to count as ambiguous. |
| `FETCH_CONNECT_TIMEOUT` | `5` | Per-hop connect timeout (seconds) for the reference fetch. |
| `FETCH_READ_TIMEOUT` | `10` | Per-hop read timeout (seconds) for the reference fetch. |
| `FETCH_TOTAL_TIMEOUT` | `20` | Wall-clock budget (seconds) for the whole reference fetch, all hops and reads included. See "Concurrency" below. |
| `FETCH_MAX_REDIRECTS` | `3` | Redirects are followed manually, at most this many hops; each hop is re-checked against the host/IP policy below. |
| `REFERENCE_URL_ALLOWED_HOSTS` | `[]` | JSON array of hosts the reference URL (and any redirect target) may use. Empty = any host (still subject to the always-blocked IP classes). See "Security Considerations" below. |
| `REFERENCE_URL_ALLOW_PRIVATE` | `true` | Whether the reference URL may resolve to a private/loopback address. Link-local/multicast/unspecified/reserved (cloud-metadata class) addresses are always blocked regardless. See "Security Considerations" below. |
| `MAX_CONCURRENT_INFERENCE` | `1` | Max concurrent `model.analyze` calls (>=1). See "Concurrency" below. |
| `INFERENCE_QUEUE_TIMEOUT` | `20.0` | Seconds a request waits for a free inference slot before returning `503 SERVICE_BUSY`. |
| `BUSY_RETRY_AFTER_SECONDS` | `5` | `Retry-After` header value (seconds) sent with a `503 SERVICE_BUSY` response. |

**Quality gates and the multi-face policy are off by default, on purpose.**
Every threshold above defaults to `0` (disabled) and `MULTI_FACE_POLICY`
defaults to `largest`, so enabling none of them reproduces today's
behaviour exactly. Turning one on creates a *new* rejection that did not
happen before. The calling application's error handling only recognizes a
fixed set of 400 error codes; any error code it doesn't recognize --
including `MULTIPLE_FACES_DETECTED` if it hasn't been updated to handle it
-- is shown to the end user as a generic service outage, not a specific
rejection reason. Don't enable a gate or `MULTI_FACE_POLICY` without
confirming the calling application handles the resulting code, and without
checking real quality-metric distributions (see `docs/evaluation.md`) so a
gate doesn't reject photos that are already being accepted today.

### Concurrency

`model.analyze` (CPU-bound face detection/embedding) runs off the event loop
in a threadpool, and only `MAX_CONCURRENT_INFERENCE` calls run at once
(`concurrency.InferenceGate`) -- a request that can't get a slot within
`INFERENCE_QUEUE_TIMEOUT` seconds gets a fast `503 SERVICE_BUSY` (with a
`Retry-After: BUSY_RETRY_AFTER_SECONDS` header) instead of piling up
indefinitely. Reference/selfie fetching and decoding also run off the event
loop, but outside the gate, so a slow upstream fetch never holds up another
request's inference slot. The reference fetch itself is bounded by a real
wall-clock deadline (`FETCH_TOTAL_TIMEOUT`, default 20s, started once before
the first hop) enforced two ways: an in-band check between streamed chunks
and before each hop, and a `threading.Timer` watchdog armed for the whole
fetch that force-closes the underlying socket if it fires -- this second
mechanism is the one that actually matters, since requests' own
`timeout=(connect, read)` only bounds a single socket operation, not the
whole call, so a server that drips one byte (or one header line) every
400ms with a 1s read timeout never trips it on its own and would otherwise
block `requests.get()` or `iter_content()` indefinitely. A connection opened
*after* the watchdog has already fired (e.g. because this thread was still
blocked in a slow DNS lookup when the deadline hit) is shut down the moment
it's created instead, so a late socket can't escape the one-shot watchdog
callback either. Actual guarantee: every fetch finishes within
`FETCH_TOTAL_TIMEOUT` of being called, plus the watchdog's own scheduling
slop (sub-second in practice), plus a DNS residual: each hop's host/IP
policy check resolves the host via the OS resolver (`socket.getaddrinfo`, up
to two uninterruptible lookups -- A and AAAA -- per hop) *before* that hop's
own timeout budget is applied, and a slow resolver has no timeout of its own
here and can't be interrupted by the watchdog, so it adds on top of
`FETCH_TOTAL_TIMEOUT` rather than counting against it. Budget: queue wait
(`INFERENCE_QUEUE_TIMEOUT`, default 20s) plus the fetch
(`FETCH_TOTAL_TIMEOUT`, default 20s, plus scheduling slop and the DNS
residual above) plus inference itself (well under a minute) must stay under
whatever client timeout the calling application uses, so it never times out
waiting on us before we've had a chance to answer.

### Threshold guidance

`COSINE_MATCH_THRESHOLD` ships at `0.5` as a starting point, not a value
measured on your own population. Before trusting it (or any of the
detection/quality thresholds above) in production:

- Run `python -m evaluation` against your own labelled data (see
  [docs/evaluation.md](docs/evaluation.md)) and pick a threshold from the
  resulting FAR/FRR curve for your actual target false-accept rate.
- **Don't tune against LFW alone.** `docs/evaluation.md` ships a public LFW
  baseline as a worked example of the harness's output, not a production
  recommendation -- a public dataset of unrelated celebrity photo pairs
  systematically understates how close a real impostor (someone
  photographed by the same camera/kiosk, under similar lighting) can get to
  a genuine match, so a threshold tuned only on LFW risks being less safe
  than your actual population needs.
- Re-run the evaluation whenever the model pack, enhancement mode, or
  preprocessing pipeline changes -- a threshold calibrated for one
  configuration doesn't necessarily transfer to another.

See `docs/evaluation.md` for the full methodology, datasets, and caveats
(selection bias in the genuine set, pair-sharing effects on sample size,
etc.) before changing this value.

### Logging

Every request gets a request ID: an inbound `X-Request-ID` header is
echoed back if it matches `^[A-Za-z0-9._-]{1,64}$`, otherwise a fresh one
(`uuid4().hex`) is generated. Either way it comes back as the response's own
`X-Request-ID` header, and every log line emitted while handling that
request -- including from the fetch/decode/inference work that runs off the
event loop in a threadpool -- carries it as `[request_id]` in the log
format, so a support ticket that quotes the header can be grepped straight
out of the logs.

`/compare-photos`, `/compare-photos-upload` and `/embed` each emit exactly
one summary line per request, at INFO level, on the
`face_recognition_service.summary` logger, whether the request succeeded or
failed. It's a flat set of `key=value` pairs, never a URL, host, path,
filename, embedding or image byte:

| Field | Meaning |
|---|---|
| `endpoint` | `compare-photos`, `compare-photos-upload`, or `embed`. |
| `outcome` | `match` / `no_match` (compare endpoints), `ok` (embed), or `error`. |
| `error_code` | The response's `error_code` on failure; `-` on success. |
| `image` | `reference` / `selfie` if the error was tagged to one photo; `-` otherwise. |
| `match` | `true` / `false` on a compare endpoint; `-` for embed or on error. |
| `cosine_similarity` | The comparison's cosine similarity, 4 decimals; `-` if not computed. |
| `threshold` | The match threshold actually used (translated into the request's metric), 4 decimals. |
| `det1` / `det2` | Detection score of image1/image2 (reference/selfie), 4 decimals. |
| `fetch_ms` | Time spent fetching the reference photo from its URL; `-` when both photos are uploads. |
| `decode_ms` | Time spent decoding/preprocessing upload(s) or the /embed base64 image. |
| `queue_ms` | Time spent waiting for a free inference slot (`concurrency.InferenceGate`) -- present even on a `503 SERVICE_BUSY`. |
| `infer_ms` | Time spent in the model's own `analyze` call(s). |
| `total_ms` | Wall-clock time for the whole request. |
| `q1_face_px` / `q2_face_px` | `face_size_px` quality metric for the reference/selfie photo. |
| `q2_blur` | `blur_variance` quality metric for the selfie photo. |

Timings are whole milliseconds; missing values (a stage never reached, or a
metric the model didn't compute) print as `-`.

`LOG_LEVEL` applies in Docker: the image's `CMD` runs
`python -m face_recognition_service.main`, which goes through this
service's own `if __name__ == "__main__":` block (`uvicorn.run(...,
log_level=settings.log_level)`) instead of invoking `uvicorn` directly, so
setting `LOG_LEVEL` in the container's environment actually changes what
gets logged.

## Docker Deployment

For the full production deploy procedure (nginx + certbot topology,
required/recommended `.env` values, capacity tuning, security notes, and a
full troubleshooting table by error code), see
**[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**. This section is only a
quick local/manual reference.

### Build Image

```bash
docker build -t face-recognition:latest .
```

### Run Container

`API_TOKEN` is required -- the service will not start without it:

```bash
docker run -d \
  --name face-recognition \
  -p 8000:8000 \
  -e API_TOKEN=$(openssl rand -hex 32) \
  -e LOG_LEVEL=info \
  -e DEVICE=cpu \
  face-recognition:latest
```

### Docker Compose

This repo ships its own `docker-compose.yml`, built for a self-contained
nginx + certbot topology (bundled reverse proxy, Cloudflare DNS-01 for TLS,
ports 80/443 published). It requires `API_TOKEN` in the environment or a
`.env` file next to it:

```bash
cp .env.example .env
# edit .env: set API_TOKEN, DOMAIN_NAME, LETSENCRYPT_EMAIL, CLOUDFLARE_API_TOKEN at minimum
# then issue the first TLS certificate BEFORE `up` -- see DEPLOYMENT_GUIDE.md §4,
# nginx will crash-loop without one on a fresh clone
API_TOKEN=<your-token> docker compose up -d
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for the complete procedure,
including the Cloudflare/certbot setup, the one-shot certificate issuance
step, and nginx domain substitution.

## Performance Tuning

### CPU Optimization

- `MODEL_NAME=buffalo_l` or `buffalo_sc` are optional, lighter packs for faster inference
  than the `antelopev2` default. **Every threshold above is a starting point
  calibrated for antelopev2** (see "Threshold guidance" and
  `docs/evaluation.md`); re-run `python -m evaluation --model buffalo_l` (or
  `buffalo_sc`) against your own data before switching packs in production,
  don't just swap the model name. Note the Dockerfile only bakes
  `antelopev2` into the image -- switching packs also means adjusting how
  the pack reaches the container (see DEPLOYMENT_GUIDE.md "Model pack").
- Set `ORT_INTRA_OP_THREADS` to the container's CPU quota (e.g. `2`) instead of leaving
  it at the `0` default, which lets ONNX Runtime size the thread pool from host cores --
  wrong inside a CPU-limited container.
- Increase workers for concurrent requests: `--workers 4`
- Use ONNX Runtime CPU provider (default)

### GPU Acceleration

1. Install CUDA and cuDNN
2. Install GPU version: `pip install onnxruntime-gpu`
3. Set environment: `DEVICE=cuda`

### Benchmarks

No benchmark numbers ship with this template -- measured throughput and
latency depend heavily on the host CPU, `ORT_INTRA_OP_THREADS`, and image
size. See DEPLOYMENT_GUIDE.md "Capacity tuning" for a benchmark procedure
you can run against your own deployment.

## Error Handling

All errors return JSON with standardized format:

```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "detail": "Optional detailed information",
  "image": "reference | selfie | null"
}
```

`image` says which photo an image-specific error is about, on `/compare-photos`
and `/compare-photos-upload` (`image1` = "reference", `image2` = "selfie").
It is `null` for server-side faults (`MODEL_NOT_LOADED`, `SERVICE_UNAVAILABLE`,
`PROCESSING_ERROR`) and on the single-image `/embed` endpoint.

**Error Codes:**
- `INVALID_IMAGE`: Image decode/format error
- `NO_FACE_DETECTED`: No face found in image
- `FACE_LOW_QUALITY`: Raised when no detected face meets `MIN_FACE_QUALITY` (the
  detector's own confidence gate, always on), or when an opt-in quality gate fails
  (`MIN_FACE_SIZE_PX`, `MIN_INTEROCULAR_PX`, `MIN_BLUR_VARIANCE`, `MAX_ABS_YAW_PROXY`,
  `MAX_ABS_ROLL_DEG`, `MIN_EMBEDDING_NORM`). Those gates default off (see "Configuration"
  above) and never apply to the reference photo unless `QUALITY_GATES_APPLY_TO_REFERENCE=true`.
- `MULTIPLE_FACES_DETECTED`: Opt-in only, via `MULTI_FACE_POLICY=reject_ambiguous` (default
  `largest` never returns this code). A calling application that only recognizes a fixed
  set of 400 codes may show this as a generic outage until it's updated to handle it --
  see "Configuration" above.
- `IMAGE_TOO_LARGE`: Exceeds size limit
- `UNSUPPORTED_FORMAT`: Invalid image format
- `REFERENCE_UNAVAILABLE`: The reference photo URL (`image1` on `/compare-photos`) itself is bad --
  a 4xx response (404, 403, 410, ...) from `raise_for_status()`. The photo's owner needs a fresh
  photo/link. `image: "reference"`. HTTP 400.
- `SERVICE_UNAVAILABLE`: The reference photo URL couldn't be *reached* at all -- timeout, DNS
  failure, TLS/connection error, or a 5xx from the far end. This is a fault in the fetch path, not
  evidence the link or photo is bad, so it is never tagged (`image: null`). HTTP 503.
- `REFERENCE_URL_NOT_ALLOWED`: The reference URL's host (or a redirect target's host) failed the
  host/IP policy -- see "Security Considerations" below. Never tagged (`image: null`). HTTP 400.
- `MODEL_NOT_LOADED`: Model initialization failed. HTTP 503.
- `INVALID_EMBEDDING`: Embedding validation failed
- `PROCESSING_ERROR`: Unexpected server-side processing error. HTTP 500.

Every other error code above is HTTP 400. A request/validation error raised as an
`HTTPException` (e.g. a malformed `distance_metric`) keeps its own status instead of
going through this envelope.

## Limitations and Best Practices

### Accuracy Considerations

- **Poor Lighting**: Low accuracy in very dark or overexposed images
- **Extreme Poses**: Side profiles reduce accuracy
- **Occlusions**: Masks, sunglasses, or hats affect performance
- **Low Resolution**: Minimum 32x32 pixels for face region
- **No liveness detection**: this service does not verify a photo was taken
  of a live person at capture time -- see DEPLOYMENT_GUIDE.md "Security notes"

### Best Practices

1. **Multiple Enrollment Images**: Register 3-5 photos per user in different conditions
2. **Quality Control**: Validate image quality before processing
3. **Threshold Tuning**: See "Threshold guidance" above -- calibrate against your own
   data with `docs/evaluation.md`, not a public benchmark alone.
   - Lower threshold: More false accepts (different person accepted)
   - Higher threshold: More false rejects (same person rejected)
4. **Lighting**: Ensure consistent lighting between enrollment and verification
5. **Face Size**: Larger faces (closer to camera) work better

### Operational Recommendations

- **Health Monitoring**: Poll `/api/v1/health` regularly
- **Timeout Handling**: Set your client timeout to at least 60 seconds
- **Retry Logic**: Implement exponential backoff for transient errors
- **Rate Limiting**: Configured at the bundled nginx layer (see `nginx/nginx.conf`), not in this service itself
- **Authentication**: Already required -- every request needs the `API_TOKEN` bearer token (see "Security Considerations")

## Security Considerations

1. **No Data Persistence**: Service doesn't store any user data
2. **Non-Root User**: Docker container runs as non-root user
3. **Input Validation**: All inputs are validated before processing
4. **Size Limits**: Image size limited to prevent DoS attacks
5. **CORS**: Configure allowed origins for production (off by default)

### Reference URL host/IP policy (SSRF hardening)

`fetch_image_from_url` (the reference photo fetch, `image1` on
`/compare-photos`) checks every hop -- the original URL and each redirect
target -- against a host/IP policy before requesting it:

- **The host is derived from what `requests`/urllib3 will actually connect
  to** (`urllib3.util.parse_url` on the prepared request URL), not from
  `urlsplit(url)` alone. A raw backslash in the netloc, or a userinfo
  (`user@host`) trick combined with one, can make `urlsplit` read a
  different host than the one urllib3 opens a socket to. A backslash in the
  netloc, or any disagreement between the two parses, fails closed as
  `REFERENCE_URL_NOT_ALLOWED` rather than trusting either parser. Both a
  bracketed IPv6 literal and an IPv4-mapped IPv6 address are normalised
  first, so neither form is either a false-positive mismatch or a way to
  smuggle a blocked address past the always-blocked list under a different
  representation.
- **`REFERENCE_URL_ALLOWED_HOSTS`** (default `[]`, any host allowed): a
  non-empty JSON array restricts the reference URL and every redirect
  target to an exact, case-insensitive host match (a trailing `.` is
  normalised away first). A miss is `REFERENCE_URL_NOT_ALLOWED`.
- **Always blocked, regardless of settings**: link-local, multicast,
  unspecified and reserved addresses -- the cloud-metadata class (e.g.
  `169.254.169.254`) included -- plus a short explicit list of known
  metadata endpoints that don't fall into any of those address *classes*
  (`100.100.100.200`, `fd00:ec2::254`). No legitimate reference photo ever
  lives there.
- **`REFERENCE_URL_ALLOW_PRIVATE`** (default `true`): also allows any
  address that isn't globally routable (`not ip.is_global` -- private/
  loopback ranges, CGNAT `100.64.0.0/10`, IPv6 ULA `fc00::/7`, and 6to4/
  Teredo transition addresses) when true. **Defaults to true on purpose** --
  blocking private addresses out of the box risks locking out every user
  if the calling application's reference-photo host happens to resolve to a
  private/CGNAT/ULA address in a given deployment; enabling the block is
  opt-in until that's been verified.

**Recommended production values, once the reference-photo host(s) are
known:** set `REFERENCE_URL_ALLOWED_HOSTS` to the real host(s) and
`REFERENCE_URL_ALLOW_PRIVATE=false`. Resolve those hosts first and confirm
they're not private, so the allowlist doesn't lock everyone out on day one.

Redirects are followed manually (`allow_redirects=False` on every request,
at most `FETCH_MAX_REDIRECTS` hops) specifically so a redirect can't be used
to reach a host/address the initial-URL check would have rejected -- every
hop, including redirect targets, goes through the same policy. A malformed
redirect target (bad scheme or no host) is `SERVICE_UNAVAILABLE`, not
`INVALID_IMAGE` -- it's a fault in the far end's redirect, not evidence the
original reference photo is bad. The original URL keeps `INVALID_IMAGE` for
the same bad-scheme/no-host case, since that one really is the caller's own
input.

**Residual risks:**
- **DNS-rebinding TOCTOU**: the host is resolved and checked, then requested
  moments later; a host could resolve to an allowed address for the check
  and a different (blocked) one by the time `requests` actually connects.
  This is a narrow window, not eliminated by this hardening.
- **Resolver time is unbounded by `FETCH_TOTAL_TIMEOUT`**: `_resolve_host`'s
  own wall-clock cost is bounded only by the OS resolver, not by the fetch
  deadline (the deadline starts before the first hop's resolution).

### Reference fetch byte/time caps

- Response bytes are streamed (`iter_content(65536)`) with a running total
  capped at `MAX_IMAGE_SIZE`; an oversize `Content-Length` short-circuits
  before any bytes are read. `iter_content` transparently decodes
  `Content-Encoding` (gzip/deflate), so the cap is already on *decoded*
  bytes, not wire bytes -- but a single ~64 KiB compressed read can still
  decompress to several MB in memory before that chunk is handed back for
  the running-total check, so a highly compressible single chunk is a
  transient memory spike this cap doesn't prevent.
- The whole fetch (every hop, every read) has a `FETCH_TOTAL_TIMEOUT`
  wall-clock budget, enforced two ways: an in-band check between streamed
  chunks and before each hop, and a `threading.Timer` watchdog armed for the
  whole fetch that force-closes the underlying socket if the deadline fires
  while still blocked. The watchdog is what actually matters --
  `timeout=(connect, read)` only bounds a single socket read, not the whole
  call, so a server that drips data (or headers) slower than the read
  timeout but faster than it never trips it would otherwise block
  `requests.get()`/`iter_content()` indefinitely and pin a threadpool
  worker. A connection opened after the watchdog has already fired is shut
  down the moment it's created, so a late socket (e.g. one opened right
  after a slow DNS lookup) can't escape the one-shot watchdog callback
  either. A forced close always surfaces as `SERVICE_UNAVAILABLE` ("timed
  out"). Actual guarantee: every fetch finishes within `FETCH_TOTAL_TIMEOUT`
  of being called, plus the watchdog's own scheduling slop (sub-second in
  practice) plus the DNS residual above (up to two uninterruptible
  `getaddrinfo` lookups per hop).

**Production Checklist:** see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for
the full deploy procedure. Summary:
- [ ] Set a generated `API_TOKEN` (required -- compose refuses to start without it)
- [ ] Leave `CORS_ENABLED=false` unless a browser-based caller actually needs it
- [ ] Set `REFERENCE_URL_ALLOWED_HOSTS` to your real reference-photo host(s) and
      set `REFERENCE_URL_ALLOW_PRIVATE=false`
- [ ] Re-measure `DETECTION_THRESHOLD`/`MIN_FACE_QUALITY`/`COSINE_MATCH_THRESHOLD` against
      your own data with `docs/evaluation.md` before trusting the shipped starting points
- [ ] Set up monitoring/alerting and log aggregation on the summary log line (see "Logging" above)

## Troubleshooting

### Model Pack Resolution and Download Issues

The service resolves its model pack under `INSIGHTFACE_HOME` (default
`~/.insightface`, or `/app/.insightface` inside the Docker image -- see
`Dockerfile`) at `<INSIGHTFACE_HOME>/models/<MODEL_NAME>`. It accepts either
a flat layout (`.onnx` files directly under that directory, as `buffalo_l`
and `buffalo_sc` extract) or the nested `<name>/<name>/*.onnx` layout the
upstream `antelopev2` zip extracts to -- both are detected automatically
(`face_recognition_service/models/loader.py:resolve_pack_dir`). If the pack
isn't present under either layout, it's downloaded and unzipped on first
use.

The Docker image bakes `antelopev2` into `/app/.insightface/models/antelopev2`
at build time (see `Dockerfile`), so a fresh container never needs to
download it. Running locally, if you haven't fetched `antelopev2` yet, either
let the service download it on first use, or point `INSIGHTFACE_HOME` at a
directory where you've already placed it in one of the two layouts above.
`buffalo_l` and `buffalo_sc` are commonly cached already from earlier
InsightFace use, since they extract flat.

### Memory Issues

Reduce memory usage:
- Use the lighter `buffalo_l` or `buffalo_sc` pack instead of `antelopev2` -- re-evaluate
  thresholds first, see "CPU Optimization" above
- Reduce max image size: `MAX_IMAGE_SIZE=5242880`
- Limit workers: `--workers 1`

### Import Errors

If you get import errors:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt -c constraints.txt
```

## Development

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt -c constraints.txt
```

### Lint, Type Check, and Test

```bash
ruff check .            # lint (import order, pyflakes, bugbear)
mypy face_recognition_service   # advisory type check
pytest -m "not integration"     # offline tests
pytest -m integration           # real-model tests (needs model files under ~/.insightface)
```

### Accuracy Evaluation

`python -m evaluation` measures FAR/FRR, EER and the threshold at a target FAR for the
service's own pipeline on LFW or on your own labelled dataset. See
[docs/evaluation.md](docs/evaluation.md) for usage, data-handling rules and the
public LFW baseline.

## Extensibility

This architecture supports future enhancements:

1. **ONNX/TensorRT**: Replace InsightFace backend
2. **Batch Processing**: Add endpoint for multiple images
3. **Async Processing**: Queue-based processing for high volume
4. **Model Versioning**: A/B test different models
5. **Metrics**: Add Prometheus metrics export
6. **Liveness Detection**: Add anti-spoofing checks (not implemented here -- see DEPLOYMENT_GUIDE.md "Security notes")

## License

This project is provided as-is for integration into your own systems.

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
