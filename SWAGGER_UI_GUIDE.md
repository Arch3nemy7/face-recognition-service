# Swagger UI Testing Guide

## How to Test Face Comparison with File Upload

This guide shows you how to easily test the face recognition API using the Swagger UI interface with file upload buttons.

---

## Step-by-Step Instructions

### 1. Start the Server

```bash
# Activate virtual environment (if needed)
source .venv/bin/activate

# Start the server
python -m shelia_face_recognition_service.main
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### 2. Open Swagger UI

Open your web browser and navigate to:

```
http://localhost:8000/docs
```

You will see the interactive API documentation (Swagger UI).

---

### 3. Find the File Upload Endpoint

In the Swagger UI page:

1. Scroll down to the **"Face Recognition"** section
2. Look for the endpoint: **`POST /api/v1/compare-photos-upload`**
3. You'll see the description: *"Compare two photos using file upload (convenient for testing in Swagger UI)"*

---

### 4. Click "Try it out"

1. Click on the **`POST /api/v1/compare-photos-upload`** row to expand it
2. On the right side, click the **"Try it out"** button
3. The form will become editable

---

### 5. Upload Your Photos

You will see the form with these fields:

**image1** *(required)*
- Click the **"Choose File"** button
- Select your first photo from your computer
- Supported formats: JPEG, PNG, BMP, WebP

**image2** *(required)*
- Click the **"Choose File"** button
- Select your second photo from your computer
- Make sure each photo contains exactly ONE clear face

**distance_metric** *(optional)*
- Default: `cosine`
- Options: `cosine` or `euclidean`
- You can leave it as default or change it

---

### 6. Execute the Request

1. Click the blue **"Execute"** button at the bottom
2. Wait a few seconds for processing (typically 1-3 seconds)
3. Scroll down to see the response

---

### 7. View the Results

You'll see the response in the **"Response body"** section:

```json
{
  "match": true,
  "similarity": 0.8523,
  "distance": 0.1234,
  "distance_metric": "cosine",
  "image1_detection_score": 0.9876,
  "image2_detection_score": 0.9654
}
```

**What do these values mean?**

- **`match`**: `true` = same person, `false` = different people
- **`similarity`**: 0-1 score (higher = more similar)
  - > 0.8 = Very likely same person
  - 0.6-0.8 = Likely same person
  - < 0.6 = Different people
- **`distance`**: Lower values = more similar
- **`image1_detection_score`** & **`image2_detection_score`**: Face detection confidence (0-1)

---

## Example: Comparing Two Photos

### Success Case

**Request:**
- image1: `john_photo1.jpg` (clear frontal face)
- image2: `john_photo2.jpg` (same person, different angle)
- distance_metric: `cosine`

**Response:**
```json
{
  "match": true,
  "similarity": 0.8934,
  "distance": 0.1066,
  "distance_metric": "cosine",
  "image1_detection_score": 0.9912,
  "image2_detection_score": 0.9876
}
```

**Interpretation:** ✅ Same person (high similarity, low distance)

---

### Different People

**Request:**
- image1: `person_a.jpg`
- image2: `person_b.jpg`
- distance_metric: `cosine`

**Response:**
```json
{
  "match": false,
  "similarity": 0.3421,
  "distance": 0.6579,
  "distance_metric": "cosine",
  "image1_detection_score": 0.9845,
  "image2_detection_score": 0.9801
}
```

**Interpretation:** ❌ Different people (low similarity, high distance)

---

## Common Errors and Solutions

### Error: "No face detected in the image"

**Cause:** The image doesn't contain a detectable face, or the face is too small/unclear

**Solution:**
- Use a photo with a clear, frontal face
- Ensure good lighting
- Face should be at least 100x100 pixels
- Avoid heavily cropped or blurry images

---

### Error: "Multiple faces detected in one image"

**Cause:** The image contains more than one face

**Solution:**
- Crop the image to show only one person
- Use a photo with a single subject

---

### Error: "Image too large"

**Cause:** Image file exceeds 10MB

**Solution:**
- Resize or compress the image
- Convert to JPEG with lower quality setting

---

## Testing with Different Metrics

### Cosine Distance (Default)

Best for general face comparison. Range: 0-2

```
distance_metric: cosine
```

**Thresholds:**
- distance < 0.4 = match
- distance > 0.6 = no match

---

### Euclidean Distance

Alternative metric. Range: 0-∞

```
distance_metric: euclidean
```

**Thresholds:**
- distance < 1.0 = match
- distance > 1.5 = no match

---

## Comparison: File Upload vs JSON Endpoint

| Feature | `/compare-photos-upload` | `/compare-photos` |
|---------|-------------------------|-------------------|
| Input Method | File picker | Base64 string |
| Best For | Manual testing | Programmatic API |
| Content-Type | multipart/form-data | application/json |
| Swagger UI | Easy to use ✅ | Requires base64 encoding |
| cURL | `-F` flag | JSON payload |
| Laravel/PHP | Use file upload | Use base64 encoding |

---

## Next Steps

### For Manual Testing
- Use **`/api/v1/compare-photos-upload`** with the Swagger UI
- Perfect for testing and validation

### For Production Integration
- Use **`/api/v1/compare-photos`** with base64-encoded images
- Better for API clients and backend integration
- See [COMPARE_PHOTOS_API.md](COMPARE_PHOTOS_API.md) for code examples

---

## Troubleshooting

### Server Not Starting

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill <PID>

# Or use a different port
uvicorn shelia_face_recognition_service.main:app --port 8001
```

### Cannot Access Swagger UI

1. Make sure server is running
2. Check the URL: `http://localhost:8000/docs`
3. Try `http://127.0.0.1:8000/docs`
4. Check firewall settings

### Face Detection Issues

- Use high-quality photos with clear faces
- Ensure good lighting
- Face should be frontal (not profile view)
- Minimum face size: 100x100 pixels
- No sunglasses or heavy occlusion

---

## Additional Resources

- **Full API Documentation**: [COMPARE_PHOTOS_API.md](COMPARE_PHOTOS_API.md)
- **Integration Guide**: See README.md
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Alternative UI**: http://localhost:8000/redoc
