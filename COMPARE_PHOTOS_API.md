# Compare Photos API Endpoint

## Overview

The face recognition service provides **two endpoints** for comparing photos:

1. **`/api/v1/compare-photos`** - JSON-based endpoint for programmatic API usage (base64-encoded images)
2. **`/api/v1/compare-photos-upload`** - File upload endpoint for easy testing via Swagger UI (file picker)

Both endpoints provide the same functionality - they compare two photos to determine if they contain the same person.

## Quick Start for Testing

**Want to test the API quickly?**

1. Start the server: `python -m shelia_face_recognition_service.main`
2. Open your browser to: **http://localhost:8000/docs**
3. Find the **`POST /api/v1/compare-photos-upload`** endpoint
4. Click "Try it out"
5. Use the **file picker buttons** to select two photos
6. Click "Execute"
7. See the results!

---

## Endpoint 1: File Upload (Recommended for Testing)

### Details

- **URL**: `/api/v1/compare-photos-upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Use Case**: Testing via Swagger UI, quick manual testing

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image1` | file | Yes | - | First image file (use file picker in Swagger UI) |
| `image2` | file | Yes | - | Second image file (use file picker in Swagger UI) |
| `distance_metric` | string | No | "cosine" | Distance metric: "cosine" or "euclidean" |

### How to Use in Swagger UI

1. Navigate to **http://localhost:8000/docs**
2. Scroll to **Face Recognition** section
3. Click on **`POST /api/v1/compare-photos-upload`**
4. Click **"Try it out"**
5. Click **"Choose File"** for `image1` and select your first photo
6. Click **"Choose File"** for `image2` and select your second photo
7. (Optional) Change `distance_metric` if needed
8. Click **"Execute"**
9. View the response below!

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/compare-photos-upload?distance_metric=cosine" \
  -H "accept: application/json" \
  -F "image1=@/path/to/photo1.jpg" \
  -F "image2=@/path/to/photo2.jpg"
```

---

## Endpoint 2: JSON-based (Recommended for Programmatic Use)

### Endpoint Details

- **URL**: `/api/v1/compare-photos`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Schema

```json
{
  "image1": "base64-encoded-image-string",
  "image2": "base64-encoded-image-string",
  "distance_metric": "cosine"  // optional, default: "cosine", options: "cosine" | "euclidean"
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image1` | string | Yes | - | First base64-encoded image (with or without data URI prefix) |
| `image2` | string | Yes | - | Second base64-encoded image (with or without data URI prefix) |
| `distance_metric` | string | No | "cosine" | Distance metric to use: "cosine" or "euclidean" |

### Image Requirements

- **Format**: JPEG, PNG, BMP, WebP
- **Size**: Maximum 10MB per image
- **Dimensions**: Minimum 32x32, Maximum 4096x4096 pixels
- **Faces**: Each image must contain **exactly ONE** clear, frontal face
- **Quality**: Good lighting, minimal occlusion, clear facial features

## Response Schema

### Success Response (200 OK)

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

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `match` | boolean | Whether the faces match (based on default threshold) |
| `similarity` | float | Similarity score from 0-1 (higher = more similar) |
| `distance` | float | Distance value (lower = more similar) |
| `distance_metric` | string | The distance metric used for comparison |
| `image1_detection_score` | float | Face detection confidence for first image (0-1) |
| `image2_detection_score` | float | Face detection confidence for second image (0-1) |

### Match Thresholds

The `match` field uses these default thresholds:

- **Cosine distance**: `distance < 0.4` = match
- **Euclidean distance**: `distance < 1.0` = match

For custom thresholds, use the `similarity` or `distance` values directly.

### Error Response (400 Bad Request)

```json
{
  "error": "No face detected in the image",
  "detail": null,
  "error_code": "NO_FACE_DETECTED"
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `NO_FACE_DETECTED` | No face was detected in one or both images |
| `MULTIPLE_FACES_DETECTED` | Multiple faces detected in one image |
| `INVALID_IMAGE` | Image data is corrupted or invalid |
| `IMAGE_TOO_LARGE` | Image exceeds maximum size limit |
| `UNSUPPORTED_FORMAT` | Image format is not supported |
| `MODEL_NOT_LOADED` | Face recognition model failed to load |
| `PROCESSING_ERROR` | General processing error |

## Usage Examples

### cURL Example

```bash
# Prepare base64-encoded images
IMAGE1=$(base64 -w 0 photo1.jpg)
IMAGE2=$(base64 -w 0 photo2.jpg)

# Send request
curl -X POST http://localhost:8000/api/v1/compare-photos \
  -H "Content-Type: application/json" \
  -d "{
    \"image1\": \"$IMAGE1\",
    \"image2\": \"$IMAGE2\",
    \"distance_metric\": \"cosine\"
  }"
```

### Python Example

```python
import base64
import requests

# Read and encode images
with open('photo1.jpg', 'rb') as f:
    image1_b64 = base64.b64encode(f.read()).decode('utf-8')

with open('photo2.jpg', 'rb') as f:
    image2_b64 = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post(
    'http://localhost:8000/api/v1/compare-photos',
    json={
        'image1': image1_b64,
        'image2': image2_b64,
        'distance_metric': 'cosine'
    }
)

# Check result
if response.status_code == 200:
    result = response.json()
    print(f"Match: {result['match']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Distance: {result['distance']:.4f}")
else:
    print(f"Error: {response.json()}")
```

### JavaScript Example

```javascript
// Helper function to convert file to base64
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Compare two photos
async function comparePhotos(file1, file2) {
  const image1 = await fileToBase64(file1);
  const image2 = await fileToBase64(file2);

  const response = await fetch('http://localhost:8000/api/v1/compare-photos', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image1,
      image2,
      distance_metric: 'cosine'
    })
  });

  const result = await response.json();

  if (response.ok) {
    console.log('Match:', result.match);
    console.log('Similarity:', result.similarity);
    console.log('Distance:', result.distance);
  } else {
    console.error('Error:', result.error);
  }
}
```

### PHP Example (Laravel)

```php
use Illuminate\Support\Facades\Http;

// Read and encode images
$image1 = base64_encode(file_get_contents('photo1.jpg'));
$image2 = base64_encode(file_get_contents('photo2.jpg'));

// Send request
$response = Http::post('http://localhost:8000/api/v1/compare-photos', [
    'image1' => $image1,
    'image2' => $image2,
    'distance_metric' => 'cosine'
]);

// Check result
if ($response->successful()) {
    $result = $response->json();
    echo "Match: " . ($result['match'] ? 'Yes' : 'No') . "\n";
    echo "Similarity: " . $result['similarity'] . "\n";
    echo "Distance: " . $result['distance'] . "\n";
} else {
    $error = $response->json();
    echo "Error: " . $error['error'] . "\n";
}
```

## Understanding Similarity Scores

### Cosine Similarity

- **Range**: Distance 0-2, Similarity 0-1
- **Interpretation**:
  - Similarity > 0.8 = Very likely same person
  - Similarity 0.6-0.8 = Likely same person
  - Similarity 0.4-0.6 = Uncertain
  - Similarity < 0.4 = Different people

### Euclidean Distance

- **Range**: Distance 0-∞, Similarity 0-1
- **Interpretation**:
  - Distance < 0.8 = Very likely same person
  - Distance 0.8-1.2 = Likely same person
  - Distance 1.2-1.5 = Uncertain
  - Distance > 1.5 = Different people

## Best Practices

1. **Image Quality**: Use high-quality, well-lit photos for best results
2. **Face Position**: Ensure faces are frontal and clearly visible
3. **One Face Per Image**: Only one face should be present in each image
4. **Consistent Lighting**: Similar lighting conditions improve accuracy
5. **Error Handling**: Always handle potential errors (no face, multiple faces, etc.)
6. **Batch Processing**: For comparing one photo against multiple photos, consider using `/api/v1/embed` + `/api/v1/compare` for better performance
7. **Caching**: Cache embeddings for photos that will be compared multiple times

## Integration with Laravel Backend

This endpoint is designed to integrate seamlessly with a Laravel backend:

```php
// Laravel Controller Example
namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class FaceComparisonController extends Controller
{
    private $faceServiceUrl = 'http://localhost:8000';

    public function compare(Request $request)
    {
        $request->validate([
            'image1' => 'required|image|max:10240',
            'image2' => 'required|image|max:10240',
        ]);

        // Encode images
        $image1 = base64_encode(file_get_contents($request->file('image1')));
        $image2 = base64_encode(file_get_contents($request->file('image2')));

        // Call face recognition service
        $response = Http::timeout(60)->post("{$this->faceServiceUrl}/api/v1/compare-photos", [
            'image1' => $image1,
            'image2' => $image2,
            'distance_metric' => $request->input('metric', 'cosine')
        ]);

        if ($response->successful()) {
            return response()->json($response->json());
        }

        return response()->json([
            'error' => 'Face comparison failed',
            'details' => $response->json()
        ], $response->status());
    }
}
```

## Performance Considerations

- **Processing Time**: Typically 500ms - 2s per comparison (depends on image size and hardware)
- **Concurrency**: The service is stateless and can handle multiple concurrent requests
- **GPU Acceleration**: Use CUDA-enabled GPU for faster processing (set `DEVICE=cuda` in `.env`)
- **Rate Limiting**: Consider implementing rate limiting on your Laravel backend

## Testing

Use the provided test scripts:

```bash
# Test with your own photos
python test_with_real_photos.py photo1.jpg photo2.jpg

# Test with different metrics
python test_with_real_photos.py photo1.jpg photo2.jpg --metric euclidean

# Test with remote server
python test_with_real_photos.py photo1.jpg photo2.jpg --url http://192.168.1.100:8000
```

## API Documentation

Visit the interactive API documentation when the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
