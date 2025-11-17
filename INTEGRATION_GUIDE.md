# Integration Guide: Shelia Face Recognition Service

This guide shows how to integrate the face recognition microservice with your main attendance backend.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Integration Pattern A: Backend Does Matching](#integration-pattern-a-backend-does-matching-recommended)
- [Integration Pattern B: Service Does Matching](#integration-pattern-b-service-does-matching-optional)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Threshold Tuning](#threshold-tuning)
- [Error Handling](#error-handling)
- [Production Considerations](#production-considerations)

## Architecture Overview

```
┌────────────────────────────────────────────────┐
│              Main Backend                      │
│  ┌──────────────────────────────────────────┐  │
│  │ Database (PostgreSQL/MySQL/MongoDB)      │  │
│  │ ┌─────────────────────────────────────┐  │  │
│  │ │ users table/collection              │  │  │
│  │ │ ├─ user_id                          │  │  │
│  │ │ ├─ name                             │  │  │
│  │ │ ├─ face_embedding (512 floats)      │  │  │
│  │ │ └─ enrollment_date                  │  │  │
│  │ │                                     │  │  │
│  │ │ attendance table/collection         │  │  │
│  │ │ ├─ attendance_id                    │  │  │
│  │ │ ├─ user_id                          │  │  │
│  │ │ ├─ timestamp                        │  │  │
│  │ │ ├─ similarity_score                 │  │  │
│  │ │ └─ photo                            │  │  │
│  │ └─────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────┘  │
│                      │                          │
│                      │ HTTP API Calls           │
│                      ▼                          │
│  ┌──────────────────────────────────────────┐  │
│  │  Face Recognition Microservice          │  │
│  │  (Shelia Face Recognition Service)      │  │
│  │                                          │  │
│  │  • POST /api/v1/embed                   │  │
│  │  • POST /api/v1/compare                 │  │
│  │  • GET  /api/v1/health                  │  │
│  │                                          │  │
│  │  (Stateless - No Data Storage)          │  │
│  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────┘
```

## Integration Pattern A: Backend Does Matching (Recommended)

In this pattern, your backend stores all embeddings and only calls the face service for embedding extraction. Your backend performs the matching logic.

### Advantages
- Full control over matching logic
- Can implement complex business rules
- Better database query optimization
- Easier to add additional user filtering

### User Enrollment Flow

```python
# Python Example - User Enrollment Endpoint
from typing import Optional
import base64
import requests
from datetime import datetime

# Configuration
FACE_SERVICE_URL = "http://localhost:8000"

def enroll_user(user_id: str, name: str, photo_base64: str) -> dict:
    """
    Enroll a new user with their face photo.

    Args:
        user_id: Unique user identifier
        name: User's name
        photo_base64: Base64-encoded face photo

    Returns:
        Dictionary with enrollment result
    """
    try:
        # Step 1: Extract face embedding from photo
        response = requests.post(
            f"{FACE_SERVICE_URL}/api/v1/embed",
            json={"image": photo_base64},
            timeout=30
        )

        if response.status_code != 200:
            error = response.json()
            return {
                "success": False,
                "error": error.get("error", "Face extraction failed"),
                "error_code": error.get("error_code")
            }

        result = response.json()
        embedding = result["embedding"]
        detection_score = result.get("detection_score", 0.0)

        # Step 2: Validate embedding quality
        if detection_score < 0.8:
            return {
                "success": False,
                "error": "Face detection confidence too low. Please provide a clearer photo.",
                "detection_score": detection_score
            }

        # Step 3: Store user and embedding in database
        # (Example using SQLAlchemy - adapt to your ORM/database)
        from your_app.database import db_session
        from your_app.models import User

        user = User(
            id=user_id,
            name=name,
            face_embedding=embedding,  # Store as JSON or binary
            detection_score=detection_score,
            enrollment_date=datetime.utcnow()
        )

        db_session.add(user)
        db_session.commit()

        return {
            "success": True,
            "user_id": user_id,
            "detection_score": detection_score
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Face service unavailable: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Enrollment failed: {str(e)}"
        }


# JavaScript/Node.js Example
async function enrollUser(userId, name, photoBase64) {
    try {
        // Step 1: Extract face embedding
        const response = await fetch('http://localhost:8000/api/v1/embed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: photoBase64 }),
            timeout: 30000
        });

        if (!response.ok) {
            const error = await response.json();
            return {
                success: false,
                error: error.error,
                errorCode: error.error_code
            };
        }

        const result = await response.json();
        const embedding = result.embedding;
        const detectionScore = result.detection_score || 0.0;

        // Step 2: Validate quality
        if (detectionScore < 0.8) {
            return {
                success: false,
                error: 'Face detection confidence too low',
                detectionScore
            };
        }

        // Step 3: Store in database
        await db.users.create({
            id: userId,
            name,
            faceEmbedding: embedding,
            detectionScore,
            enrollmentDate: new Date()
        });

        return {
            success: true,
            userId,
            detectionScore
        };

    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}
```

### Attendance Check Flow

```python
# Python Example - Attendance Check
import numpy as np
from scipy.spatial.distance import cosine

def check_attendance(photo_base64: str, threshold: float = 0.4) -> dict:
    """
    Check attendance by matching face against enrolled users.

    Args:
        photo_base64: Base64-encoded photo of person
        threshold: Matching threshold (lower = stricter)

    Returns:
        Dictionary with attendance result
    """
    try:
        # Step 1: Extract embedding from live photo
        response = requests.post(
            f"{FACE_SERVICE_URL}/api/v1/embed",
            json={"image": photo_base64},
            timeout=30
        )

        if response.status_code != 200:
            error = response.json()
            return {
                "success": False,
                "error": error.get("error", "Face extraction failed"),
                "error_code": error.get("error_code")
            }

        result = response.json()
        live_embedding = np.array(result["embedding"])

        # Step 2: Fetch all enrolled users from database
        from your_app.database import db_session
        from your_app.models import User

        enrolled_users = db_session.query(User).all()

        if not enrolled_users:
            return {
                "success": False,
                "error": "No enrolled users found"
            }

        # Step 3: Compare with all enrolled users
        best_match = None
        min_distance = float('inf')

        for user in enrolled_users:
            stored_embedding = np.array(user.face_embedding)
            distance = cosine(live_embedding, stored_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match = user

        # Step 4: Validate against threshold
        if min_distance > threshold:
            return {
                "success": False,
                "error": "No matching user found",
                "best_distance": min_distance,
                "threshold": threshold
            }

        # Step 5: Record attendance
        from your_app.models import Attendance

        attendance = Attendance(
            user_id=best_match.id,
            timestamp=datetime.utcnow(),
            similarity_score=1.0 - min_distance,
            distance=min_distance
        )

        db_session.add(attendance)
        db_session.commit()

        return {
            "success": True,
            "user_id": best_match.id,
            "user_name": best_match.name,
            "similarity_score": 1.0 - min_distance,
            "distance": min_distance,
            "attendance_id": attendance.id
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Face service unavailable: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Attendance check failed: {str(e)}"
        }
```

## Integration Pattern B: Service Does Matching (Optional)

In this pattern, your backend sends reference embeddings to the service for comparison.

### Advantages
- Simpler backend logic
- Service handles distance calculations
- Consistent distance metrics

### Disadvantages
- More data transferred over network
- Less control over matching logic
- Harder to filter by user groups/departments

### Attendance Check Flow

```python
# Python Example - Using Compare Endpoint
def check_attendance_with_compare(photo_base64: str, threshold: float = 0.4) -> dict:
    """
    Check attendance using the compare endpoint.
    """
    try:
        # Step 1: Extract live embedding
        embed_response = requests.post(
            f"{FACE_SERVICE_URL}/api/v1/embed",
            json={"image": photo_base64},
            timeout=30
        )

        if embed_response.status_code != 200:
            error = embed_response.json()
            return {"success": False, "error": error.get("error")}

        live_embedding = embed_response.json()["embedding"]

        # Step 2: Fetch enrolled users
        from your_app.database import db_session
        from your_app.models import User

        enrolled_users = db_session.query(User).all()

        # Step 3: Prepare reference embeddings
        reference_embeddings = [
            {
                "id": user.id,
                "embedding": user.face_embedding
            }
            for user in enrolled_users
        ]

        # Step 4: Call compare endpoint
        compare_response = requests.post(
            f"{FACE_SERVICE_URL}/api/v1/compare",
            json={
                "query_embedding": live_embedding,
                "reference_embeddings": reference_embeddings,
                "distance_metric": "cosine"
            },
            timeout=30
        )

        if compare_response.status_code != 200:
            return {"success": False, "error": "Comparison failed"}

        result = compare_response.json()
        best_match = result["best_match"]

        # Step 5: Validate threshold
        if best_match["distance"] > threshold:
            return {
                "success": False,
                "error": "No matching user found",
                "best_distance": best_match["distance"]
            }

        # Step 6: Record attendance
        user = db_session.query(User).filter_by(id=best_match["id"]).first()

        from your_app.models import Attendance
        attendance = Attendance(
            user_id=user.id,
            timestamp=datetime.utcnow(),
            similarity_score=best_match["similarity"],
            distance=best_match["distance"]
        )

        db_session.add(attendance)
        db_session.commit()

        return {
            "success": True,
            "user_id": user.id,
            "user_name": user.name,
            "similarity_score": best_match["similarity"],
            "distance": best_match["distance"]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Complete Implementation Examples

### Flask Backend Example

```python
# app.py - Complete Flask integration
from flask import Flask, request, jsonify
import base64
import requests
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)
FACE_SERVICE_URL = "http://localhost:8000"

# In-memory storage (replace with real database)
users_db = {}
attendance_db = []

@app.route('/api/enroll', methods=['POST'])
def enroll():
    """Enroll a new user."""
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    photo = data.get('photo')  # base64

    # Call face service
    response = requests.post(
        f"{FACE_SERVICE_URL}/api/v1/embed",
        json={"image": photo}
    )

    if response.status_code != 200:
        return jsonify(response.json()), response.status_code

    embedding = response.json()["embedding"]

    # Store user
    users_db[user_id] = {
        "name": name,
        "embedding": embedding
    }

    return jsonify({
        "success": True,
        "user_id": user_id,
        "message": "User enrolled successfully"
    })

@app.route('/api/check-attendance', methods=['POST'])
def check_attendance():
    """Check attendance for a person."""
    data = request.json
    photo = data.get('photo')
    threshold = data.get('threshold', 0.4)

    # Extract embedding
    response = requests.post(
        f"{FACE_SERVICE_URL}/api/v1/embed",
        json={"image": photo}
    )

    if response.status_code != 200:
        return jsonify(response.json()), response.status_code

    live_embedding = np.array(response.json()["embedding"])

    # Find best match
    best_match_id = None
    min_distance = float('inf')

    for user_id, user_data in users_db.items():
        stored_embedding = np.array(user_data["embedding"])
        distance = cosine(live_embedding, stored_embedding)

        if distance < min_distance:
            min_distance = distance
            best_match_id = user_id

    # Check threshold
    if min_distance > threshold:
        return jsonify({
            "success": False,
            "error": "No match found",
            "best_distance": float(min_distance)
        }), 404

    # Record attendance
    attendance_db.append({
        "user_id": best_match_id,
        "user_name": users_db[best_match_id]["name"],
        "timestamp": datetime.utcnow().isoformat(),
        "distance": float(min_distance)
    })

    return jsonify({
        "success": True,
        "user_id": best_match_id,
        "user_name": users_db[best_match_id]["name"],
        "distance": float(min_distance),
        "similarity": 1.0 - min_distance
    })

if __name__ == '__main__':
    app.run(port=5000)
```

## Threshold Tuning

Choosing the right threshold is critical for balancing false accepts vs. false rejects.

### Recommended Approach

1. **Collect Test Data**: Gather genuine (same person) and impostor (different person) pairs
2. **Calculate Distances**: For each pair, compute the cosine distance
3. **Analyze Distribution**: Plot histograms of genuine vs. impostor distances
4. **Choose Threshold**: Select threshold that minimizes error based on your requirements

### Typical Thresholds (Cosine Distance)

- **High Security** (low false accept rate): 0.2 - 0.3
- **Balanced**: 0.3 - 0.5
- **High Convenience** (low false reject rate): 0.5 - 0.7

### Example Threshold Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

def analyze_thresholds(genuine_distances, impostor_distances):
    """
    Analyze different thresholds and compute error rates.
    """
    thresholds = np.linspace(0, 1, 100)
    far_rates = []  # False Accept Rate
    frr_rates = []  # False Reject Rate

    for threshold in thresholds:
        # False accepts: impostors accepted (distance < threshold)
        false_accepts = sum(1 for d in impostor_distances if d < threshold)
        far = false_accepts / len(impostor_distances)

        # False rejects: genuines rejected (distance >= threshold)
        false_rejects = sum(1 for d in genuine_distances if d >= threshold)
        frr = false_rejects / len(genuine_distances)

        far_rates.append(far)
        frr_rates.append(frr)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, far_rates, label='False Accept Rate (FAR)')
    plt.plot(thresholds, frr_rates, label='False Reject Rate (FRR)')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.title('Threshold Analysis')
    plt.show()

    # Find Equal Error Rate (EER)
    eer_idx = np.argmin(np.abs(np.array(far_rates) - np.array(frr_rates)))
    eer_threshold = thresholds[eer_idx]
    eer_rate = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2

    print(f"Equal Error Rate (EER) Threshold: {eer_threshold:.3f}")
    print(f"EER: {eer_rate * 100:.2f}%")

    return eer_threshold
```

## Error Handling

### Handling Service Unavailability

```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry on service failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
            raise last_exception
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def call_face_service(endpoint, data):
    """Call face service with retry logic."""
    response = requests.post(
        f"{FACE_SERVICE_URL}{endpoint}",
        json=data,
        timeout=30
    )
    response.raise_for_status()
    return response.json()
```

## Production Considerations

### 1. Add Authentication

```python
# Add API key to all requests
FACE_SERVICE_API_KEY = "your-secure-api-key"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": FACE_SERVICE_API_KEY
}

response = requests.post(
    f"{FACE_SERVICE_URL}/api/v1/embed",
    json={"image": photo_base64},
    headers=headers
)
```

### 2. Connection Pooling

```python
# Use a session for connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Use session for all requests
response = session.post(f"{FACE_SERVICE_URL}/api/v1/embed", json=data)
```

### 3. Async Processing for High Volume

```python
# Use asyncio for concurrent processing
import asyncio
import aiohttp

async def process_attendance_async(photos):
    async with aiohttp.ClientSession() as session:
        tasks = [
            check_attendance_async(session, photo)
            for photo in photos
        ]
        return await asyncio.gather(*tasks)

async def check_attendance_async(session, photo):
    async with session.post(
        f"{FACE_SERVICE_URL}/api/v1/embed",
        json={"image": photo}
    ) as response:
        return await response.json()
```

### 4. Monitoring

```python
# Log all face service calls
import logging

logger = logging.getLogger(__name__)

def call_face_service_with_logging(endpoint, data):
    start_time = time.time()
    try:
        response = requests.post(
            f"{FACE_SERVICE_URL}{endpoint}",
            json=data,
            timeout=30
        )
        duration = time.time() - start_time

        logger.info(f"Face service call: {endpoint}, "
                   f"status: {response.status_code}, "
                   f"duration: {duration:.2f}s")

        return response
    except Exception as e:
        logger.error(f"Face service error: {endpoint}, error: {str(e)}")
        raise
```

---

This integration guide should help you successfully integrate the face recognition microservice with your attendance system. Choose the pattern that best fits your requirements and scale accordingly.
