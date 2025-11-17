#!/usr/bin/env python3
"""Test script for the compare-photos API endpoint."""

import base64
import io
import json

import requests
from PIL import Image, ImageDraw


def create_test_image(color=(255, 200, 200), size=(200, 200)):
    """Create a simple test image."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw a simple face-like pattern
    center_x, center_y = size[0] // 2, size[1] // 2

    # Draw circle for head
    draw.ellipse(
        [center_x - 60, center_y - 60, center_x + 60, center_y + 60],
        fill=color,
        outline='black',
        width=3
    )

    # Draw eyes
    draw.ellipse(
        [center_x - 30, center_y - 20, center_x - 15, center_y - 5],
        fill='black'
    )
    draw.ellipse(
        [center_x + 15, center_y - 20, center_x + 30, center_y - 5],
        fill='black'
    )

    # Draw mouth
    draw.arc(
        [center_x - 25, center_y + 10, center_x + 25, center_y + 30],
        start=0,
        end=180,
        fill='black',
        width=3
    )

    return img


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_str


def test_compare_photos_endpoint():
    """Test the /api/v1/compare-photos endpoint."""

    # API endpoint
    url = "http://localhost:8000/api/v1/compare-photos"

    print("=" * 60)
    print("Testing /api/v1/compare-photos endpoint")
    print("=" * 60)

    # Test 1: Compare two similar images
    print("\n[Test 1] Comparing two similar images...")
    img1 = create_test_image(color=(255, 200, 200))
    img2 = create_test_image(color=(255, 205, 205))  # Slightly different color

    payload = {
        "image1": image_to_base64(img1),
        "image2": image_to_base64(img2),
        "distance_metric": "cosine"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success! Response:")
            print(json.dumps(result, indent=2))
            print(f"\nMatch: {result['match']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Metric: {result['distance_metric']}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server. Is it running on http://localhost:8000?")
        return
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return

    # Test 2: Test with euclidean metric
    print("\n" + "=" * 60)
    print("[Test 2] Testing with Euclidean distance metric...")

    payload["distance_metric"] = "euclidean"

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success! Response:")
            print(json.dumps(result, indent=2))
            print(f"\nMatch: {result['match']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Metric: {result['distance_metric']}")
        else:
            print(f"\n✗ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("Note: These are synthetic test images, not real faces.")
    print("The face detection may fail with these images.")
    print("For real testing, use actual face photos.")
    print("=" * 60)


def test_health_endpoint():
    """Test the health endpoint to verify server is running."""

    print("\n[Health Check] Checking if server is healthy...")

    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Server is {result['status']}")
            print(f"  Model loaded: {result['model_loaded']}")
            print(f"  Model name: {result.get('model_name', 'N/A')}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server at http://localhost:8000")
        return False
    except Exception as e:
        print(f"✗ Health check error: {str(e)}")
        return False


if __name__ == "__main__":
    # First check if server is healthy
    if test_health_endpoint():
        # Run the compare photos tests
        test_compare_photos_endpoint()
    else:
        print("\nPlease start the server first:")
        print("  python -m shelia_face_recognition_service.main")
