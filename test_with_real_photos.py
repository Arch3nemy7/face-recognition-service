#!/usr/bin/env python3
"""
Test script for the compare-photos API endpoint using real face images.

This script demonstrates how to use the /api/v1/compare-photos endpoint
with local image files.

Usage:
    python test_with_real_photos.py <image1_path> <image2_path>

Example:
    python test_with_real_photos.py person1.jpg person2.jpg
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def encode_image_file(image_path):
    """Read an image file and encode it to base64."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return base64_str
    except FileNotFoundError:
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {image_path}: {str(e)}")
        sys.exit(1)


def compare_photos(image1_path, image2_path, metric="cosine", api_url="http://localhost:8000"):
    """Compare two photos using the face recognition API."""

    print("=" * 70)
    print("Face Recognition - Compare Two Photos")
    print("=" * 70)
    print(f"\nImage 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Distance Metric: {metric}")
    print(f"API URL: {api_url}")
    print("\n" + "-" * 70)

    # Encode images
    print("\nEncoding images to base64...")
    image1_b64 = encode_image_file(image1_path)
    image2_b64 = encode_image_file(image2_path)
    print(f"✓ Image 1 encoded ({len(image1_b64)} bytes)")
    print(f"✓ Image 2 encoded ({len(image2_b64)} bytes)")

    # Prepare request
    endpoint = f"{api_url}/api/v1/compare-photos"
    payload = {
        "image1": image1_b64,
        "image2": image2_b64,
        "distance_metric": metric
    }

    # Send request
    print(f"\nSending request to {endpoint}...")
    try:
        response = requests.post(endpoint, json=payload, timeout=60)

        print(f"Response Status: {response.status_code}")
        print("\n" + "=" * 70)

        if response.status_code == 200:
            result = response.json()

            print("SUCCESS!")
            print("=" * 70)
            print("\nComparison Results:")
            print("-" * 70)
            print(f"  Match:               {'YES ✓' if result['match'] else 'NO ✗'}")
            print(f"  Similarity Score:    {result['similarity']:.4f} (0-1, higher = more similar)")
            print(f"  Distance:            {result['distance']:.4f} (lower = more similar)")
            print(f"  Distance Metric:     {result['distance_metric']}")
            print("\nFace Detection Scores:")
            print("-" * 70)
            print(f"  Image 1:             {result['image1_detection_score']:.4f}")
            print(f"  Image 2:             {result['image2_detection_score']:.4f}")
            print("\n" + "=" * 70)

            # Interpretation
            print("\nInterpretation:")
            print("-" * 70)
            if result['match']:
                print("  The faces in the two images appear to be the SAME person.")
                print(f"  Confidence: {'High' if result['similarity'] > 0.8 else 'Moderate'}")
            else:
                print("  The faces in the two images appear to be DIFFERENT people.")
                print(f"  Confidence: {'High' if result['similarity'] < 0.5 else 'Moderate'}")

            print("\n" + "=" * 70)
            print("\nFull JSON Response:")
            print(json.dumps(result, indent=2))

        elif response.status_code == 400:
            error = response.json()
            print("ERROR - Bad Request")
            print("=" * 70)
            print(f"Error: {error.get('error', 'Unknown error')}")
            print(f"Code:  {error.get('error_code', 'N/A')}")
            if error.get('detail'):
                print(f"Detail: {error['detail']}")

            print("\nCommon issues:")
            print("  - No face detected in one or both images")
            print("  - Multiple faces detected in one image")
            print("  - Image format not supported")
            print("  - Image quality too low")

        else:
            print(f"ERROR - HTTP {response.status_code}")
            print("=" * 70)
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\nERROR - Connection Failed")
        print("=" * 70)
        print(f"Could not connect to API server at {api_url}")
        print("\nMake sure the server is running:")
        print("  python -m shelia_face_recognition_service.main")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("\nERROR - Request Timeout")
        print("=" * 70)
        print("The request took too long to complete.")
        print("This might happen with very large images.")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR - Unexpected Error")
        print("=" * 70)
        print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare two photos using face recognition API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_with_real_photos.py photo1.jpg photo2.jpg
  python test_with_real_photos.py person_a.png person_b.png --metric euclidean
  python test_with_real_photos.py img1.jpg img2.jpg --url http://192.168.1.100:8000

Note:
  Images should contain exactly ONE clear face each.
  Supported formats: JPEG, PNG, BMP, WebP
        """
    )

    parser.add_argument(
        "image1",
        help="Path to first image file"
    )
    parser.add_argument(
        "image2",
        help="Path to second image file"
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric to use (default: cosine)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    # Validate files exist
    if not Path(args.image1).exists():
        print(f"Error: Image file not found: {args.image1}")
        sys.exit(1)

    if not Path(args.image2).exists():
        print(f"Error: Image file not found: {args.image2}")
        sys.exit(1)

    # Run comparison
    compare_photos(args.image1, args.image2, args.metric, args.url)


if __name__ == "__main__":
    main()
