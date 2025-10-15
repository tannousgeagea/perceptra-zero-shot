"""Example of using the REST API."""

import requests
from pathlib import Path


def detect_via_api():
    """Example of using the detection API."""
    url = "http://localhost:8000/detect"
    
    # Prepare the request
    with open("example_image.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "prompts": "person,car,dog",
            "model_name": "owlv2-base",
            "confidence_threshold": 0.3
        }
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['num_detections']} objects")
        for box in result['boxes']:
            print(f"  - {box['label']}: {box['confidence']:.3f}")
    else:
        print(f"Error: {response.status_code}")


def visualize_via_api():
    """Get visualization from API."""
    url = "http://localhost:8000/detect/visualize"
    
    with open("example_image.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "prompts": "person,car,dog",
            "model_name": "owlv2-base"
        }
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        with open("api_output.jpg", "wb") as f:
            f.write(response.content)
        print("Visualization saved to api_output.jpg")


def batch_detect_via_api():
    """Batch detection via API."""
    url = "http://localhost:8000/detect/batch"
    
    files = [
        ("files", open("image1.jpg", "rb")),
        ("files", open("image2.jpg", "rb")),
    ]
    data = {"prompts": "person,car"}
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['num_images']} images")


if __name__ == "__main__":
    print("1. Detecting objects...")
    detect_via_api()
    
    print("\n2. Getting visualization...")
    visualize_via_api()