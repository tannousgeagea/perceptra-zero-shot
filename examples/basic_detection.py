"""Basic usage example for perceptra-zero-shot."""

from perceptra_zero_shot import ZeroShotDetector
from perceptra_zero_shot.utils import visualize_detections
from PIL import Image


def main():
    # Initialize detector
    print("Loading model...")
    detector = ZeroShotDetector(
        model_name="owlv2-base",
        device="cuda",  # or "cpu"
        confidence_threshold=0.3
    )
    
    # Load image
    image_path = "example_image.jpg"
    image = Image.open(image_path)
    print(f"Loaded image: {image.size}")
    
    # Define objects to detect
    prompts = ["person", "car", "dog", "cat", "bicycle"]
    
    # Perform detection
    print(f"Detecting objects: {prompts}")
    result = detector.detect(
        image=image,
        prompts=prompts,
        confidence_threshold=0.3,
        apply_nms=True
    )
    
    # Print results
    print(f"\nFound {len(result)} objects in {result.inference_time:.2f}s:")
    for i, box in enumerate(result.boxes, 1):
        print(f"  {i}. {box.label}: {box.confidence:.3f} at {box.to_xyxy()}")
    
    # Visualize results
    vis_image = visualize_detections(
        image=image,
        result=result,
        output_path="output_detection.jpg"
    )
    print("\nVisualization saved to output_detection.jpg")
    
    # Export results in different formats
    data = result.to_dict()
    coco_format = result.to_coco_format()
    print(f"\nExported {len(data['detections'])} detections")
    
    # Filter by confidence
    high_conf = result.filter_by_confidence(0.5)
    print(f"High confidence detections (>0.5): {len(high_conf)}")


if __name__ == "__main__":
    main()