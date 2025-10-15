"""Batch processing example."""

from perceptra_zero_shot import ZeroShotDetector
from pathlib import Path


def main():
    # Initialize detector
    detector = ZeroShotDetector("owlv2-base", device="cuda")
    
    # Get all images in a directory
    image_dir = Path("images")
    image_paths = list(image_dir.glob("*.jpg"))
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process batch
    results = detector.detect_batch(
        images=[str(p) for p in image_paths],
        prompts=["person", "vehicle", "animal"],
        confidence_threshold=0.4
    )
    
    # Analyze results
    total_detections = sum(len(r) for r in results)
    print(f"\nTotal detections across all images: {total_detections}")
    
    for i, (path, result) in enumerate(zip(image_paths, results)):
        print(f"\n{path.name}: {len(result)} objects")
        label_counts = {}
        for box in result.boxes:
            label_counts[box.label] = label_counts.get(box.label, 0) + 1
        for label, count in label_counts.items():
            print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()
