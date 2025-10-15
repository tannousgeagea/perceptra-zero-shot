"""Compare different models on the same image."""

from perceptra_zero_shot import ZeroShotDetector, ModelRegistry
from PIL import Image
import time


def main():
    # Get available models
    models = ModelRegistry.list_models()
    print(f"Available models: {models}\n")
    
    # Load image
    image = Image.open("example_image.jpg")
    prompts = ["person", "car", "dog"]
    
    # Compare models
    results = {}
    for model_name in ["owlv2-base", "grounding-dino"]:
        if not ModelRegistry.is_registered(model_name):
            continue
            
        print(f"Testing {model_name}...")
        
        detector = ZeroShotDetector(model_name, device="cuda")
        
        start = time.time()
        result = detector.detect(image, prompts)
        elapsed = time.time() - start
        
        results[model_name] = {
            "detections": len(result),
            "time": elapsed,
            "result": result
        }
        
        print(f"  Detections: {len(result)}")
        print(f"  Time: {elapsed:.3f}s\n")
    
    # Summary
    print("\n=== Summary ===")
    for model_name, data in results.items():
        print(f"{model_name}:")
        print(f"  Total: {data['detections']} objects")
        print(f"  Speed: {data['time']:.3f}s")
        print(f"  FPS: {1/data['time']:.2f}")


if __name__ == "__main__":
    main()