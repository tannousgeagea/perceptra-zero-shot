"""Data models for perceptra-zero-shot."""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Any
import numpy as np
from pathlib import Path


@dataclass
class Detection:
    """
    A single object detection result.
    
    Attributes:
        image_id: Identifier for the source image
        bbox: Bounding box as [x1, y1, x2, y2] in absolute coordinates
        label: Predicted label (from provided label set)
        confidence: Detection confidence score [0, 1]
        mask: Optional segmentation mask (H, W) boolean array
    """
    image_id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate detection data."""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 coordinates, got {len(self.bbox)}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.mask is not None and not isinstance(self.mask, np.ndarray):
            raise ValueError("mask must be a numpy array or None")


@dataclass
class DetectionResult:
    """
    Complete detection results for one or more images.
    
    Attributes:
        detections: List of all detections across all images
        metadata: Optional metadata about the detection run
    """
    detections: List[Detection]
    metadata: dict = field(default_factory=dict)
    
    def get_detections_for_image(self, image_id: str) -> List[Detection]:
        """Get all detections for a specific image."""
        return [d for d in self.detections if d.image_id == image_id]
    
    def get_detections_for_label(self, label: str) -> List[Detection]:
        """Get all detections for a specific label."""
        return [d for d in self.detections if d.label == label]
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """Return a new result with detections above confidence threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(detections=filtered, metadata=self.metadata)


# Type alias for image inputs
ImageInput = Union[str, Path, np.ndarray, Any]  # Any for PIL.Image
