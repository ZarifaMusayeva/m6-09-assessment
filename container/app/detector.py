#!/usr/bin/env python3
"""
ONNX-based YOLOv26 cat detector optimised for CPU inference.
Loads best.onnx and runs fast inference with letterboxing.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    """Load ONNX model and run CPU-optimised inference."""

    def __init__(self, onnx_path, imgsz=640, conf=0.25, class_names=("cat",)):
        """
        Args:
            onnx_path: Path to best.onnx (YOLOv26 end-to-end export)
            imgsz: Input image size (640)
            conf: Confidence threshold (0.0-1.0)
            class_names: Tuple of class names
        """
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"✓ Loaded YOLOv26 ONNX model: {onnx_path}")
        print(f"  Input shape: {self.session.get_inputs()[0].shape}")
        print(f"  Output shape: {self.session.get_outputs()[0].shape}")

    def _letterbox(self, img, size=640):
        """
        Resize image to (size, size) while preserving aspect ratio.
        Pad with gray (114, 114, 114) — YOLOv26 standard.
        """
        w, h = img.size
        
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))
        
        return canvas, scale, (pad_x, pad_y)

    def predict(self, image_path):
        """
        Run end-to-end inference on a single image.
        """
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        img_padded, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)

        img_array = np.array(img_padded, dtype=np.float32) / 255.0
        img_chw = img_array.transpose(2, 0, 1)  # HWC → CHW
        img_batch = img_chw[None, ...]  # Add batch dimension

        outputs = self.session.run(None, {self.input_name: img_batch})
        detections = outputs[0]  # Shape: (1, 300, 6)
        detections = detections[0]  # Shape: (300, 6)

        results = []
        for detection in detections:
            x1, y1, x2, y2, score, cls_id = detection

            if score < self.conf:
                continue

            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = max(0.0, min(orig_w, x1))
            y1 = max(0.0, min(orig_h, y1))
            x2 = max(0.0, min(orig_w, x2))
            y2 = max(0.0, min(orig_h, y2))

            results.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls_id)],
            })

        return results