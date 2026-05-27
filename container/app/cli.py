#!/usr/bin/env python3
"""
Zarifa Musayeva — YOLOv26 Cat Detection Container CLI.

Supports two subcommands:
  - info: Print STUDENT.json metadata
  - predict: Run inference on /data/input, write CSV to /data/output
"""

import sys
import json
import csv
from pathlib import Path
from app.detector import CatDetector


def cmd_info():
    """
    Print STUDENT.json to stdout (JSON format).
    Used by leaderboard runner to register entry.
    """
    try:
        # Cari faylın yerləşdiyi qovluğa əsasən STUDENT.json yolunu tapırıq
        current_dir = Path(__file__).resolve().parent  # /app/app
        root_dir = current_dir.parent                  # /app
        json_path = root_dir / "STUDENT.json"
        
        if not json_path.exists():
            print(f"ERROR: STUDENT.json not found at {json_path}")
            return 1

        with open(json_path, "r", encoding="utf-8") as f:
            student_data = json.load(f)
        
        # Məlumatı ekrana çıxarırıq və dərhal flush edirik
        print(json.dumps(student_data, indent=2))
        sys.stdout.flush()
        return 0
    
    except Exception as e:
        print(f"ERROR inside cmd_info: {e}")
        sys.stdout.flush()
        return 1


def cmd_predict():
    """
    Run YOLOv26 ONNX inference on all images in /data/input.
    Write predictions.csv to /data/output with standard schema.
    """
    try:
        # Setup paths
        input_dir = Path("/data/input")
        output_dir = Path("/data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"

        # Initialize detector (CPU-optimised)
        print("Loading YOLOv26 ONNX model...")
        sys.stdout.flush()
        
        detector = CatDetector(
            onnx_path="/app/models/best.onnx",
            imgsz=640,
            conf=0.25,  # Confidence threshold
            class_names=("cat",)
        )
        print("Model ready.\n")
        sys.stdout.flush()

        # Open CSV for writing
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header (required schema)
            writer.writerow([
                "image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"
            ])

            # Find all images (sorted, preserving subdirs)
            image_files = sorted(input_dir.rglob("*"))
            image_count = 0

            for img_file in image_files:
                # Skip directories
                if not img_file.is_file():
                    continue
                
                # Only process image extensions
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Get relative path (preserve subdirs, use forward slashes)
                try:
                    rel_path = img_file.relative_to(input_dir).as_posix()
                except ValueError:
                    continue

                image_count += 1
                print(f"  [{image_count}] {rel_path}...", end=" ", flush=True)

                try:
                    # Run inference
                    boxes = detector.predict(str(img_file))

                    if len(boxes) == 0:
                        # No detections: write one empty row
                        writer.writerow([rel_path, "", "", "", "", "", ""])
                        print("0 boxes")
                        sys.stdout.flush()
                    else:
                        # Write one row per detection
                        for box in boxes:
                            writer.writerow([
                                rel_path,
                                box["xmin"],
                                box["ymin"],
                                box["xmax"],
                                box["ymax"],
                                box["confidence"],
                                box["class"]
                            ])
                        print(f"{len(boxes)} boxes")
                        sys.stdout.flush()

                except Exception as e:
                    print(f"ERROR processing image {rel_path}: {e}")
                    sys.stdout.flush()
                    writer.writerow([rel_path, "", "", "", "", "", ""])

        # Success
        print(f"\n✓ Processed {image_count} images")
        print(f"✓ Predictions written to {csv_path}")
        sys.stdout.flush()
        return 0

    except Exception as e:
        print(f"ERROR in predict: {e}")
        sys.stdout.flush()
        return 1


def main():
    """Parse command-line arguments and dispatch."""
    if len(sys.argv) < 2:
        print("Usage: cli.py [info|predict]")
        sys.stdout.flush()
        return 1

    command = sys.argv[1]

    if command == "info":
        return cmd_info()
    elif command == "predict":
        return cmd_predict()
    else:
        print(f"ERROR: Unknown command '{command}'")
        print("Usage: cli.py [info|predict]")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())