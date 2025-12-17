🌱 Early Plant Disease Detection System (Raspberry Pi + AI)
Overview

This project aims to detect plant diseases at an early stage using computer vision and deep learning, enabling timely intervention before the disease spreads across the crop.
The system is designed to run on resource-constrained hardware (Raspberry Pi) and operates fully autonomously, capturing images at regular intervals and logging predictions without human intervention.

System Architecture

The solution integrates:
-> Raspberry Pi – edge device for deployment
-> Camera Module – periodic image capture
-> Deep Learning Model – disease classification
-> Automation (cron + shell scripts) – scheduled execution

AI Model

-> Trained using an ensemble learning approach
-> Base architectures: MobileNet, EfficientNetV2
-> Ensemble improves robustness and accuracy compared to a single model
-> Final model exported to ONNX format (.onnx) for:
    -> Lightweight inference
    -> Cross-framework compatibility
    -> Efficient execution on Raspberry Pi using ONNX Runtime

Image Acquisition

-> Camera attached to Raspberry Pi
-> Images captured using the rpicam command
-> One image captured every 30 minutes

Execution Environment

Since the Python script depends on multiple external libraries that may conflict with the system Python environment, a Python virtual environment is used.
This ensures:
-> Dependency isolation
-> Reproducibility
-> Stable execution on Raspberry Pi

Automation Workflow

The entire pipeline is automated using cron jobs and a shell script.
run.sh responsibilities:
-> Navigate to the project directory
-> Delete the previously captured image
-> Capture a new image using the Raspberry Pi camera
-> Activate the Python virtual environment
-> Execute the inference script (run.py)
-> Log the top 3 disease predictions to logs.txt

Scheduling

-> Cron job triggers run.sh every 30 minutes
-> System runs continuously without manual intervention

Output

-> Predictions appended to logs.txt
-> Each entry contains:
    -> Timestamp
    -> Top 3 predicted disease classes
    -> Confidence scores
-> Logs can be used for monitoring disease progression, analytics, and alerts

Key Advantages

-> Edge-based inference (no internet dependency)
-> Low-cost and scalable
-> Early disease detection
-> Automated and unattended operation
-> Optimized for embedded hardware

Future Improvements

-> Real-time alert system (SMS / App / Dashboard)
-> Severity estimation
-> GPS-tagged disease mapping
-> Cloud sync for long-term analytics



Limitations:
- Model accuracy depends on lighting and image quality
- Fixed image capture interval (30 minutes)
- No real-time alerting in current version
- Designed for single-camera deployment

1. Cron triggers run.sh every 30 minutes
2. run.sh navigates to the project directory
3. Old image is deleted
4. New image is captured using rpicam
5. Python virtual environment is activated
6. run.py loads the ONNX model
7. Image is preprocessed and passed to the model
8. Top 3 predictions are generated
9. Results are appended to logs.txt

