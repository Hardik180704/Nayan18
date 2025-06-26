# Enhanced Nayan: AI-Powered Navigation System for the Visually Impaired

## Project Overview
Enhanced Nayan is an advanced computer vision and machine learning-based system designed to assist visually impaired individuals in navigating their surroundings safely and independently. The system uses real-time object detection, depth estimation, and audio feedback to provide detailed information about the environment, including obstacles, hazards, text, currency, and human interactions. It leverages state-of-the-art models like YOLOv8 for object detection and MiDaS for depth estimation, combined with voice-based interaction to deliver a comprehensive assistive experience.

## Key Features
- **Object Detection**: Identifies objects in the environment using YOLOv8, including people, vehicles, and potential obstacles.
- **Depth Estimation**: Utilizes MiDaS to estimate distances to objects and detect spatial features like walls, stairs, and doorways.
- **Hazard Detection**: Recognizes potential dangers such as stairs, vehicles, and unmarked obstacles, providing immediate audio alerts.
- **Currency Detection**: Identifies Indian Rupee (INR) notes based on color and size for financial assistance.
- **Text Recognition**: Performs OCR to read text in the environment, aiding in understanding signs or labels.
- **Navigation Guidance**: Plans safe paths through obstacles using a sector-based approach and depth information.
- **Emotion Detection**: Detects facial emotions to provide social context (requires a pre-trained emotion model).
- **Voice Interaction**: Supports voice commands for scene description, object identification, navigation, and more.
- **Environmental Analysis**: Recognizes different environments (e.g., kitchen, street, park) and provides crowd density information.
- **Specialized Detections**: Identifies crosswalks, traffic signals, sidewalks, doorways, and human gestures for enhanced navigation.

## Requirements

### Hardware Requirements
- **Webcam**: A standard USB or built-in webcam for capturing real-time video.
- **Computer**: A system with at least 8GB RAM and a modern CPU (GPU recommended for faster processing).
- **Speakers/Headphones**: For audio feedback and alerts.
- **Microphone (Optional)**: For voice command interaction (not implemented in the provided code).

### Software Requirements
- **Python**: Version 3.8 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **Dependencies**: Install required Python packages using:
  ```bash
  pip install -r requirements.txt
  ```

### Required Python Packages
Create a `requirements.txt` file with the following:
```
opencv-python>=4.5.5
numpy>=1.21.0
pyttsx3>=2.90
ultralytics>=8.0.0
torch>=1.8.0
torchvision>=0.9.0
sounddevice>=0.4.4
soundfile>=0.10.3
requests>=2.26.0
pytesseract>=0.3.8
```
**Optional for Emotion Detection**:
```
tensorflow>=2.6.0
keras>=2.6.0
```

### Additional Dependencies
- **Tesseract OCR**: Required for text recognition. Install and add to system PATH:
  - Windows: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
- **YOLOv8 Model**: `yolov8n.pt` is automatically downloaded by Ultralytics.
- **MiDaS Model**: Automatically downloaded via `torch.hub`.
- **Emotion Detection Model**: A pre-trained `emotion_model.h5` file is required (not included).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Nayan.git
   cd Nayan
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR (see above).
4. Optional: Place `emotion_model.h5` in the project directory for emotion detection.

## Usage
1. Run the system:
   ```bash
   python app.py
   ```
2. The system initializes the webcam and provides:
   - Visual output (object detection and depth map).
   - Audio feedback for obstacles, hazards, and navigation.
   - Simulated voice commands (e.g., `describe`, `identify`, `read`).
3. Exit with the `ESC` key.

## Limitations
- Emotion detection requires a pre-trained `emotion_model.h5` (not included).
- Voice commands are simulated; actual speech recognition needs additional libraries.
- Currency detection is simplified; a specialized model is recommended.
- Depth estimation requires calibration for precision.

## Future Improvements
- Integrate real speech recognition.
- Enhance currency detection with a dedicated model.
- Add GPS for precise location tracking.
- Optimize for low-power devices.

## Contributing
Contributions are welcome! Fork the repository, create a branch, commit changes, and submit a pull request.

## License
MIT License. See the `LICENSE` file for details.

## Acknowledgments
- YOLOv8 by Ultralytics.
- MiDaS by Intel-ISL.
- PyTTSX3 for text-to-speech.
- Tesseract OCR for text recognition.