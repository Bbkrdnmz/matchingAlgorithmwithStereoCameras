# Stereo Camera Plank Detection and Matching System

A sophisticated computer vision system that detects, tracks, and matches planks using a dual-camera (stereo) setup with real-time object tracking and stereo matching capabilities.

## 🎯 Features

- **Dual Camera Support**: Simultaneous image capture from left and right cameras
- **YOLOv5 Integration**: High-accuracy plank detection using custom trained models
- **Camera Calibration**: Lens distortion correction and stereo calibration support
- **DeepSORT Tracking**: Persistent object tracking with stable IDs during movement
- **Stereo Matching**: Automatic matching of identical planks between left and right cameras
- **Real-time Visualization**: Display with confidence scores and unique IDs
- **Counter System**: Real-time tracking of detected plank count

## 🛠️ Technologies

- **OpenCV**: Image processing and camera management
- **PyTorch**: YOLOv5 model loading and inference
- **DeepSORT**: Object tracking and ID management
- **NumPy**: Numerical computations
- **Python 3.7+**: Main programming language

## 📋 Requirements

### Hardware
- 2x USB cameras (preferably same model)
- Windows 11 operating system
- Minimum 12GB RAM
- CUDA-compatible GPU (optional, for acceleration)

### Software
```bash
pip install opencv-python
pip install torch torchvision
pip install numpy
pip install deep-sort-realtime
pip install ultralytics
```

## 📁 File Structure

```
MachingAlg/
├── matching.py                    # Main stereo matching script
├── best.pt                       # Trained YOLOv5 model
├── stereo_calibration_data.xml   # Camera calibration file
└── yolov5/                       # YOLOv5 source code (ultralytics/yolov5)
```

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd MachingAlg
```

2. **Download YOLOv5:**
```bash
git clone https://github.com/ultralytics/yolov5.git
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Perform camera calibration:**
   - Run `stereoCalibration1.py` script
   - Capture 30-50 images with chessboard
   - Ensure `stereo_calibration_data.xml` is created

## 🚀 Usage

### Basic Usage
```bash
python matching.py
```

### Test Camera IDs
```bash
python test.py
```

### Perform Calibration
```bash
python stereoCalibration1.py
```

## 🔧 Configuration

### Camera Settings
```python
# Camera IDs (check with test.py)
cap_left = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap_right = cv2.VideoCapture(1, cv2.CAP_MSMF)

# Resolution settings
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Model Path
```python
# Specify your model path here
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'C:\Users\donme\Desktop\MachingAlg\best.pt', force_reload=True)
```

### Calibration File
```python
# Calibration file path
fs = cv2.FileStorage(r'C:\Users\donme\Desktop\MachingAlg\stereo_calibration_data.xml', cv2.FILE_STORAGE_READ)
```

## 🎮 Controls

- **'q' key**: Exit program
- **ESC key**: Exit program

## 📊 Output Format

### Left Camera
- Green bounding boxes
- Confidence scores (top)
- Object IDs (bottom)
- Total plank count (top-left corner)

### Right Camera
- Green bounding boxes
- Confidence scores (top)
- Matched IDs (bottom)
- Unmatched objects marked with "-"

## 🔍 Algorithm Details

### 1. Image Preprocessing
- Lens distortion correction using camera calibration
- Object detection with YOLOv5
- Confidence threshold (default: 0.5)

### 2. Object Tracking
- ID assignment using DeepSORT algorithm on left camera
- `max_age=100` parameter for long-term tracking
- Centroid-based matching

### 3. Stereo Matching
- IoU (Intersection over Union) calculation
- Centroid distance calculation
- Greedy algorithm for best match selection
- One-to-one mapping (each ID used only once)

## 🐛 Troubleshooting

### Camera Not Opening
```bash
# Check camera IDs
python test.py
```

### Model Not Loading
- Ensure `best.pt` file is in correct location
- Check PyTorch and YOLOv5 installation

### Calibration Error
- Check if `stereo_calibration_data.xml` exists
- Repeat calibration process

### ID Instability
- Adjust DeepSORT parameters (`max_age`, `max_iou_distance`)
- Check camera calibration accuracy

## ⚡ Performance Optimization

### GPU Usage
```python
# Enable CUDA usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### FPS Improvement
- Increase confidence threshold (0.7-0.8)
- Reduce image resolution
- Optimize DeepSORT parameters

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👨‍💻 Developer

**Ebubekir** - *Internship Project* - [GitHub Profile](https://github.com/username)

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/yolov5) - YOLOv5 implementation
- [DeepSORT](https://github.com/nwojke/deep_sort) - Object tracking algorithm
- [OpenCV](https://opencv.org/) - Computer vision library

---

**Note**: This project was developed as part of an internship program and is continuously being improved. Feel free to open an issue for any problems or suggestions.

---

## 📖 Description

This project implements a real-time stereo vision system for plank detection and tracking in industrial environments. The system uses two synchronized USB cameras to capture stereo images, applies camera calibration to correct lens distortions, and employs YOLOv5 for object detection. The key innovation lies in the stereo matching algorithm that ensures the same plank detected in the left camera is correctly identified and matched in the right camera, preventing confusion between different planks.

The system features persistent object tracking using DeepSORT, which maintains stable IDs even when planks move or are temporarily occluded. The stereo matching process uses both IoU (Intersection over Union) and centroid distance calculations to achieve accurate one-to-one mapping between cameras. Real-time visualization displays confidence scores, unique IDs, and a running count of detected planks.

This solution is particularly useful for quality control, inventory management, and automated sorting systems in wood processing facilities where accurate plank identification and tracking are critical for operational efficiency.
