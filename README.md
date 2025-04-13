# Object Detection and Crowd Monitoring System

## Author
**P N Bhargav Teja**

## Description
A real-time object detection and crowd monitoring system built with Flask, OpenCV, and YOLO. This application provides a web interface for monitoring video streams, detecting objects, and counting people in real-time.

## Features
- Real-time object detection using YOLO v3
- Crowd counting and monitoring
- Web-based interface with live video streaming
- Secure HTTPS implementation
- User authentication and role-based access
- API endpoints for integration with other systems

## Prerequisites
- Python 3.9 or higher
- OpenCV
- TensorFlow
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/objectdetection-counting.git
cd objectdetection-counting
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO model files:
```bash
python download_models.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface:
- Local: https://127.0.0.1:3000
- Network: https://[your-ip]:3000

## Project Structure
```
├── app.py              # Main Flask application
├── main.py            # Core application logic
├── requirements.txt   # Project dependencies
├── users.json        # User data
├── static/           # Static assets
├── templates/        # HTML templates
├── ssl/             # SSL certificates
├── coco.names       # YOLO class names
├── yolov3.weights   # YOLO model weights
├── yolov3.cfg       # YOLO configuration
└── download_models.py # Model download script
```

## API Endpoints

### Authentication
- `POST /api/login` - User login
- `POST /api/register` - User registration

### Video Streams
- `GET /api/streams` - List all video streams
- `POST /api/streams` - Add new video stream
- `DELETE /api/streams/<id>` - Remove video stream

### Object Detection
- `GET /api/detection/<stream_id>` - Get detection results
- `GET /api/count/<stream_id>` - Get crowd count

## Security
- HTTPS encryption
- Password hashing
- Role-based access control
- Secure session management

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- YOLO v3 for object detection
- OpenCV for computer vision tasks
- Flask for web framework
- VidGear for video streaming

## Contact
P N Bhargav Teja
- GitHub: [Bhargavteja-9779](https://github.com/Bhargavteja-9779)
- Email: bhargavteja.pn15@gmail.com
