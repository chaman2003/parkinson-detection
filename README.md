# 🧠 Parkinson's Detection PWA

A **Progressive Web Application** for early Parkinson's disease detection using AI-powered multimodal analysis of voice and tremor patterns.

## 🎯 Features

- **🎤 Voice Analysis**: Advanced audio processing using MFCC, spectral, and prosodic features
- **📱 Tremor Detection**: High-precision motion sensor analysis (3-12 Hz tremor band)
- **🤖 AI Models**: Ensemble machine learning (Random Forest + SVM + XGBoost)
- **📊 Real-time Quality Monitoring**: Live data quality indicators and metrics
- **🔄 PWA Capabilities**: Offline functionality, installable interface
- **🎛️ Test Modes**: Voice-only, tremor-only, or complete analysis options

## 🚀 Live Demo

- **Frontend**: [Deployed on Vercel](https://your-frontend-url.vercel.app)
- **Backend API**: [Deployed on Vercel](https://your-backend-url.vercel.app)

## 📊 Data Accuracy Standards

- **Audio**: 128kbps bitrate, 50ms precision, real-time SNR monitoring
- **Motion**: 100Hz sampling, 6-decimal precision, microsecond timestamps
- **Validation**: 80% quality threshold enforcement
- **Processing**: Research-validated parameter settings

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/chaman2003/parkinson-detection.git
   cd parkinson-detection
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   python app.py
   ```

3. **Frontend Setup**
   - Open `index.html` with Live Server in VS Code
   - Or serve with: `python -m http.server 8080`

## 🌐 API Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Main analysis endpoint
- `POST /api/demo` - Demo analysis with mock results
- `GET /api/models/info` - Model information

## 📄 License

This project is licensed under the MIT License.

**⚠️ Medical Disclaimer**: This application is for research and educational purposes only.

## Project Structure

```
parkinson/
├── frontend/                 # PWA Frontend
│   ├── index.html           # Main HTML file
│   ├── styles.css           # CSS styles
│   ├── app.js               # Main JavaScript logic
│   ├── manifest.json        # PWA manifest
│   ├── sw.js                # Service worker
│   └── assets/              # Images and icons
└── backend/                 # Python Backend
    ├── app.py               # Flask/FastAPI server
    ├── ml_models.py         # Machine learning pipeline
    ├── requirements.txt     # Python dependencies
    └── models/              # Trained model files
```

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python app.py`

### Frontend Setup
1. Open the frontend directory
2. Serve the files using a local web server
3. Access the app through HTTPS (required for PWA features)

## Usage

1. Open the PWA on your smartphone
2. Grant microphone and motion sensor permissions
3. Follow the instructions to record voice and tremor data
4. View the analysis results with confidence scores

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python, Flask/FastAPI
- **ML Libraries**: scikit-learn, XGBoost, NumPy, SciPy
- **Audio Processing**: Web Audio API, librosa (backend)
- **Motion Detection**: DeviceMotionEvent API