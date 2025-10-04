# Parkinson's Detection System

AI-powered Parkinson's disease detection using voice analysis and tremor detection through multimodal machine learning.

## Quick Start

### Backend Setup

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
python train.py

# Run server
python app.py
```

Server will run on `http://localhost:5000`

### Frontend Setup

Simply open `frontend/index.html` in a web browser or use a local server:

```bash
cd frontend
python -m http.server 8000
```

Then navigate to `http://localhost:8000`

## Project Structure

```
parkinson/
├── backend/
│   ├── app.py                 # Main Flask server (run this)
│   ├── train.py               # Model training (run once)
│   ├── requirements.txt       # Python dependencies
│   ├── README.md             # Backend documentation
│   ├── utils/                # ML utilities
│   │   ├── audio_features.py     # Voice feature extraction
│   │   ├── tremor_features.py    # Motion feature extraction
│   │   ├── ml_models.py          # ML pipeline
│   │   ├── data_loader.py        # Dataset loading
│   │   ├── data_storage.py       # Results storage
│   │   └── dataset_matcher.py    # Dataset matching
│   ├── datasets/             # Training data
│   │   ├── tremor_simplified.csv      # 4,151 tremor samples
│   │   ├── voice_labels.csv           # 40 voice files
│   │   ├── frontend_params.json       # Parameter mapping
│   │   └── voice_dataset/             # Voice audio files
│   ├── models/               # Trained models (.pkl files)
│   ├── uploads/              # Temporary file uploads
│   └── recorded_data/        # Test recordings storage
│
├── frontend/
│   ├── index.html            # Main entry point
│   ├── js/                   # JavaScript modules
│   │   ├── app.js                # Main application logic
│   │   ├── excel-export.js       # Results export functionality
│   │   └── sw.js                 # Service worker
│   ├── css/                  # Stylesheets
│   │   ├── styles.css            # Main styles
│   │   └── quality-indicators.css # Quality indicator styles
│   ├── assets/               # Icons and images
│   ├── manifest.json         # PWA manifest
│   ├── sensor-test.html      # Sensor debugging tool
│   ├── sensor-test.js        # Sensor test logic
│   └── proxy.py              # Proxy server for ngrok
│
└── README.md                 # This file
```

## Features

### Voice Analysis
- Extracts 138 audio features (MFCC, Spectral, Prosodic, Voice Quality)
- Analyzes speech patterns and vocal characteristics
- Compares against known Parkinson's voice signatures

### Tremor Detection
- Extracts 25 motion features from device sensors
- Analyzes acceleration and rotation patterns
- Detects tremor frequency and amplitude

### Combined Analysis
- Ensemble ML models (SVM + RandomForest + GradientBoosting)
- Weighted prediction combining voice and tremor analysis
- Dataset matching for validation

## API Endpoints

### Backend API

- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze voice and/or tremor data
- `POST /api/analyze-stream` - Streaming analysis with progress
- `GET /api/models/info` - Model information
- `GET /api/storage/stats` - Storage statistics

## Data Flow

```
Frontend → Backend → ML Pipeline → Results → Storage
   ↓                      ↓
WebM Audio          138 Voice Features
Motion Data         25 Tremor Features
```

### Frontend Input
- **Voice**: WebM audio file (10 seconds recording)
- **Tremor**: JSON array of motion samples
  - `x`, `y`, `z` (acceleration in m/s²)
  - `timestamp` (performance.now() in milliseconds)

### Backend Processing
- **Voice**: WebM → WAV → 138 features
- **Tremor**: Motion data → 25 features
- **ML**: Ensemble models → Prediction + Confidence

## Models

### Training Data
- **Tremor**: 4,151 samples (2,051 healthy, 2,100 affected)
- **Voice**: 40 Parkinson's-affected samples

### ML Architecture
- Ensemble Voting Classifier
  - Support Vector Machine (SVM)
  - Random Forest (100 estimators)
  - Gradient Boosting (100 estimators)
- StandardScaler for feature normalization

### Performance
- Tremor Model: ~64% accuracy
- Combined Analysis: Weighted voice + tremor predictions

## Development

### Retrain Models

```bash
cd backend
python train.py
```

Training takes ~90 seconds and saves 6 model files to `models/`:
- `tremor_model.pkl`, `tremor_scaler.pkl`
- `voice_model.pkl`, `voice_scaler.pkl`
- `voice_dataset_mapping.pkl`, `tremor_dataset_mapping.pkl`

### Test Backend

```bash
cd backend
python app.py
```

Visit `http://localhost:5000/api/health` to verify server is running.

### Test Frontend

Open `frontend/index.html` in browser or run:

```bash
cd frontend
python -m http.server 8000
```

## Production Deployment

### Backend
- Compatible with Flask production servers (Gunicorn, uWSGI)
- Vercel-ready (includes `application = app` export)

### Frontend
- Static files - can be hosted anywhere
- Progressive Web App (PWA) ready
- Works offline after first load

## Requirements

### Python (Backend)
- Python 3.8+
- Flask 2.3+
- scikit-learn 1.3+
- librosa 0.10+
- pandas, numpy, scipy

See `backend/requirements.txt` for complete list.

### Browser (Frontend)
- Modern browser with:
  - MediaRecorder API support
  - DeviceMotion API support
  - HTTPS (for sensor access on mobile)

## Testing Tools

### Sensor Test Page
`frontend/sensor-test.html` - Verify device sensors are working correctly

### Proxy Server
`frontend/proxy.py` - Forward requests when using ngrok tunnels

## License

Research and educational use only.

## Support

For issues or questions, please check the documentation in:
- `backend/README.md` - Backend setup and API details
- `backend/datasets/training_summary.json` - Training details
- `backend/datasets/frontend_params.json` - Parameter mappings
