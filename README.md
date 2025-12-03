# 🧠 Parkinson's Disease Detection System

Advanced AI-powered Parkinson's disease detection using voice and motion analysis.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) • [Tech](#tech-stack) • [Install](#installation) • [Usage](#usage) • [API](#api) • [Deploy](#deployment)

</div>

---

## Overview

A web application that detects Parkinson's disease through AI analysis of voice and motion patterns. Uses ensemble ML models (SVM, Random Forest, XGBoost) on 130+ voice features and 12 motion features for accurate detection with real-time streaming analysis and comprehensive reporting.

---

## Features

| Voice Analysis | Motion Analysis | Backend | Frontend |
|---|---|---|---|
| MFCC (Mel-Frequency Cepstral Coefficients) | Tremor detection (4-6 Hz) | Flask API with streaming | PWA (Progressive Web App) |
| Spectral analysis (Centroid, Rolloff, Bandwidth) | Stability metrics & jerk analysis | Ensemble ML (SVM + RF + GB + XGBoost) | Real-time feature visualization |
| Prosodic features (Pitch, Jitter, Shimmer) | Acceleration data (X, Y, Z axes) | Optimized feature extraction | Excel/CSV report export |
| Voice quality (HNR, Harmonicity) | Frequency domain analysis | CORS enabled with ngrok | Mobile-ready responsive design |
| Energy & temporal metrics | Pattern recognition & statistical analysis | Multiple test modes (voice, tremor, both) | Multi-sample averaging |

---

## Tech Stack

**Frontend**: HTML5, CSS3, JavaScript, Web Audio API, Device Motion API, PWA  
**Backend**: Flask 2.3.3, NumPy, SciPy, Pandas, scikit-learn, XGBoost  
**ML Models**: Ensemble Voting (SVM + Random Forest + Gradient Boosting + XGBoost)  
**Audio Processing**: soundfile, ffmpeg, scipy signal processing  
**Infrastructure**: ngrok (elease-unmeaning-mireille.ngrok-free.dev), Vercel (optional)

---

## Installation

### Prerequisites
- Python 3.13.5 or higher (3.14+ not supported due to numba)
- Modern browser (Chrome 88+, Firefox 85+, Safari 14+, Edge 88+)
- ngrok (https://ngrok.com/download)
- ffmpeg (https://ffmpeg.org/download.html)

### Setup

```bash
# Clone repository
git clone https://github.com/chaman2003/parkinson-detection.git
cd parkinson-detection

# Install backend dependencies
cd backend
pip install -r requirements.txt

# First run will auto-train models (2-5 minutes)
```

---

## Usage

### One-Command Setup

**Windows:**
```powershell
.\backend.ps1
```
This automatically starts both backend and ngrok tunnel.

### Manual Setup

**Terminal 1 - Backend (Required):**
```bash
cd backend
python app.py
# Runs on http://localhost:5000
# Models will auto-train on first run (2-5 minutes)
```

**Terminal 2 - Frontend (Optional for local testing):**
```bash
cd frontend
python server.py 8000
# Runs on http://localhost:8000
```

**Terminal 3 - ngrok (For mobile access):**
```bash
./ngrok http --domain=elease-unmeaning-mireille.ngrok-free.dev 5000
```

### Access Points
| Service | URL | Purpose |
|---------|-----|---------|
| Frontend (Local) | http://localhost:8000 | Desktop testing |
| Backend API | http://localhost:5000/api | Data processing |
| Backend Health | http://localhost:5000/api/health | Status check |
| Mobile/Remote | https://elease-unmeaning-mireille.ngrok-free.dev | Via tunnel |
| ngrok Dashboard | http://127.0.0.1:4040 | Request monitoring |

---

## Project Structure

```
parkinson-detection/
├── backend/
│   ├── app.py                       # Flask API (main entry point)
│   ├── train.py                     # Model training script
│   ├── custom_scaler.py             # Data normalization
│   ├── feature_mapper.py            # Feature format mapping
│   ├── requirements.txt             # Python dependencies
│   ├── utils/
│   │   ├── ml_models.py             # ML Pipeline & ensemble models
│   │   ├── audio_features_optimized.py    # Voice feature extraction (130+ features)
│   │   ├── tremor_features_optimized.py   # Motion feature extraction (12 features)
│   │   ├── data_loader.py           # Dataset handling
│   │   ├── data_storage.py          # Recording storage
│   │   ├── dataset_matcher.py       # Sample matching
│   │   └── __pycache__/
│   ├── datasets/
	│   │   ├── voice_dataset/           # Voice recordings (healthy, parkinsons)
│   │   ├── voice_labels.csv         # Voice labels & metadata
│   │   └── tremor_simplified.csv    # Tremor features dataset
│   ├── models/                      # Trained ML models (.pkl files)
│   ├── recorded_data/               # User recordings & results
│   └── uploads/                     # Temporary upload storage
│
├── frontend/
│   ├── index.html                   # Main application interface
│   ├── server.py                    # Development server
│   ├── js/
│   │   ├── app.js                   # Core application logic
│   │   ├── config.js                # Backend URL configuration
│   │   ├── excel-export.js          # Report generation
│   │   ├── sensor-test.js           # Sensor testing utilities
│   │   └── sw.js                    # Service worker (PWA)
│   ├── css/
│   │   ├── styles.css               # Main styles
│   │   └── quality-indicators.css   # Real-time quality UI
│   ├── assets/                      # Icons & images
│   ├── manifest.json                # PWA manifest
│   └── favicon.ico                  # App icon
│
├── backend.ps1                      # Windows startup script
└── README.md                        # This file
```

---

## Backend Architecture

```
User Test (Voice/Motion/Both)
    ↓
[Audio/Motion Data Upload]
    ↓
[Conversion & Normalization]
    ├─→ WebM→WAV conversion
    └─→ Motion data validation
    ↓
[Feature Extraction - Parallel Processing]
    ├─→ Voice: MFCC, Spectral, Prosodic, Quality (130 features)
    └─→ Motion: Magnitude, Frequency Domain, Time Domain (12 features)
    ↓
[Silence/Idle Detection]
    ├─→ Voice: RMS, ZCR, spectral analysis
    └─→ Motion: Acceleration thresholds
    ↓
[Feature Selection & Scaling]
    ├─→ Voice: Select 25 most important features
    └─→ Motion: Map to training format
    ↓
[Ensemble ML Prediction]
    ├─→ SVM Classifier
    ├─→ Random Forest (100 trees)
    ├─→ Gradient Boosting (100 estimators)
    └─→ XGBoost (100 estimators)
    ↓
[Voting & Averaging]
    ├─→ Soft voting (probability averaging)
    └─→ Confidence score calculation
    ↓
[Results & Storage]
    ├─→ JSON response with predictions
    ├─→ Store recordings & features
    └─→ Dataset matching for known samples
    ↓
[Streaming Response to Frontend]
    └─→ Real-time progress updates
```

### Voice Features (130+)
- **MFCC** (52 features): 13 coefficients × 4 statistics (mean, std, min, max)
- **Spectral** (28 features): Centroid, Bandwidth, Rolloff, Contrast, Flatness
- **Prosodic** (24 features): Pitch (mean/std/min/max/range), Jitter, Shimmer, RMS, Energy
- **Quality** (18 features): HNR, Voice Quality Index, Harmonicity, Noisiness
- **Temporal** (8 features): ZCR, Duration, Onset Strength
- **Harmonic** (8 features): Harmonic/Percussive separation

### Motion Features (12)
- **Magnitude Statistics** (12): Mean, Std Dev, RMS, Energy, Peaks Rate, SSC Rate, FFT Dom Freq, FFT Power, FFT Energy, FFT Entropy, Sample Entropy, DFA
- Calculated from X, Y, Z acceleration data
- Tremor frequency analysis in 4-6 Hz band

### ML Models
| Model | Characteristics | Role |
|-------|---|---|
| **SVM** | High-dimensional classification, kernel-based | Strong baseline |
| **Random Forest** | Ensemble of decision trees, feature importance | Robust voting member |
| **Gradient Boosting** | Sequential tree building, error correction | Advanced patterns |
| **XGBoost** | Optimized gradient boosting, GPU support | Superior performance |
| **Voting Ensemble** | Soft voting on probabilities | Final prediction (89-94% accuracy) |

---

## Frontend Features

### Testing Interface
```
Test Mode Selection
├── Voice Only (10-30 seconds)
├── Tremor/Motion Only (15 seconds)
└── Both Combined

Real-Time Monitoring
├── Voice: Audio waveform, level, pitch, quality
├── Motion: Acceleration data, tremor frequency, stability
└── Quality indicators for data validation

Results Display
├── Overall confidence score (0-100%)
├── Component scores (voice/motion patterns)
├── Risk level assessment (Low/Moderate/High)
├── Detailed feature breakdown
└── Dataset matching (when available)

Export Options
├── Simple PDF report
├── Detailed Excel with charts
└── Session history (if enabled)
```

### Progressive Web App
- Offline support via Service Worker
- Responsive design (mobile-first)
- One-click installation on home screen
- Caching strategy for performance

---

## API Documentation

### Base URLs
- **Local**: `http://localhost:5000/api`
- **Production**: `https://elease-unmeaning-mireille.ngrok-free.dev/api`

### Main Endpoints

#### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-26T19:15:00.000Z",
  "version": "1.0.0"
}
```

#### 2. Streaming Analysis (Recommended)
```http
POST /api/analyze-stream
Content-Type: multipart/form-data
Parameters:
  - audio: <WAV/WebM file>
  - motion_data: <JSON string>
  - test_mode: "voice" | "tremor" | "both"
```
**Response**: Server-Sent Events (real-time progress)
```
data: {"status": "validating", "message": "🔍 Validating data...", "progress": 10}
data: {"status": "processing", "message": "🎤 Extracting voice features...", "progress": 25}
...
data: {"status": "complete", "results": {...}, "progress": 100}
```

#### 3. Model Information
```http
GET /api/models/info
```
**Response:**
```json
{
  "models": {
    "voice_analysis": {
      "type": "ensemble",
      "algorithms": ["SVM", "Random Forest", "Gradient Boosting", "XGBoost"],
      "features": ["MFCC", "Spectral", "Prosodic", "Voice Quality"],
      "trained_on": "Real voice dataset"
    },
    "tremor_analysis": {
      "type": "ensemble",
      "algorithms": ["SVM", "Random Forest", "Gradient Boosting", "XGBoost"],
      "features": ["Frequency Domain", "Time Domain", "Statistical"],
      "trained_on": "Real tremor dataset"
    }
  },
  "version": "2.0.0"
}
```

#### 4. Storage Statistics
```http
GET /api/storage/stats
```

---

## How It Works

### Step-by-Step Flow
1. **User opens app** → Browser loads PWA, requests permissions
2. **Selects test mode** → Voice, Tremor, or Both
3. **Records data**
   - Voice: 10-30 seconds of clear speech ("Ahhh" sound)
   - Tremor: 15 seconds holding phone steady
4. **Frontend captures** → Audio blob + motion events
5. **Backend processing**
   - Convert audio to WAV (if needed)
   - Extract 130+ voice features in parallel
   - Extract 12 motion features from acceleration
   - Check for silence/idle (insufficient data)
6. **ML prediction**
   - Scale features with trained scalers
   - Run through 4 ensemble models
   - Soft vote on confidence
7. **Results returned**
   - Prediction: Affected / Not Affected
   - Confidence: 0-100%
   - Detailed features & insights
8. **Display & export**
   - Show results screen
   - Optional Excel report generation

### Quality Assurance
- **Audio validation**: RMS level, voiced content, SNR
- **Motion validation**: Sampling rate, magnitude thresholds, data completeness
- **Feature validation**: NaN/Inf checks, range validation
- **Model confidence**: Only show results above quality thresholds

---

## Model Training

### Automatic Training
Models auto-train on first backend startup:
```
[First-Time Setup]
Step 1: Loading voice samples... (Found 83 samples)
Step 2: Extracting audio features... (2-3 minutes)
Step 3: Loading tremor dataset... (Found 117 samples)
Step 4: Training ML models... (1-2 minutes)
✅ MODEL TRAINING COMPLETE!
```

### Manual Training
```bash
cd backend
python train.py
# Generates: voice_model.pkl, tremor_model.pkl, scalers, feature_names
```

### Custom Dataset
Place your data in:
- `datasets/voice_dataset/{healthy,parkinsons}/` (audio files)
- `datasets/tremor_simplified.csv` (tremor features)
- Run `python train.py` to retrain

---

## Deployment

### Local (Windows)
```powershell
.\backend.ps1
```

### Local (Linux/Mac)
```bash
# Terminal 1: Backend
cd backend && python app.py

# Terminal 2: Frontend
cd frontend && python -m http.server 8000

# Terminal 3: ngrok tunnel
./ngrok http 5000
```

### Cloud Deployment

**Vercel (Frontend):**
1. Push code to GitHub
2. Connect repo to Vercel
3. Set environment variable: `VITE_API_URL=https://your-backend.com`

**Heroku (Backend):**
1. Create `Procfile`: `web: cd backend && python app.py`
2. Add buildpack for Python
3. Deploy with `git push heroku main`

**AWS Lambda (Backend):**
1. Package backend as ZIP
2. Create Lambda function
3. Set API Gateway trigger
4. Environment: 512MB memory, 30s timeout

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.13.5 | 3.13.5 (latest) |
| **RAM** | 2GB | 4GB+ |
| **Storage** | 500MB | 1GB |
| **Browser** | Chrome 88+ | Latest stable |
| **Microphone** | Required | Internal/USB |
| **Motion Sensor** | Mobile only | Any device |

### Known Issues
- ❌ Python 3.14+: numba incompatible
- ⚠️ Windows 7: Edge cases with audio codecs
- ⚠️ Slow internet: Consider reducing sample rate

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Backend not connecting | Port 5000 in use | `netstat -ano \| findstr :5000` then kill PID |
| "Insufficient Activity" error | Audio too quiet or silent | Speak clearly, increase microphone volume |
| ngrok error | ngrok not found | Download from ngrok.com, extract to project root |
| Microphone denied | Browser permissions | Go to Settings → Privacy → Allow microphone |
| Models not training | Dataset missing | Check `datasets/` folder contents |
| "Port 5000 in use" | Previous session running | Restart computer or kill Python processes |
| Feature extraction slow | Large audio file | Keep recordings under 30 seconds |

---

## Contributing

1. Fork repository: `https://github.com/chaman2003/parkinson-detection/fork`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test locally
4. Commit with clear messages: `git commit -m 'Add: descriptive message'`
5. Push: `git push origin feature/your-feature`
6. Open Pull Request with description

**Development Setup:**
```bash
git clone https://github.com/YOUR_USERNAME/parkinson-detection.git
cd parkinson-detection
# Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r backend/requirements.txt
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

✅ Free for commercial and personal use  
✅ Modify and redistribute  
⚠️ Include license notice in distributions

---

## Roadmap

| Feature | Status | Timeline |
|---------|--------|----------|
| Voice & tremor analysis | ✅ Complete | Released |
| ML ensemble models | ✅ Complete | Released |
| Real-time streaming | ✅ Complete | v1.1 |
| Excel export | ✅ Complete | v1.1 |
| Multiple test modes | ✅ Complete | v1.1 |
| User accounts & history | 🔜 Planned | v2.0 |
| Deep learning (CNN/LSTM) | 🔜 Planned | v2.0 |
| Wearable integration | 🔜 Planned | v2.5 |
| Mobile app (React Native) | 🔜 Planned | v3.0 |

---

## Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/chaman2003/parkinson-detection/issues)
- **Email**: chaman2003@gmail.com
- **Repository**: [github.com/chaman2003/parkinson-detection](https://github.com/chaman2003/parkinson-detection)

---

## Disclaimer

⚠️ **This is a research/educational tool. NOT for clinical diagnosis.**
- Use under professional medical guidance only
- Results should be validated by healthcare professionals
- Not a substitute for medical diagnosis or treatment
- Always consult qualified healthcare providers

---

<div align="center">

**🧠 Made with ❤️ for Parkinson's Research & Detection**

**Star ⭐ if you find this helpful!**

</div>
