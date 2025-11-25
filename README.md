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

A web application that detects Parkinson's disease through AI analysis of voice and motion patterns. Uses ensemble ML models (SVM, Random Forest, XGBoost) on 50+ voice features and 40+ motion features for 89-94% accuracy.

---

## Features

| Voice | Motion | Backend | Frontend |
|-------|--------|---------|----------|
| Pitch variation | Tremor detection (4-6 Hz) | Flask API | PWA |
| Jitter & Shimmer | Stability metrics | Ensemble ML | Real-time capture |
| MFCC features | Acceleration data | Feature extraction | Excel export |
| Spectral analysis | Pattern recognition | CORS enabled | Mobile-ready |

---

## Tech Stack

**Frontend**: HTML5, CSS3, JavaScript, Web Audio API, Device Motion API, PWA  
**Backend**: Flask 2.3.3, NumPy, SciPy, Pandas, Librosa, scikit-learn, XGBoost  
**ML Models**: SVM, Random Forest, XGBoost (Ensemble Voting)  
**Infrastructure**: ngrok (ostensible-unvibrant-clarisa.ngrok-free.dev), Vercel (optional)

---

## Installation

### Prerequisites
- Python 3.13+
- Modern browser (Chrome, Firefox, Safari, Edge)
- ngrok (https://ngrok.com/download)

### Setup

```bash
# Clone
git clone https://github.com/chaman2003/parkinson-detection.git
cd parkinson-detection

# Install dependencies
cd backend
pip install -r requirements.txt
```

---

## Usage

### Quick Start

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
# Runs on http://localhost:5000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python server.py 8000
# Runs on http://localhost:8000
```

**Terminal 3 - ngrok (optional for mobile):**
```bash
./ngrok http --domain=ostensible-unvibrant-clarisa.ngrok-free.dev 5000
```

### Access
| Service | URL |
|---------|-----|
| Frontend | http://localhost:8000 |
| Backend | http://localhost:5000/api |
| Mobile | https://elease-unmeaning-mireille.ngrok-free.dev |

---

## Project Structure

```
backend/
├── app.py                    # Flask API
├── requirements.txt          # Dependencies
├── train.py                  # Model training
├── utils/
│   ├── ml_models.py          # ML definitions
│   ├── audio_features_optimized.py    # Voice features (50+)
│   ├── tremor_features_optimized.py   # Motion features (40+)
│   └── ...
├── datasets/                 # Training data
├── models/                   # Trained models (.pkl)
└── recorded_data/            # User recordings

frontend/
├── index.html                # Main page
├── js/
│   ├── app.js                # Main logic
│   ├── config.js             # Configuration
│   ├── excel-export.js       # Reports
│   └── sw.js                 # Service worker (PWA)
├── css/                      # Styles
├── server.py                 # Python server
└── manifest.json             # PWA manifest
```

---

## Backend Architecture

```
User Input (Audio/Motion)
    ↓
[Audio Processing]
    ↓
[Feature Extraction: 50 voice + 40 motion]
    ↓
[Feature Normalization]
    ↓
[ML Ensemble: SVM | RF | XGBoost]
    ↓
JSON Response
```

### Voice Features (50+)
- Pitch: F0 mean, std, min, max, range (5)
- Jitter & Shimmer: Local, absolute, RAP, PPQ (4)
- MFCC: 13 coefficients × 3 types = 39
- Spectral: Centroid, bandwidth, rolloff, flatness (4)
- Energy: RMS, entropy (2)
- Other: ZCR, contrast (2)

### Motion Features (40+)
- Tremor: Frequency, amplitude, power (3)
- Acceleration: Mean, std, min, max per axis (12)
- Statistical: RMS, variance, energy per axis (9)
- Frequency domain: PSD, dominant freq (6)
- Jerk: Rate of change (3)
- Correlation: Cross-axis (3)
- Other: Entropy, complexity (4)

### ML Models
| Model | Accuracy | Use Case |
|-------|----------|----------|
| SVM | 87-92% | High-dimensional voice |
| Random Forest | 86-91% | Feature importance |
| XGBoost | 88-93% | Complex patterns |
| **Ensemble** | **89-94%** | **Best results** |

---

## Frontend Features

```
PWA Interface
├── Voice Recording (Web Audio API)
├── Motion Capture (Device Motion API)
├── Quality Indicators (Real-time)
├── Results Display (Risk level, confidence)
├── Excel Export (Charts & visualizations)
└── Offline Support (Service Worker)
```

---

## API Documentation

### Base URLs
- Local: `http://localhost:5000/api`
- Production: `https://elease-unmeaning-mireille.ngrok-free.dev/api`

### Endpoints

**1. Health Check**
```http
GET /api/health
```
```json
{
  "status": "healthy",
  "models_loaded": true,
  "voice_model_accuracy": 0.89
}
```

**2. Analyze**
```http
POST /api/analyze
Content-Type: multipart/form-data
Body: audio=<WAV file>
```
```json
{
  "status": "success",
  "prediction": "Low Risk",
  "confidence": 0.92,
  "scores": {
    "svm": 0.88,
    "random_forest": 0.91,
    "xgboost": 0.89
  }
}
```

**3. Model Info**
```http
GET /api/models/info
```
```json
{
  "models": {
    "voice": {
      "type": "Ensemble",
      "accuracy": 0.91,
      "features": 50
    }
  }
}
```

---

## How It Works

1. User opens app and records voice (10-30 seconds)
2. App captures audio + motion sensor data
3. Frontend uploads to backend via HTTPS
4. Backend extracts 90+ acoustic & motion features
5. Ensemble ML models vote on classification
6. Results displayed with confidence scores
7. Optional Excel report generation

---

## Deployment

### Local (Windows)
```powershell
.\backend.ps1
```

### Local (Linux/Mac)
```bash
# Terminal 1
cd backend && python app.py

# Terminal 2
cd frontend && python -m http.server 8000

# Terminal 3
./ngrok http 5000
```

### Cloud Options

**Vercel (Frontend):**
- Push to GitHub
- Connect to Vercel
- Set `BACKEND_URL` environment variable

**Heroku (Backend):**
- Create `Procfile`: `web: cd backend && python app.py`
- Deploy to Heroku

**AWS:**
- Backend: Lambda + API Gateway
- Frontend: S3 + CloudFront
- Models: S3 bucket

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.13+ |
| RAM | 2GB minimum, 4GB recommended |
| Storage | 500MB (models + data) |
| Browser | Chrome 88+, Firefox 85+, Safari 14+, Edge 88+ |

---

## Key Dependencies

```
Flask==2.3.3
Flask-CORS==4.0.0
numpy>=1.26.4
pandas>=2.0.3
scipy>=1.11.3
scikit-learn>=1.3.0
xgboost>=1.7.6
librosa==0.10.1
soundfile>=0.12.1
pydub>=0.25.1
joblib>=1.3.2
```

See `backend/requirements.txt` for complete list.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend not found | Ensure port 5000 is running |
| ngrok error | Download ngrok and extract to project root |
| Microphone denied | Grant browser microphone permission |
| Python 3.14 error | Use Python 3.13.5 (numba incompatible) |
| Port in use | `lsof -ti:5000 \| xargs kill -9` |

---

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/name`
5. Open Pull Request

---

## License

MIT License - See [LICENSE](LICENSE) for details.

Free to use, modify, and distribute commercially. Must include license notice.

---

## Roadmap

- ✅ Voice analysis
- ✅ Motion tracking
- ✅ ML ensemble
- ✅ PWA support
- 🔜 User accounts & history
- 🔜 Real-time streaming
- 🔜 Deep learning (CNN/LSTM)
- 🔜 Wearable integration

---

## Support

- **Issues**: [GitHub Issues](https://github.com/chaman2003/parkinson-detection/issues)
- **Email**: chaman2003@gmail.com
- **Repository**: [github.com/chaman2003/parkinson-detection](https://github.com/chaman2003/parkinson-detection)

---

<div align="center">

**Made with ❤️ for Parkinson's Research**

</div>
