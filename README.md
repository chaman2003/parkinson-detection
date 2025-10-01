# Parkinson's Detection System

AI-powered system for early detection of Parkinson's disease using voice analysis and tremor data.

## 🚀 Quick Start

**Run the application:**

```bash
python app.py
```

The server will start at: http://localhost:5000

On first run, models will be trained automatically (takes 2-3 minutes).  
Subsequent runs load models instantly!

## 📋 Requirements

- Python 3.13.5
- Virtual environment (`.venv`)
- Dependencies: `pip install -r backend/requirements.txt`

## 📁 Project Structure

```
parkinson/
├── app.py                  # Main application (run this!) - At root level
├── backend/
│   ├── core/               # Core ML modules (organized)
│   │   ├── __init__.py     # Package initialization
│   │   ├── ml_models.py    # ML ensemble models
│   │   ├── audio_features.py   # Audio feature extraction
│   │   ├── tremor_features.py  # Tremor feature extraction
│   │   ├── data_loader.py      # Dataset utilities
│   │   └── data_storage.py     # Storage manager
│   ├── models/             # Trained models (auto-generated)
│   ├── datasets/           # Training data
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Web interface
│   └── ...
└── .venv/                  # Python environment
```

## 🎯 Features

- **Voice Analysis**: 138 audio features extracted from voice recordings
- **Tremor Detection**: 25 motion-based features from accelerometer data
- **ML Ensemble**: SVM + Random Forest + Gradient Boosting
- **Auto-Training**: Automatically trains models on first run
- **REST API**: Easy integration with any frontend

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Analyze voice & tremor data |
| GET | `/api/models/info` | Model information |

## 📊 Training Data

- **Voice Samples**: 81 recordings (41 healthy, 40 Parkinson's)
- **Tremor Data**: 4,151 samples (2,051 healthy, 2,100 affected)
- **Training Time**: ~3 minutes (CPU)
- **Accuracy**: 65.5% (voice), 43.7% (tremor)

## 🔄 Retraining Models

To retrain from scratch:

1. Delete models: `Remove-Item -Recurse backend\models\*`
2. Run app: `python backend\app.py`

## 📖 Documentation

- `SETUP_GUIDE.md` - Detailed setup and cleanup information
- `TRAINING_COMPLETE.md` - Training details and configuration
- `backend/REAL_DATASET_GUIDE.md` - Dataset documentation

## 🛠️ Tech Stack

- **Backend**: Flask, Python 3.13
- **ML**: scikit-learn 1.7.2
- **Audio**: librosa 0.10.1
- **Frontend**: Vanilla JS, HTML5, CSS3

## ✅ Single File Operation

Everything runs from **one file**: `app.py` (at project root)

- ✅ Automatic model detection
- ✅ Auto-training if needed
- ✅ Instant loading if trained
- ✅ No manual steps required
- ✅ All helper modules organized in `backend/core/`

## 📦 Clean Architecture

- **`app.py`** - Single entry point at root level
- **`backend/core/`** - All interconnected Python modules (organized)
- **`backend/models/`** - Trained ML models
- **`backend/datasets/`** - Training data
- **`frontend/`** - Web interface

## 📝 License

This is an educational/research project for Parkinson's disease detection.

## 🎉 Getting Started

Just run:
```bash
python app.py
```

That's it! The app handles everything else automatically. 🚀
