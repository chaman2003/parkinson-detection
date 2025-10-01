# 🧠 Parkinson's Disease Detection System# 🧠 Parkinson's Detection PWA



> **AI-Powered Medical Analysis Using Voice & Motion Detection**  A **Progressive Web Application** for early Parkinson's disease detection using AI-powered multimodal analysis of voice and tremor patterns.

> Real-time ML inference with 350+ features and ensemble algorithms

## 🎯 Features

---

- **🎤 Voice Analysis**: Advanced audio processing using MFCC, spectral, and prosodic features

## 📋 Table of Contents- **📱 Tremor Detection**: High-precision motion sensor analysis (3-12 Hz tremor band)

- **🤖 AI Models**: Ensemble machine learning (Random Forest + SVM + XGBoost)

1. [Overview](#overview)- **📊 Real-time Quality Monitoring**: Live data quality indicators and metrics

2. [Quick Start](#quick-start)- **🔄 PWA Capabilities**: Offline functionality, installable interface

3. [System Architecture](#system-architecture)- **🎛️ Test Modes**: Voice-only, tremor-only, or complete analysis options

4. [Features](#features)

5. [API Documentation](#api-documentation)## 🚀 Live Demo

6. [Development](#development)

7. [Model Information](#model-information)- **Frontend**: [Deployed on Vercel](https://your-frontend-url.vercel.app)

8. [Troubleshooting](#troubleshooting)- **Backend API**: [Deployed on Vercel](https://your-backend-url.vercel.app)



---## 📊 Data Accuracy Standards



## 🎯 Overview- **Audio**: 128kbps bitrate, 50ms precision, real-time SNR monitoring

- **Motion**: 100Hz sampling, 6-decimal precision, microsecond timestamps

A comprehensive web-based Parkinson's disease detection system that uses:- **Validation**: 80% quality threshold enforcement

- **Voice Analysis**: 150+ audio features (MFCC, spectral, prosodic, voice quality)- **Processing**: Research-validated parameter settings

- **Tremor Detection**: 200+ motion features (FFT, time/frequency domain, statistical)

- **Ensemble ML**: 4 algorithms (SVM, Random Forest, Gradient Boosting, XGBoost)## 🛠️ Installation & Setup

- **Real-time Analysis**: 3-5 second processing with 100% training accuracy

### Local Development

### Technology Stack

1. **Clone the repository**

**Backend:**   ```bash

- Python 3.13+ with Flask   git clone https://github.com/chaman2003/parkinson-detection.git

- Scikit-learn + XGBoost   cd parkinson-detection

- Librosa (audio processing)   ```

- NumPy + SciPy (scientific computing)

2. **Backend Setup**

**Frontend:**   ```bash

- Vanilla JavaScript   cd backend

- Web Audio API   python -m venv .venv

- Device Motion API   .venv\Scripts\activate  # Windows

- Progressive Web App (PWA)   pip install -r requirements.txt

   python app.py

---   ```



## 🚀 Quick Start3. **Frontend Setup**

   - Open `index.html` with Live Server in VS Code

### Prerequisites   - Or serve with: `python -m http.server 8080`



- Python 3.13+ installed## 🌐 API Endpoints

- Microphone access

- Motion sensors (mobile device recommended)- `GET /api/health` - Health check

- `POST /api/analyze` - Main analysis endpoint

### Installation- `POST /api/demo` - Demo analysis with mock results

- `GET /api/models/info` - Model information

**1. Clone the repository:**

```bash## 📄 License

git clone https://github.com/chaman2003/parkinson-detection.git

cd parkinson-detectionThis project is licensed under the MIT License.

```

**⚠️ Medical Disclaimer**: This application is for research and educational purposes only.

**2. Create virtual environment (recommended):**

```bash## Project Structure

python -m venv .venv

.venv\Scripts\activate  # Windows```

source .venv/bin/activate  # Linux/Macparkinson/

```├── frontend/                 # PWA Frontend

│   ├── index.html           # Main HTML file

**3. Install dependencies:**│   ├── styles.css           # CSS styles

```bash│   ├── app.js               # Main JavaScript logic

pip install -r backend/requirements.txt│   ├── manifest.json        # PWA manifest

```│   ├── sw.js                # Service worker

│   └── assets/              # Images and icons

### Running the Application└── backend/                 # Python Backend

    ├── app.py               # Flask/FastAPI server

**Single Command Setup:**    ├── ml_models.py         # Machine learning pipeline

```bash    ├── requirements.txt     # Python dependencies

cd backend    └── models/              # Trained model files

python app.py```

```

## Setup Instructions

**What happens:**

- **First run**: Automatically generates 1000 synthetic training samples and trains ML models (~30 seconds)### Backend Setup

- **Subsequent runs**: Loads cached models instantly1. Navigate to the backend directory

- **Server**: Starts on `http://localhost:5000`2. Install dependencies: `pip install -r requirements.txt`

3. Run the server: `python app.py`

**Access the frontend:**

1. Open `index.html` in your browser### Frontend Setup

2. Allow microphone and motion permissions1. Open the frontend directory

3. Start testing!2. Serve the files using a local web server

3. Access the app through HTTPS (required for PWA features)

---

## Usage

## 🏗️ System Architecture

1. Open the PWA on your smartphone

### Directory Structure2. Grant microphone and motion sensor permissions

3. Follow the instructions to record voice and tremor data

```4. View the analysis results with confidence scores

parkinson/

├── backend/## Technology Stack

│   ├── app.py                  # Main server with integrated training

│   ├── ml_models.py            # Ensemble ML pipeline- **Frontend**: HTML5, CSS3, JavaScript (ES6+)

│   ├── audio_features.py       # 150+ audio feature extraction- **Backend**: Python, Flask/FastAPI

│   ├── tremor_features.py      # 200+ motion feature extraction- **ML Libraries**: scikit-learn, XGBoost, NumPy, SciPy

│   ├── requirements.txt        # Python dependencies- **Audio Processing**: Web Audio API, librosa (backend)

│   ├── models/                 # Trained model files (auto-generated)- **Motion Detection**: DeviceMotionEvent API
│   │   ├── voice_model.pkl
│   │   ├── voice_scaler.pkl
│   │   ├── tremor_model.pkl
│   │   └── tremor_scaler.pkl
│   └── uploads/                # Temporary audio files
├── index.html                  # Main frontend application
├── app.js                      # Frontend logic
├── styles.css                  # UI styling
├── sw.js                       # Service worker (PWA)
├── manifest.json               # PWA manifest
└── assets/                     # Icons and images
```

### Processing Pipeline

```
User Input (Audio + Motion)
    ↓
Feature Extraction
    ├─→ Audio: 150+ features (librosa)
    └─→ Motion: 200+ features (FFT, stats)
    ↓
Feature Scaling (StandardScaler)
    ↓
Ensemble ML Inference
    ├─→ SVM (RBF kernel)
    ├─→ Random Forest (200 trees)
    ├─→ Gradient Boosting (150 estimators)
    └─→ XGBoost (200 estimators)
    ↓
Soft Voting (Probability Averaging)
    ↓
Prediction + Confidence Scores
```

---

## ✨ Features

### Audio Analysis (150+ Features)

**MFCC (Mel-Frequency Cepstral Coefficients):**
- 13 coefficients + deltas + delta-deltas
- Voice characteristic fingerprinting

**Spectral Features:**
- Spectral centroid, bandwidth, rolloff
- Zero-crossing rate
- Frequency domain analysis

**Prosodic Features:**
- Fundamental frequency (F0) extraction
- Jitter (frequency perturbation)
- Shimmer (amplitude perturbation)
- Voice tremor indicators

**Voice Quality:**
- Harmonic-to-Noise Ratio (HNR)
- Signal-to-noise estimation
- Voice quality metrics

**Temporal Features:**
- Onset strength detection
- Tempo estimation
- Rhythmic patterns

**Harmonic Features:**
- Chroma features (12 pitch classes)
- Tonnetz (tonal centroid features)
- Harmonic structure analysis

### Tremor Analysis (200+ Features)

**Frequency Domain:**
- FFT analysis (0-25 Hz range)
- 4-6 Hz tremor band detection
- Power spectral density
- Dominant frequency identification

**Time Domain:**
- Acceleration statistics (X, Y, Z axes)
- Mean, standard deviation, variance
- Min/max amplitude detection

**Statistical Features:**
- Skewness (asymmetry)
- Kurtosis (tail heaviness)
- Percentiles (25th, 50th, 75th)

**Tremor-Specific:**
- Tremor amplitude calculation
- Zero-crossing rate
- Jerk analysis (rate of acceleration change)

**Movement Patterns:**
- Autocorrelation analysis
- Movement intensity variation
- Directional statistics

**Stability Metrics:**
- Postural stability index
- Path length calculation
- Movement smoothness

### ML Ensemble

**Algorithms:**
1. **SVM (Support Vector Machine)**
   - RBF kernel
   - Gamma: 'scale'
   - Probability estimates enabled

2. **Random Forest**
   - 200 trees
   - Max depth: 20
   - Min samples split: 5

3. **Gradient Boosting**
   - 150 estimators
   - Learning rate: 0.1
   - Max depth: 5

4. **XGBoost**
   - 200 estimators
   - Learning rate: 0.1
   - Max depth: 5
   - GPU acceleration support

**Training:**
- **Voting**: Soft voting (probability averaging)
- **Cross-Validation**: 5-fold
- **Scaling**: StandardScaler normalization
- **Persistence**: Pickle-based model caching

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T10:30:00.000Z",
  "version": "1.0.0"
}
```

#### 2. Analyze (Main ML Endpoint)
```http
POST /api/analyze
Content-Type: multipart/form-data
```

**Parameters:**
- `audio` (file): Audio recording (WebM/WAV/MP3)
- `motion_data` (JSON string): Array of motion samples

**Motion Data Format:**
```json
[
  {
    "timestamp": 1000,
    "accelerationX": 0.5,
    "accelerationY": -0.3,
    "accelerationZ": 9.8,
    "rotationAlpha": 45.2,
    "rotationBeta": 12.5,
    "rotationGamma": -5.3
  }
]
```

**Response:**
```json
{
  "prediction": "Not Affected",
  "confidence": 0.85,
  "voice_confidence": 0.82,
  "tremor_confidence": 0.88,
  "features": {
    "Voice Stability": 0.65,
    "Tremor Frequency": 0.78,
    "Voice Quality": 0.72,
    "Postural Stability": 0.68,
    "Vocal Tremor": 0.45,
    "Motion Variability": 0.73
  },
  "metadata": {
    "processing_time": 3.2,
    "audio_features_count": 150,
    "tremor_features_count": 200,
    "model_type": "ensemble_ml"
  }
}
```

#### 3. Model Information
```http
GET /api/models/info
```

**Response:**
```json
{
  "voice_model": {
    "type": "VotingClassifier",
    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],
    "n_features": 150,
    "trained": true
  },
  "tremor_model": {
    "type": "VotingClassifier",
    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],
    "n_features": 200,
    "trained": true
  }
}
```

### Example Usage

**cURL:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "audio=@recording.webm" \
  -F "motion_data=[{\"timestamp\":1000,\"accelerationX\":0.5,...}]"
```

**Python:**
```python
import requests

files = {'audio': open('recording.webm', 'rb')}
data = {'motion_data': '[{"timestamp":1000,"accelerationX":0.5,...}]'}

response = requests.post('http://localhost:5000/api/analyze', files=files, data=data)
print(response.json())
```

**JavaScript:**
```javascript
const formData = new FormData();
formData.append('audio', audioBlob, 'recording.webm');
formData.append('motion_data', JSON.stringify(motionData));

const response = await fetch('http://localhost:5000/api/analyze', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```

---

## 💻 Development

### Training Your Own Models

**Using Real Data:**

1. **Prepare your dataset:**
   ```python
   # Format: (samples, features)
   voice_data = np.array([...])  # Shape: (n_samples, 150)
   tremor_data = np.array([...])  # Shape: (n_samples, 200)
   labels = np.array([0, 1, 0, ...])  # 0=healthy, 1=affected
   ```

2. **Replace training function in `backend/app.py`:**
   ```python
   def generate_training_data():
       # Load your real dataset here
       voice_data = load_voice_data()
       tremor_data = load_tremor_data()
       labels = load_labels()
       return voice_data, tremor_data, labels
   ```

3. **Delete cached models and retrain:**
   ```bash
   rm backend/models/*.pkl
   python backend/app.py
   ```

### Customizing ML Pipeline

**Edit `backend/ml_models.py`:**

```python
# Change hyperparameters
svm = SVC(kernel='rbf', C=10.0, gamma='auto')  # Adjust C and gamma
rf = RandomForestClassifier(n_estimators=500)  # More trees

# Add new algorithms
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50))
```

### Adding New Features

**Audio features (`backend/audio_features.py`):**
```python
def extract_custom_feature(self, y, sr):
    # Your feature extraction logic
    return feature_value
```

**Motion features (`backend/tremor_features.py`):**
```python
def extract_custom_motion_feature(self, data):
    # Your motion analysis logic
    return feature_value
```

### Frontend Customization

**Modify UI (`index.html`):**
- Update HTML structure
- Change recording durations
- Add new test modes

**Update logic (`app.js`):**
- Modify API endpoints
- Change data processing
- Update result visualization

---

## 🧪 Model Information

### Training Performance

**Synthetic Data Results:**
- **Voice Model CV Accuracy**: 100% (5-fold cross-validation)
- **Tremor Model CV Accuracy**: 100% (5-fold cross-validation)
- **Training Samples**: 1000 (500 healthy, 500 affected)
- **Training Time**: ~30 seconds (first run only)

**Note:** These accuracies are based on synthetic data with clear separation. Real-world clinical data will have lower accuracies.

### Model Files

Models are automatically saved to `backend/models/`:
- `voice_model.pkl` - Trained voice classifier
- `voice_scaler.pkl` - Voice feature scaler
- `tremor_model.pkl` - Trained tremor classifier
- `tremor_scaler.pkl` - Tremor feature scaler

**File sizes:**
- Each model: ~5-10 MB
- Total storage: ~20-40 MB

### Feature Importance

**Top Voice Features:**
1. Jitter (frequency perturbation)
2. Shimmer (amplitude perturbation)
3. MFCC coefficients
4. Harmonic-to-Noise Ratio
5. Fundamental frequency variation

**Top Tremor Features:**
1. 4-6 Hz power spectral density
2. Dominant frequency
3. Acceleration variance
4. Movement jerk
5. Postural stability metrics

---

## 🔧 Troubleshooting

### Installation Issues

**"No module named 'X'"**
```bash
pip install -r backend/requirements.txt
```

**"Python not found"**
- Install Python 3.13+ from [python.org](https://www.python.org)
- Add Python to PATH during installation

**Virtual environment issues**
```bash
# Recreate venv
rm -rf .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Runtime Issues

**"Port 5000 already in use"**
```bash
# Windows: Find and kill process
netstat -ano | findstr :5000
taskkill /PID <process_id> /F

# Or change port in app.py
app.run(debug=True, port=5001)
```

**"Training takes too long"**
- Normal for first run (~30 seconds)
- Reduce samples in `generate_training_data()`:
  ```python
  n_samples = 100  # Instead of 1000
  ```

**"Low prediction accuracy"**
- Current models use synthetic data
- Train with real clinical data for production
- Collect diverse patient samples
- Validate with cross-dataset testing

### Frontend Issues

**"Microphone not working"**
- Check browser permissions (camera icon in address bar)
- Use HTTPS or localhost only (WebRTC requirement)
- Try different browser (Chrome/Edge recommended)

**"Motion sensors not working"**
- Use mobile device (gyroscope required)
- Enable motion permissions in browser
- Some laptops have accelerometers

**"CORS errors"**
- Ensure Flask-CORS is installed
- Check backend is running on localhost:5000
- Clear browser cache

### Model Issues

**"Models not loading"**
```bash
# Check if models exist
ls backend/models/

# Delete and retrain
rm backend/models/*.pkl
python backend/app.py
```

**"Feature extraction errors"**
- Check audio file format (WebM/WAV/MP3 supported)
- Ensure audio is not empty or corrupted
- Verify motion data is valid JSON array

**"Inconsistent predictions"**
- Models trained on synthetic data
- Need real clinical data for reliability
- Consider ensemble temperature tuning
- Add prediction confidence thresholds

---

## 📊 System Requirements

### Minimum Requirements

**Hardware:**
- CPU: Dual-core 2.0 GHz
- RAM: 2 GB
- Storage: 500 MB
- Microphone + motion sensors

**Software:**
- Python 3.13+
- Modern browser (Chrome 90+, Edge 90+, Firefox 88+)
- Windows/Linux/macOS

### Recommended Requirements

**Hardware:**
- CPU: Quad-core 3.0 GHz
- RAM: 4 GB
- Storage: 1 GB
- SSD for faster model loading

**Software:**
- Python 3.13+
- Chrome/Edge latest
- GPU support (CUDA) for XGBoost acceleration

### Dependencies

**Python Packages:**
```
Flask>=3.0.0
Flask-CORS>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
librosa>=0.10.0
scipy>=1.11.0
```

**Browser APIs:**
- Web Audio API
- MediaRecorder API
- DeviceMotion API
- Fetch API
- Web Workers (PWA)

---

## 🚀 Production Deployment

### For Clinical Use

**1. Collect Real Data:**
- Collaborate with medical institutions
- Get ethical approval and patient consent
- Record voice + motion from diagnosed patients
- Include healthy control group
- Balance dataset (equal positive/negative samples)

**2. Retrain Models:**
```python
# Load your clinical dataset
voice_features = extract_voice_features(audio_files)
tremor_features = extract_tremor_features(motion_data)
labels = patient_diagnoses

# Train with your data
pipeline = ParkinsonMLPipeline()
pipeline.train_models(voice_features, labels, tremor_features, labels)
pipeline.save_models('models/')
```

**3. Validate Performance:**
- Use holdout test set (20% of data)
- Calculate sensitivity/specificity
- Test with cross-institutional data
- Clinical trial validation

**4. Deploy Server:**
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**5. Security:**
- Enable HTTPS (SSL/TLS certificates)
- Implement user authentication
- Add data encryption
- HIPAA compliance for medical data
- Regular security audits

### Scaling

**Horizontal Scaling:**
- Load balancer (nginx)
- Multiple Flask instances
- Redis for session management
- Database for result storage

**Optimization:**
- Model quantization (reduce size)
- Feature selection (reduce computation)
- Caching predictions
- Async processing with Celery

---

## 📝 License

This project is for educational and research purposes. For clinical use, consult with medical professionals and obtain necessary certifications.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Real clinical data integration
- Deep learning models (CNN/RNN)
- Mobile app development
- Additional biomarker analysis
- Multi-language support

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review code comments in source files
3. Open GitHub issue with error logs

---

## 🙏 Acknowledgments

Built with research from:
- Parkinson's Disease detection literature
- Voice analysis studies
- Tremor frequency research
- ML ensemble methods

---

**Made with ❤️ for medical AI research**
