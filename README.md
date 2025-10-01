# 🧠 Parkinson's Disease Detection System# 🧠 Parkinson's Disease Detection System# 🧠 Parkinson's Detection PWA



> **AI-Powered Medical Analysis Using Voice & Motion Detection**  

> Real-time ML inference with 350+ features and ensemble algorithms

> **AI-Powered Medical Analysis Using Voice & Motion Detection**  A **Progressive Web Application** for early Parkinson's disease detection using AI-powered multimodal analysis of voice and tremor patterns.

---

> Real-time ML inference with 350+ features and ensemble algorithms

## 📋 Table of Contents

## 🎯 Features

1. [Overview](#overview)

2. [Quick Start](#quick-start)---

3. [System Architecture](#system-architecture)

4. [Features](#features)- **🎤 Voice Analysis**: Advanced audio processing using MFCC, spectral, and prosodic features

5. [API Documentation](#api-documentation)

6. [Development](#development)## 📋 Table of Contents- **📱 Tremor Detection**: High-precision motion sensor analysis (3-12 Hz tremor band)

7. [Model Information](#model-information)

8. [Troubleshooting](#troubleshooting)- **🤖 AI Models**: Ensemble machine learning (Random Forest + SVM + XGBoost)



---1. [Overview](#overview)- **📊 Real-time Quality Monitoring**: Live data quality indicators and metrics



## 🎯 Overview2. [Quick Start](#quick-start)- **🔄 PWA Capabilities**: Offline functionality, installable interface



A comprehensive web-based Parkinson's disease detection system that uses:3. [System Architecture](#system-architecture)- **🎛️ Test Modes**: Voice-only, tremor-only, or complete analysis options

- **Voice Analysis**: 150+ audio features (MFCC, spectral, prosodic, voice quality)

- **Tremor Detection**: 200+ motion features (FFT, time/frequency domain, statistical)4. [Features](#features)

- **Ensemble ML**: 4 algorithms (SVM, Random Forest, Gradient Boosting, XGBoost)

- **Real-time Analysis**: 3-5 second processing with 100% training accuracy5. [API Documentation](#api-documentation)## 🚀 Live Demo



### Technology Stack6. [Development](#development)



**Backend:**7. [Model Information](#model-information)- **Frontend**: [Deployed on Vercel](https://your-frontend-url.vercel.app)

- Python 3.13+ with Flask

- Scikit-learn + XGBoost8. [Troubleshooting](#troubleshooting)- **Backend API**: [Deployed on Vercel](https://your-backend-url.vercel.app)

- Librosa (audio processing)

- NumPy + SciPy (scientific computing)



**Frontend:**---## 📊 Data Accuracy Standards

- Vanilla JavaScript

- Web Audio API

- Device Motion API

- Progressive Web App (PWA)## 🎯 Overview- **Audio**: 128kbps bitrate, 50ms precision, real-time SNR monitoring



---- **Motion**: 100Hz sampling, 6-decimal precision, microsecond timestamps



## 🚀 Quick StartA comprehensive web-based Parkinson's disease detection system that uses:- **Validation**: 80% quality threshold enforcement



### Prerequisites- **Voice Analysis**: 150+ audio features (MFCC, spectral, prosodic, voice quality)- **Processing**: Research-validated parameter settings



- Python 3.13+ installed- **Tremor Detection**: 200+ motion features (FFT, time/frequency domain, statistical)

- Microphone access

- Motion sensors (mobile device recommended)- **Ensemble ML**: 4 algorithms (SVM, Random Forest, Gradient Boosting, XGBoost)## 🛠️ Installation & Setup



### Installation- **Real-time Analysis**: 3-5 second processing with 100% training accuracy



**1. Clone the repository:**### Local Development

```bash

git clone https://github.com/chaman2003/parkinson-detection.git### Technology Stack

cd parkinson-detection

```1. **Clone the repository**



**2. Create virtual environment (recommended):****Backend:**   ```bash

```bash

python -m venv .venv- Python 3.13+ with Flask   git clone https://github.com/chaman2003/parkinson-detection.git

.venv\Scripts\activate  # Windows

source .venv/bin/activate  # Linux/Mac- Scikit-learn + XGBoost   cd parkinson-detection

```

- Librosa (audio processing)   ```

**3. Install dependencies:**

```bash- NumPy + SciPy (scientific computing)

pip install -r backend/requirements.txt

```2. **Backend Setup**



### Running the Application**Frontend:**   ```bash



**Start the backend:**- Vanilla JavaScript   cd backend

```bash

cd backend- Web Audio API   python -m venv .venv

python app.py

```- Device Motion API   .venv\Scripts\activate  # Windows



**What happens:**- Progressive Web App (PWA)   pip install -r requirements.txt

- **First run**: Automatically generates 1000 synthetic training samples and trains ML models (~30 seconds)

- **Subsequent runs**: Loads cached models instantly   python app.py

- **Server**: Starts on `http://localhost:5000`

---   ```

**Access the frontend:**

1. Open `frontend/index.html` in your browser

2. Allow microphone and motion permissions

3. Start testing!## 🚀 Quick Start3. **Frontend Setup**



---   - Open `index.html` with Live Server in VS Code



## 🏗️ System Architecture### Prerequisites   - Or serve with: `python -m http.server 8080`



### Directory Structure



```- Python 3.13+ installed## 🌐 API Endpoints

parkinson/

├── README.md                   # This comprehensive guide- Microphone access

├── backend/

│   ├── app.py                  # Main server with integrated training- Motion sensors (mobile device recommended)- `GET /api/health` - Health check

│   ├── ml_models.py            # Ensemble ML pipeline

│   ├── audio_features.py       # 150+ audio feature extraction- `POST /api/analyze` - Main analysis endpoint

│   ├── tremor_features.py      # 200+ motion feature extraction

│   ├── requirements.txt        # Python dependencies### Installation- `POST /api/demo` - Demo analysis with mock results

│   ├── models/                 # Trained model files (auto-generated)

│   │   ├── voice_model.pkl- `GET /api/models/info` - Model information

│   │   ├── voice_scaler.pkl

│   │   ├── tremor_model.pkl**1. Clone the repository:**

│   │   └── tremor_scaler.pkl

│   └── uploads/                # Temporary audio files```bash## 📄 License

└── frontend/

    ├── index.html              # Main web applicationgit clone https://github.com/chaman2003/parkinson-detection.git

    ├── app.js                  # Frontend logic

    ├── styles.css              # UI stylingcd parkinson-detectionThis project is licensed under the MIT License.

    ├── quality-indicators.css  # Quality indicators styling

    ├── sw.js                   # Service worker (PWA)```

    ├── manifest.json           # PWA manifest

    ├── package.json            # NPM configuration**⚠️ Medical Disclaimer**: This application is for research and educational purposes only.

    ├── favicon.ico             # Browser icon

    └── assets/                 # Icons and images**2. Create virtual environment (recommended):**

```

```bash## Project Structure

### Processing Pipeline

python -m venv .venv

```

User Input (Audio + Motion).venv\Scripts\activate  # Windows```

    ↓

Feature Extractionsource .venv/bin/activate  # Linux/Macparkinson/

    ├─→ Audio: 150+ features (librosa)

    └─→ Motion: 200+ features (FFT, stats)```├── frontend/                 # PWA Frontend

    ↓

Feature Scaling (StandardScaler)│   ├── index.html           # Main HTML file

    ↓

Ensemble ML Inference**3. Install dependencies:**│   ├── styles.css           # CSS styles

    ├─→ SVM (RBF kernel)

    ├─→ Random Forest (200 trees)```bash│   ├── app.js               # Main JavaScript logic

    ├─→ Gradient Boosting (150 estimators)

    └─→ XGBoost (200 estimators)pip install -r backend/requirements.txt│   ├── manifest.json        # PWA manifest

    ↓

Soft Voting (Probability Averaging)```│   ├── sw.js                # Service worker

    ↓

Prediction + Confidence Scores│   └── assets/              # Images and icons

```

### Running the Application└── backend/                 # Python Backend

---

    ├── app.py               # Flask/FastAPI server

## ✨ Features

**Single Command Setup:**    ├── ml_models.py         # Machine learning pipeline

### Audio Analysis (150+ Features)

```bash    ├── requirements.txt     # Python dependencies

**MFCC (Mel-Frequency Cepstral Coefficients):**

- 13 coefficients + deltas + delta-deltascd backend    └── models/              # Trained model files

- Voice characteristic fingerprinting

python app.py```

**Spectral Features:**

- Spectral centroid, bandwidth, rolloff```

- Zero-crossing rate

- Frequency domain analysis## Setup Instructions



**Prosodic Features:****What happens:**

- Fundamental frequency (F0) extraction

- Jitter (frequency perturbation)- **First run**: Automatically generates 1000 synthetic training samples and trains ML models (~30 seconds)### Backend Setup

- Shimmer (amplitude perturbation)

- Voice tremor indicators- **Subsequent runs**: Loads cached models instantly1. Navigate to the backend directory



**Voice Quality:**- **Server**: Starts on `http://localhost:5000`2. Install dependencies: `pip install -r requirements.txt`

- Harmonic-to-Noise Ratio (HNR)

- Signal-to-noise estimation3. Run the server: `python app.py`

- Voice quality metrics

**Access the frontend:**

**Temporal Features:**

- Onset strength detection1. Open `index.html` in your browser### Frontend Setup

- Tempo estimation

- Rhythmic patterns2. Allow microphone and motion permissions1. Open the frontend directory



**Harmonic Features:**3. Start testing!2. Serve the files using a local web server

- Chroma features (12 pitch classes)

- Tonnetz (tonal centroid features)3. Access the app through HTTPS (required for PWA features)

- Harmonic structure analysis

---

### Tremor Analysis (200+ Features)

## Usage

**Frequency Domain:**

- FFT analysis (0-25 Hz range)## 🏗️ System Architecture

- 4-6 Hz tremor band detection

- Power spectral density1. Open the PWA on your smartphone

- Dominant frequency identification

### Directory Structure2. Grant microphone and motion sensor permissions

**Time Domain:**

- Acceleration statistics (X, Y, Z axes)3. Follow the instructions to record voice and tremor data

- Mean, standard deviation, variance

- Min/max amplitude detection```4. View the analysis results with confidence scores



**Statistical Features:**parkinson/

- Skewness (asymmetry)

- Kurtosis (tail heaviness)├── backend/## Technology Stack

- Percentiles (25th, 50th, 75th)

│   ├── app.py                  # Main server with integrated training

**Tremor-Specific:**

- Tremor amplitude calculation│   ├── ml_models.py            # Ensemble ML pipeline- **Frontend**: HTML5, CSS3, JavaScript (ES6+)

- Zero-crossing rate

- Jerk analysis (rate of acceleration change)│   ├── audio_features.py       # 150+ audio feature extraction- **Backend**: Python, Flask/FastAPI



**Movement Patterns:**│   ├── tremor_features.py      # 200+ motion feature extraction- **ML Libraries**: scikit-learn, XGBoost, NumPy, SciPy

- Autocorrelation analysis

- Movement intensity variation│   ├── requirements.txt        # Python dependencies- **Audio Processing**: Web Audio API, librosa (backend)

- Directional statistics

│   ├── models/                 # Trained model files (auto-generated)- **Motion Detection**: DeviceMotionEvent API

**Stability Metrics:**│   │   ├── voice_model.pkl

- Postural stability index│   │   ├── voice_scaler.pkl

- Path length calculation│   │   ├── tremor_model.pkl

- Movement smoothness│   │   └── tremor_scaler.pkl

│   └── uploads/                # Temporary audio files

### ML Ensemble├── index.html                  # Main frontend application

├── app.js                      # Frontend logic

**Algorithms:**├── styles.css                  # UI styling

1. **SVM (Support Vector Machine)**├── sw.js                       # Service worker (PWA)

   - RBF kernel├── manifest.json               # PWA manifest

   - Gamma: 'scale'└── assets/                     # Icons and images

   - Probability estimates enabled```



2. **Random Forest**### Processing Pipeline

   - 200 trees

   - Max depth: 20```

   - Min samples split: 5User Input (Audio + Motion)

    ↓

3. **Gradient Boosting**Feature Extraction

   - 150 estimators    ├─→ Audio: 150+ features (librosa)

   - Learning rate: 0.1    └─→ Motion: 200+ features (FFT, stats)

   - Max depth: 5    ↓

Feature Scaling (StandardScaler)

4. **XGBoost**    ↓

   - 200 estimatorsEnsemble ML Inference

   - Learning rate: 0.1    ├─→ SVM (RBF kernel)

   - Max depth: 5    ├─→ Random Forest (200 trees)

   - GPU acceleration support    ├─→ Gradient Boosting (150 estimators)

    └─→ XGBoost (200 estimators)

**Training:**    ↓

- **Voting**: Soft voting (probability averaging)Soft Voting (Probability Averaging)

- **Cross-Validation**: 5-fold    ↓

- **Scaling**: StandardScaler normalizationPrediction + Confidence Scores

- **Persistence**: Pickle-based model caching```



------



## 📡 API Documentation## ✨ Features



### Base URL### Audio Analysis (150+ Features)

```

http://localhost:5000/api**MFCC (Mel-Frequency Cepstral Coefficients):**

```- 13 coefficients + deltas + delta-deltas

- Voice characteristic fingerprinting

### Endpoints

**Spectral Features:**

#### 1. Health Check- Spectral centroid, bandwidth, rolloff

```http- Zero-crossing rate

GET /api/health- Frequency domain analysis

```

**Prosodic Features:**

**Response:**- Fundamental frequency (F0) extraction

```json- Jitter (frequency perturbation)

{- Shimmer (amplitude perturbation)

  "status": "healthy",- Voice tremor indicators

  "timestamp": "2025-10-01T10:30:00.000Z",

  "version": "1.0.0"**Voice Quality:**

}- Harmonic-to-Noise Ratio (HNR)

```- Signal-to-noise estimation

- Voice quality metrics

#### 2. Analyze (Main ML Endpoint)

```http**Temporal Features:**

POST /api/analyze- Onset strength detection

Content-Type: multipart/form-data- Tempo estimation

```- Rhythmic patterns



**Parameters:****Harmonic Features:**

- `audio` (file): Audio recording (WebM/WAV/MP3)- Chroma features (12 pitch classes)

- `motion_data` (JSON string): Array of motion samples- Tonnetz (tonal centroid features)

- Harmonic structure analysis

**Motion Data Format:**

```json### Tremor Analysis (200+ Features)

[

  {**Frequency Domain:**

    "timestamp": 1000,- FFT analysis (0-25 Hz range)

    "accelerationX": 0.5,- 4-6 Hz tremor band detection

    "accelerationY": -0.3,- Power spectral density

    "accelerationZ": 9.8,- Dominant frequency identification

    "rotationAlpha": 45.2,

    "rotationBeta": 12.5,**Time Domain:**

    "rotationGamma": -5.3- Acceleration statistics (X, Y, Z axes)

  }- Mean, standard deviation, variance

]- Min/max amplitude detection

```

**Statistical Features:**

**Response:**- Skewness (asymmetry)

```json- Kurtosis (tail heaviness)

{- Percentiles (25th, 50th, 75th)

  "prediction": "Not Affected",

  "confidence": 0.85,**Tremor-Specific:**

  "voice_confidence": 0.82,- Tremor amplitude calculation

  "tremor_confidence": 0.88,- Zero-crossing rate

  "features": {- Jerk analysis (rate of acceleration change)

    "Voice Stability": 0.65,

    "Tremor Frequency": 0.78,**Movement Patterns:**

    "Voice Quality": 0.72,- Autocorrelation analysis

    "Postural Stability": 0.68,- Movement intensity variation

    "Vocal Tremor": 0.45,- Directional statistics

    "Motion Variability": 0.73

  },**Stability Metrics:**

  "metadata": {- Postural stability index

    "processing_time": 3.2,- Path length calculation

    "audio_features_count": 150,- Movement smoothness

    "tremor_features_count": 200,

    "model_type": "ensemble_ml"### ML Ensemble

  }

}**Algorithms:**

```1. **SVM (Support Vector Machine)**

   - RBF kernel

#### 3. Model Information   - Gamma: 'scale'

```http   - Probability estimates enabled

GET /api/models/info

```2. **Random Forest**

   - 200 trees

**Response:**   - Max depth: 20

```json   - Min samples split: 5

{

  "voice_model": {3. **Gradient Boosting**

    "type": "VotingClassifier",   - 150 estimators

    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],   - Learning rate: 0.1

    "n_features": 150,   - Max depth: 5

    "trained": true

  },4. **XGBoost**

  "tremor_model": {   - 200 estimators

    "type": "VotingClassifier",   - Learning rate: 0.1

    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],   - Max depth: 5

    "n_features": 200,   - GPU acceleration support

    "trained": true

  }**Training:**

}- **Voting**: Soft voting (probability averaging)

```- **Cross-Validation**: 5-fold

- **Scaling**: StandardScaler normalization

### Example Usage- **Persistence**: Pickle-based model caching



**cURL:**---

```bash

curl -X POST http://localhost:5000/api/analyze \## 📡 API Documentation

  -F "audio=@recording.webm" \

  -F "motion_data=[{\"timestamp\":1000,\"accelerationX\":0.5,...}]"### Base URL

``````

http://localhost:5000/api

**Python:**```

```python

import requests### Endpoints



files = {'audio': open('recording.webm', 'rb')}#### 1. Health Check

data = {'motion_data': '[{"timestamp":1000,"accelerationX":0.5,...}]'}```http

GET /api/health

response = requests.post('http://localhost:5000/api/analyze', files=files, data=data)```

print(response.json())

```**Response:**

```json

**JavaScript:**{

```javascript  "status": "healthy",

const formData = new FormData();  "timestamp": "2025-10-01T10:30:00.000Z",

formData.append('audio', audioBlob, 'recording.webm');  "version": "1.0.0"

formData.append('motion_data', JSON.stringify(motionData));}

```

const response = await fetch('http://localhost:5000/api/analyze', {

  method: 'POST',#### 2. Analyze (Main ML Endpoint)

  body: formData```http

});POST /api/analyze

const result = await response.json();Content-Type: multipart/form-data

``````



---**Parameters:**

- `audio` (file): Audio recording (WebM/WAV/MP3)

## 💻 Development- `motion_data` (JSON string): Array of motion samples



### Training Your Own Models**Motion Data Format:**

```json

**Using Real Data:**[

  {

1. **Prepare your dataset:**    "timestamp": 1000,

   ```python    "accelerationX": 0.5,

   # Format: (samples, features)    "accelerationY": -0.3,

   voice_data = np.array([...])  # Shape: (n_samples, 150)    "accelerationZ": 9.8,

   tremor_data = np.array([...])  # Shape: (n_samples, 200)    "rotationAlpha": 45.2,

   labels = np.array([0, 1, 0, ...])  # 0=healthy, 1=affected    "rotationBeta": 12.5,

   ```    "rotationGamma": -5.3

  }

2. **Replace training function in `backend/app.py`:**]

   ```python```

   def generate_training_data():

       # Load your real dataset here**Response:**

       voice_data = load_voice_data()```json

       tremor_data = load_tremor_data(){

       labels = load_labels()  "prediction": "Not Affected",

       return voice_data, tremor_data, labels  "confidence": 0.85,

   ```  "voice_confidence": 0.82,

  "tremor_confidence": 0.88,

3. **Delete cached models and retrain:**  "features": {

   ```bash    "Voice Stability": 0.65,

   rm backend/models/*.pkl    "Tremor Frequency": 0.78,

   python backend/app.py    "Voice Quality": 0.72,

   ```    "Postural Stability": 0.68,

    "Vocal Tremor": 0.45,

### Customizing ML Pipeline    "Motion Variability": 0.73

  },

**Edit `backend/ml_models.py`:**  "metadata": {

    "processing_time": 3.2,

```python    "audio_features_count": 150,

# Change hyperparameters    "tremor_features_count": 200,

svm = SVC(kernel='rbf', C=10.0, gamma='auto')  # Adjust C and gamma    "model_type": "ensemble_ml"

rf = RandomForestClassifier(n_estimators=500)  # More trees  }

}

# Add new algorithms```

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100, 50))#### 3. Model Information

``````http

GET /api/models/info

### Adding New Features```



**Audio features (`backend/audio_features.py`):****Response:**

```python```json

def extract_custom_feature(self, y, sr):{

    # Your feature extraction logic  "voice_model": {

    return feature_value    "type": "VotingClassifier",

```    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],

    "n_features": 150,

**Motion features (`backend/tremor_features.py`):**    "trained": true

```python  },

def extract_custom_motion_feature(self, data):  "tremor_model": {

    # Your motion analysis logic    "type": "VotingClassifier",

    return feature_value    "estimators": ["SVM", "RandomForest", "GradientBoosting", "XGBoost"],

```    "n_features": 200,

    "trained": true

### Frontend Customization  }

}

**Modify UI (`frontend/index.html`):**```

- Update HTML structure

- Change recording durations### Example Usage

- Add new test modes

**cURL:**

**Update logic (`frontend/app.js`):**```bash

- Modify API endpointscurl -X POST http://localhost:5000/api/analyze \

- Change data processing  -F "audio=@recording.webm" \

- Update result visualization  -F "motion_data=[{\"timestamp\":1000,\"accelerationX\":0.5,...}]"

```

---

**Python:**

## 🧪 Model Information```python

import requests

### Training Performance

files = {'audio': open('recording.webm', 'rb')}

**Synthetic Data Results:**data = {'motion_data': '[{"timestamp":1000,"accelerationX":0.5,...}]'}

- **Voice Model CV Accuracy**: 100% (5-fold cross-validation)

- **Tremor Model CV Accuracy**: 100% (5-fold cross-validation)response = requests.post('http://localhost:5000/api/analyze', files=files, data=data)

- **Training Samples**: 1000 (500 healthy, 500 affected)print(response.json())

- **Training Time**: ~30 seconds (first run only)```



**Note:** These accuracies are based on synthetic data with clear separation. Real-world clinical data will have lower accuracies.**JavaScript:**

```javascript

### Model Filesconst formData = new FormData();

formData.append('audio', audioBlob, 'recording.webm');

Models are automatically saved to `backend/models/`:formData.append('motion_data', JSON.stringify(motionData));

- `voice_model.pkl` - Trained voice classifier

- `voice_scaler.pkl` - Voice feature scalerconst response = await fetch('http://localhost:5000/api/analyze', {

- `tremor_model.pkl` - Trained tremor classifier  method: 'POST',

- `tremor_scaler.pkl` - Tremor feature scaler  body: formData

});

**File sizes:**const result = await response.json();

- Each model: ~5-10 MB```

- Total storage: ~20-40 MB

---

### Feature Importance

## 💻 Development

**Top Voice Features:**

1. Jitter (frequency perturbation)### Training Your Own Models

2. Shimmer (amplitude perturbation)

3. MFCC coefficients**Using Real Data:**

4. Harmonic-to-Noise Ratio

5. Fundamental frequency variation1. **Prepare your dataset:**

   ```python

**Top Tremor Features:**   # Format: (samples, features)

1. 4-6 Hz power spectral density   voice_data = np.array([...])  # Shape: (n_samples, 150)

2. Dominant frequency   tremor_data = np.array([...])  # Shape: (n_samples, 200)

3. Acceleration variance   labels = np.array([0, 1, 0, ...])  # 0=healthy, 1=affected

4. Movement jerk   ```

5. Postural stability metrics

2. **Replace training function in `backend/app.py`:**

---   ```python

   def generate_training_data():

## 🔧 Troubleshooting       # Load your real dataset here

       voice_data = load_voice_data()

### Installation Issues       tremor_data = load_tremor_data()

       labels = load_labels()

**"No module named 'X'"**       return voice_data, tremor_data, labels

```bash   ```

pip install -r backend/requirements.txt

```3. **Delete cached models and retrain:**

   ```bash

**"Python not found"**   rm backend/models/*.pkl

- Install Python 3.13+ from [python.org](https://www.python.org)   python backend/app.py

- Add Python to PATH during installation   ```



**Virtual environment issues**### Customizing ML Pipeline

```bash

# Recreate venv**Edit `backend/ml_models.py`:**

rm -rf .venv

python -m venv .venv```python

.venv\Scripts\activate# Change hyperparameters

pip install -r backend/requirements.txtsvm = SVC(kernel='rbf', C=10.0, gamma='auto')  # Adjust C and gamma

```rf = RandomForestClassifier(n_estimators=500)  # More trees



### Runtime Issues# Add new algorithms

from sklearn.neural_network import MLPClassifier

**"Port 5000 already in use"**mlp = MLPClassifier(hidden_layer_sizes=(100, 50))

```bash```

# Windows: Find and kill process

netstat -ano | findstr :5000### Adding New Features

taskkill /PID <process_id> /F

**Audio features (`backend/audio_features.py`):**

# Or change port in app.py```python

app.run(debug=True, port=5001)def extract_custom_feature(self, y, sr):

```    # Your feature extraction logic

    return feature_value

**"Training takes too long"**```

- Normal for first run (~30 seconds)

- Reduce samples in `generate_training_data()`:**Motion features (`backend/tremor_features.py`):**

  ```python```python

  n_samples = 100  # Instead of 1000def extract_custom_motion_feature(self, data):

  ```    # Your motion analysis logic

    return feature_value

**"Low prediction accuracy"**```

- Current models use synthetic data

- Train with real clinical data for production### Frontend Customization

- Collect diverse patient samples

- Validate with cross-dataset testing**Modify UI (`index.html`):**

- Update HTML structure

### Frontend Issues- Change recording durations

- Add new test modes

**"Microphone not working"**

- Check browser permissions (camera icon in address bar)**Update logic (`app.js`):**

- Use HTTPS or localhost only (WebRTC requirement)- Modify API endpoints

- Try different browser (Chrome/Edge recommended)- Change data processing

- Update result visualization

**"Motion sensors not working"**

- Use mobile device (gyroscope required)---

- Enable motion permissions in browser

- Some laptops have accelerometers## 🧪 Model Information



**"CORS errors"**### Training Performance

- Ensure Flask-CORS is installed

- Check backend is running on localhost:5000**Synthetic Data Results:**

- Clear browser cache- **Voice Model CV Accuracy**: 100% (5-fold cross-validation)

- **Tremor Model CV Accuracy**: 100% (5-fold cross-validation)

### Model Issues- **Training Samples**: 1000 (500 healthy, 500 affected)

- **Training Time**: ~30 seconds (first run only)

**"Models not loading"**

```bash**Note:** These accuracies are based on synthetic data with clear separation. Real-world clinical data will have lower accuracies.

# Check if models exist

ls backend/models/### Model Files



# Delete and retrainModels are automatically saved to `backend/models/`:

rm backend/models/*.pkl- `voice_model.pkl` - Trained voice classifier

python backend/app.py- `voice_scaler.pkl` - Voice feature scaler

```- `tremor_model.pkl` - Trained tremor classifier

- `tremor_scaler.pkl` - Tremor feature scaler

**"Feature extraction errors"**

- Check audio file format (WebM/WAV/MP3 supported)**File sizes:**

- Ensure audio is not empty or corrupted- Each model: ~5-10 MB

- Verify motion data is valid JSON array- Total storage: ~20-40 MB



**"Inconsistent predictions"**### Feature Importance

- Models trained on synthetic data

- Need real clinical data for reliability**Top Voice Features:**

- Consider ensemble temperature tuning1. Jitter (frequency perturbation)

- Add prediction confidence thresholds2. Shimmer (amplitude perturbation)

3. MFCC coefficients

---4. Harmonic-to-Noise Ratio

5. Fundamental frequency variation

## 📊 System Requirements

**Top Tremor Features:**

### Minimum Requirements1. 4-6 Hz power spectral density

2. Dominant frequency

**Hardware:**3. Acceleration variance

- CPU: Dual-core 2.0 GHz4. Movement jerk

- RAM: 2 GB5. Postural stability metrics

- Storage: 500 MB

- Microphone + motion sensors---



**Software:**## 🔧 Troubleshooting

- Python 3.13+

- Modern browser (Chrome 90+, Edge 90+, Firefox 88+)### Installation Issues

- Windows/Linux/macOS

**"No module named 'X'"**

### Recommended Requirements```bash

pip install -r backend/requirements.txt

**Hardware:**```

- CPU: Quad-core 3.0 GHz

- RAM: 4 GB**"Python not found"**

- Storage: 1 GB- Install Python 3.13+ from [python.org](https://www.python.org)

- SSD for faster model loading- Add Python to PATH during installation



**Software:****Virtual environment issues**

- Python 3.13+```bash

- Chrome/Edge latest# Recreate venv

- GPU support (CUDA) for XGBoost accelerationrm -rf .venv

python -m venv .venv

### Dependencies.venv\Scripts\activate

pip install -r backend/requirements.txt

**Python Packages:**```

```

Flask>=3.0.0### Runtime Issues

Flask-CORS>=4.0.0

numpy>=1.24.0**"Port 5000 already in use"**

pandas>=2.0.0```bash

scikit-learn>=1.3.0# Windows: Find and kill process

xgboost>=2.0.0netstat -ano | findstr :5000

librosa>=0.10.0taskkill /PID <process_id> /F

scipy>=1.11.0

```# Or change port in app.py

app.run(debug=True, port=5001)

**Browser APIs:**```

- Web Audio API

- MediaRecorder API**"Training takes too long"**

- DeviceMotion API- Normal for first run (~30 seconds)

- Fetch API- Reduce samples in `generate_training_data()`:

- Web Workers (PWA)  ```python

  n_samples = 100  # Instead of 1000

---  ```



## 🚀 Production Deployment**"Low prediction accuracy"**

- Current models use synthetic data

### For Clinical Use- Train with real clinical data for production

- Collect diverse patient samples

**1. Collect Real Data:**- Validate with cross-dataset testing

- Collaborate with medical institutions

- Get ethical approval and patient consent### Frontend Issues

- Record voice + motion from diagnosed patients

- Include healthy control group**"Microphone not working"**

- Balance dataset (equal positive/negative samples)- Check browser permissions (camera icon in address bar)

- Use HTTPS or localhost only (WebRTC requirement)

**2. Retrain Models:**- Try different browser (Chrome/Edge recommended)

```python

# Load your clinical dataset**"Motion sensors not working"**

voice_features = extract_voice_features(audio_files)- Use mobile device (gyroscope required)

tremor_features = extract_tremor_features(motion_data)- Enable motion permissions in browser

labels = patient_diagnoses- Some laptops have accelerometers



# Train with your data**"CORS errors"**

pipeline = ParkinsonMLPipeline()- Ensure Flask-CORS is installed

pipeline.train_models(voice_features, labels, tremor_features, labels)- Check backend is running on localhost:5000

pipeline.save_models('models/')- Clear browser cache

```

### Model Issues

**3. Validate Performance:**

- Use holdout test set (20% of data)**"Models not loading"**

- Calculate sensitivity/specificity```bash

- Test with cross-institutional data# Check if models exist

- Clinical trial validationls backend/models/



**4. Deploy Server:**# Delete and retrain

```bashrm backend/models/*.pkl

# Use production WSGI serverpython backend/app.py

pip install gunicorn```

gunicorn -w 4 -b 0.0.0.0:5000 app:app

```**"Feature extraction errors"**

- Check audio file format (WebM/WAV/MP3 supported)

**5. Security:**- Ensure audio is not empty or corrupted

- Enable HTTPS (SSL/TLS certificates)- Verify motion data is valid JSON array

- Implement user authentication

- Add data encryption**"Inconsistent predictions"**

- HIPAA compliance for medical data- Models trained on synthetic data

- Regular security audits- Need real clinical data for reliability

- Consider ensemble temperature tuning

### Scaling- Add prediction confidence thresholds



**Horizontal Scaling:**---

- Load balancer (nginx)

- Multiple Flask instances## 📊 System Requirements

- Redis for session management

- Database for result storage### Minimum Requirements



**Optimization:****Hardware:**

- Model quantization (reduce size)- CPU: Dual-core 2.0 GHz

- Feature selection (reduce computation)- RAM: 2 GB

- Caching predictions- Storage: 500 MB

- Async processing with Celery- Microphone + motion sensors



---**Software:**

- Python 3.13+

## 📝 License- Modern browser (Chrome 90+, Edge 90+, Firefox 88+)

- Windows/Linux/macOS

This project is for educational and research purposes. For clinical use, consult with medical professionals and obtain necessary certifications.

### Recommended Requirements

---

**Hardware:**

## 🤝 Contributing- CPU: Quad-core 3.0 GHz

- RAM: 4 GB

Contributions welcome! Areas of interest:- Storage: 1 GB

- Real clinical data integration- SSD for faster model loading

- Deep learning models (CNN/RNN)

- Mobile app development**Software:**

- Additional biomarker analysis- Python 3.13+

- Multi-language support- Chrome/Edge latest

- GPU support (CUDA) for XGBoost acceleration

---

### Dependencies

## 📞 Support

**Python Packages:**

For issues or questions:```

1. Check [Troubleshooting](#troubleshooting) sectionFlask>=3.0.0

2. Review code comments in source filesFlask-CORS>=4.0.0

3. Open GitHub issue with error logsnumpy>=1.24.0

pandas>=2.0.0

---scikit-learn>=1.3.0

xgboost>=2.0.0

## 🙏 Acknowledgmentslibrosa>=0.10.0

scipy>=1.11.0

Built with research from:```

- Parkinson's Disease detection literature

- Voice analysis studies**Browser APIs:**

- Tremor frequency research- Web Audio API

- ML ensemble methods- MediaRecorder API

- DeviceMotion API

---- Fetch API

- Web Workers (PWA)

**Made with ❤️ for medical AI research**

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
