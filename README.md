#  Parkinson's Disease Detection System

Advanced AI-powered Parkinson's disease detection using voice and motion analysis.

##  Quick Links

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)

---

## Overview

The **Parkinson's Disease Detection System** is a web application that uses artificial intelligence and sensor technology to detect early signs of Parkinson's disease through voice and motion analysis.

### Key Capabilities

- **Real-time Sensor Analysis**: Device motion sensors and microphone
- **Ensemble ML Models**: SVM, Random Forest, Gradient Boosting, XGBoost
- **Comprehensive Reporting**: Detailed Excel exports
- **Progressive Web App**: Mobile-installable
- **Fast Processing**: Sub-second analysis

---

## Features

### Voice Analysis
- Pitch variation detection
- Voice quality metrics (jitter, shimmer)
- Spectral features analysis

### Motion Analysis
- Tremor detection (4-6 Hz band)
- Stability assessment
- Movement pattern analysis

---

## Technology Stack

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- Progressive Web App (PWA)
- Web Audio API & Device Motion API
- SheetJS for Excel export

### Backend
- Flask 2.3.3
- NumPy, SciPy, Pandas
- Librosa, Pydub (audio processing)
- scikit-learn, XGBoost (ML models)

### Tunneling
- **ngrok** with custom domain: ostensible-unvibrant-clarisa.ngrok-free.dev

---

## Installation

### Prerequisites

- Python 3.8+
- Modern Web Browser (Chrome 88+, Firefox 85+, Safari 14+)
- ngrok (https://ngrok.com/download)

### Quick Start (Windows)

`powershell
.\run-locally.ps1
`

This starts everything automatically with ngrok custom domain.

### Manual Setup

1. Backend: cd backend && python app.py
2. Frontend: cd frontend && python server.py 8000
3. ngrok: 
grok http --domain=ostensible-unvibrant-clarisa.ngrok-free.dev 5000

---

## Usage

### Access Points

| Environment | URL |
|-------------|-----|
| Local Frontend | http://localhost:8000 |
| Local Backend | http://localhost:5000 |
| Mobile/Remote | https://ostensible-unvibrant-clarisa.ngrok-free.dev |

### Vercel Deployment

Set BACKEND_URL = https://ostensible-unvibrant-clarisa.ngrok-free.dev in Vercel environment variables.

---

## API Documentation

### Base URL
- Local: http://localhost:5000/api
- Production: https://ostensible-unvibrant-clarisa.ngrok-free.dev/api

### Health Check
GET /api/health

### ML Analysis
POST /api/analyze

### Model Information
GET /api/models/info

---

## License

MIT License - see LICENSE file for details

---

Made with  for Parkinson's research
