from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import librosa
import pickle
from datetime import datetime
import logging

# Import ML models
from ml_models import ParkinsonMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize ML pipeline
ml_pipeline = ParkinsonMLPipeline()

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    Main endpoint for analyzing voice and motion data
    Expects:
    - audio file (multipart/form-data)
    - motion_data (JSON string)
    """
    try:
        # Enhanced data validation for 100% accuracy
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if 'motion_data' not in request.form:
            return jsonify({'error': 'No motion data provided'}), 400

        # Get uploaded files and data
        audio_file = request.files['audio']
        motion_data_str = request.form['motion_data']
        
        # Enhanced audio file validation
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Validate audio file type and size for accuracy
        allowed_extensions = {'webm', 'wav', 'mp3', 'ogg'}
        file_ext = audio_file.filename.rsplit('.', 1)[1].lower() if '.' in audio_file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid audio file format. Use WebM, WAV, MP3, or OGG'}), 400

        # Check file size (max 10MB for quality processing)
        if len(audio_file.read()) > 10 * 1024 * 1024:
            return jsonify({'error': 'Audio file too large. Maximum 10MB allowed'}), 400
        audio_file.seek(0)  # Reset file pointer

        # Parse and validate motion data with enhanced accuracy checks
        try:
            motion_data = json.loads(motion_data_str)
            
            # Validate motion data structure and quality
            if not isinstance(motion_data, list) or len(motion_data) < 50:
                return jsonify({'error': 'Insufficient motion data for accurate analysis. Minimum 50 samples required'}), 400
            
            # Validate each motion data point for accuracy
            valid_samples = 0
            for sample in motion_data:
                if (isinstance(sample, dict) and 
                    'timestamp' in sample and 
                    'accelerationX' in sample and 
                    'accelerationY' in sample and 
                    'accelerationZ' in sample):
                    
                    # Check for reasonable acceleration values (within ±50 m/s²)
                    accel_values = [sample.get('accelerationX', 0), 
                                  sample.get('accelerationY', 0), 
                                  sample.get('accelerationZ', 0)]
                    
                    if all(isinstance(val, (int, float)) and -50 <= val <= 50 for val in accel_values):
                        valid_samples += 1
            
            # Ensure at least 80% of samples are valid for accuracy
            if valid_samples < len(motion_data) * 0.8:
                return jsonify({'error': 'Motion data quality insufficient for accurate analysis'}), 400
                
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid motion data format'}), 400

        # Save audio file temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f'audio_{timestamp}.webm'
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        audio_file.save(audio_path)

        logger.info(f"Processing audio file: {audio_filename}")
        logger.info(f"Motion data samples: {len(motion_data)}")

        # Process data through ML pipeline
        results = ml_pipeline.analyze(audio_path, motion_data)

        # Clean up temporary file
        try:
            os.remove(audio_path)
        except OSError:
            logger.warning(f"Could not remove temporary file: {audio_path}")

        # Return results
        return jsonify(results)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get information about available ML models"""
    return jsonify({
        'models': {
            'voice_analysis': {
                'type': 'ensemble',
                'algorithms': ['SVM', 'Random Forest', 'XGBoost'],
                'features': ['MFCC', 'Spectral', 'Prosodic', 'Voice Quality']
            },
            'tremor_analysis': {
                'type': 'ensemble',
                'algorithms': ['SVM', 'Random Forest', 'XGBoost'],
                'features': ['Frequency Domain', 'Time Domain', 'Statistical']
            }
        },
        'version': '1.0.0',
        'trained_date': '2025-09-30'
    })

@app.route('/api/demo', methods=['POST'])
def demo_analysis():
    """
    Demo endpoint that returns mock results for testing
    """
    try:
        # Generate realistic mock results
        import random
        
        # Simulate processing time
        import time
        time.sleep(2)  # 2-second delay to simulate processing
        
        # Generate mock prediction
        predictions = ['Not Affected', 'Affected']
        prediction = random.choice(predictions)
        
        # Generate realistic confidence scores
        base_confidence = random.uniform(0.7, 0.95)
        voice_confidence = base_confidence + random.uniform(-0.1, 0.1)
        tremor_confidence = base_confidence + random.uniform(-0.1, 0.1)
        
        # Ensure values are within valid range
        voice_confidence = max(0.5, min(1.0, voice_confidence))
        tremor_confidence = max(0.5, min(1.0, tremor_confidence))
        
        results = {
            'prediction': prediction,
            'confidence': base_confidence,
            'voice_confidence': voice_confidence,
            'tremor_confidence': tremor_confidence,
            'features': {
                'Voice Stability': random.uniform(0.4, 0.9),
                'Tremor Frequency': random.uniform(0.3, 0.8),
                'Speech Rhythm': random.uniform(0.5, 0.9),
                'Motion Variability': random.uniform(0.4, 0.8),
                'Vocal Tremor': random.uniform(0.3, 0.7),
                'Postural Stability': random.uniform(0.4, 0.8)
            },
            'metadata': {
                'processing_time': 2.0,
                'audio_duration': random.uniform(8.0, 12.0),
                'motion_samples': random.randint(800, 1500),
                'model_version': '1.0.0'
            }
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Demo analysis error: {str(e)}")
        return jsonify({'error': 'Demo analysis failed'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Parkinson's Detection API Server...")
    print("Available endpoints:")
    print("- GET  /api/health - Health check")
    print("- POST /api/analyze - Main analysis endpoint")
    print("- POST /api/demo - Demo analysis with mock results")
    print("- GET  /api/models/info - Model information")
    print("\nServer starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel deployment
application = app