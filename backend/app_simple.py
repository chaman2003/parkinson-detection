from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import random
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

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
    Simplified analysis endpoint for testing
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if 'motion_data' not in request.form:
            return jsonify({'error': 'No motion data provided'}), 400

        # Get uploaded files and data
        audio_file = request.files['audio']
        motion_data_str = request.form['motion_data']
        test_mode = request.form.get('test_mode', 'both')  # 'voice', 'tremor', or 'both'
        
        # Validate audio file
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Parse motion data
        try:
            motion_data = json.loads(motion_data_str)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid motion data format'}), 400

        logger.info(f"Processing analysis request")
        logger.info(f"Audio file: {audio_file.filename}")
        logger.info(f"Motion data samples: {len(motion_data)}")
        logger.info(f"Test mode: {test_mode}")

        # Simulate processing time
        time.sleep(2)

        # Generate realistic mock results based on input data
        results = generate_realistic_results(len(motion_data), audio_file.filename, test_mode)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def generate_realistic_results(motion_samples, audio_filename, test_mode='both'):
    """Generate realistic results based on input data and test mode"""
    
    # Base prediction probabilities
    base_confidence = random.uniform(0.65, 0.90)
    
    # Adjust based on data quality (more samples = higher confidence)
    motion_quality = min(1.0, motion_samples / 1000.0) if motion_samples > 0 else 0.5
    audio_quality = 0.8 if audio_filename and audio_filename != 'empty.webm' else 0.5
    
    # Calculate individual confidences based on test mode
    if test_mode == 'voice':
        voice_confidence = base_confidence * (0.7 + 0.3 * audio_quality)
        tremor_confidence = 0.5  # Not tested
        overall_confidence = voice_confidence
    elif test_mode == 'tremor':
        voice_confidence = 0.5  # Not tested
        tremor_confidence = base_confidence * (0.7 + 0.3 * motion_quality)
        overall_confidence = tremor_confidence
    else:  # both
        voice_confidence = base_confidence * (0.7 + 0.3 * audio_quality)
        tremor_confidence = base_confidence * (0.7 + 0.3 * motion_quality)
        overall_confidence = 0.6 * voice_confidence + 0.4 * tremor_confidence
    
    # Determine prediction (lower threshold for demo)
    prediction = "Affected" if overall_confidence > 0.75 else "Not Affected"
    
    # Generate feature analysis based on test mode
    features = {}
    
    if test_mode == 'voice' or test_mode == 'both':
        features['Voice Stability'] = max(0.3, min(0.9, voice_confidence + random.uniform(-0.1, 0.1)))
        features['Speech Rhythm'] = max(0.3, min(0.9, voice_confidence * 0.9 + random.uniform(-0.1, 0.1)))
        features['Vocal Tremor'] = max(0.2, min(0.8, voice_confidence * 0.7 + random.uniform(-0.15, 0.15)))
    
    if test_mode == 'tremor' or test_mode == 'both':
        features['Tremor Frequency'] = max(0.2, min(0.8, tremor_confidence + random.uniform(-0.1, 0.1)))
        features['Motion Variability'] = max(0.3, min(0.9, tremor_confidence * 0.8 + random.uniform(-0.1, 0.1)))
        features['Postural Stability'] = max(0.4, min(0.9, tremor_confidence * 1.1 + random.uniform(-0.1, 0.1)))
    
    # Ensure all values are in valid range
    for key in features:
        features[key] = max(0.0, min(1.0, features[key]))
    
    return {
        'prediction': prediction,
        'confidence': max(0.5, min(0.95, overall_confidence)),
        'voice_confidence': max(0.5, min(0.95, voice_confidence)),
        'tremor_confidence': max(0.5, min(0.95, tremor_confidence)),
        'features': features,
        'metadata': {
            'processing_time': 2.0,
            'audio_duration': random.uniform(8.0, 12.0) if test_mode != 'tremor' else 0.0,
            'motion_samples': motion_samples if test_mode != 'voice' else 0,
            'model_version': '1.0.0 (simplified)',
            'test_mode': test_mode,
            'data_quality': {
                'motion_quality': motion_quality,
                'audio_quality': audio_quality
            }
        }
    }

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get information about available ML models"""
    return jsonify({
        'models': {
            'voice_analysis': {
                'type': 'simplified',
                'algorithms': ['Statistical Analysis'],
                'features': ['Basic Audio Features', 'Quality Metrics']
            },
            'tremor_analysis': {
                'type': 'simplified',
                'algorithms': ['Motion Analysis'],
                'features': ['Motion Quantity', 'Data Quality']
            }
        },
        'version': '1.0.0 (simplified)',
        'trained_date': '2025-09-30'
    })

@app.route('/api/demo', methods=['POST'])
def demo_analysis():
    """
    Demo endpoint that returns mock results for testing
    """
    try:
        # Simulate processing time
        time.sleep(1)
        
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
                'processing_time': 1.0,
                'audio_duration': random.uniform(8.0, 12.0),
                'motion_samples': random.randint(800, 1500),
                'model_version': '1.0.0 (demo)'
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
    print("Starting Parkinson's Detection API Server (Simplified Version)...")
    print("Available endpoints:")
    print("- GET  /api/health - Health check")
    print("- POST /api/analyze - Main analysis endpoint")
    print("- POST /api/demo - Demo analysis with mock results")
    print("- GET  /api/models/info - Model information")
    print("\nServer starting on http://localhost:5000")
    print("Frontend should be running on http://localhost:8080")
    print("\nNote: This is a simplified version for testing without heavy ML dependencies.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)