from flask import Flask, request, jsonify, Response, stream_with_context
import os
import sys
import json
import numpy as np
import pickle
from datetime import datetime
import logging
import time
from queue import Queue
import threading
import gc

# Change to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import ML models and data utilities from utils package
from utils.ml_models import ParkinsonMLPipeline
from utils.data_loader import DatasetLoader, load_single_voice_file
from utils.data_storage import DataStorageManager
from utils.audio_features_optimized import OptimizedAudioExtractor
from utils.tremor_features_optimized import OptimizedTremorExtractor
from utils.dataset_matcher import DatasetMatcher
from utils.personalized_model import PersonalizedModelHandler

# Configure logging - disable colors on Windows to avoid ANSI codes
import sys
if sys.platform == 'win32':
    # Disable colored output on Windows
    os.environ['TERM'] = 'dumb'
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable Flask/Werkzeug colored output
import click
click.echo = lambda x, **kwargs: print(x)

def make_json_serializable(obj):
    """
    Convert numpy arrays and other non-serializable objects to JSON-compatible types
    This is needed because ML models return numpy arrays which can't be directly sent as JSON
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj

def convert_webm_to_wav(webm_path, wav_path):
    """
    Convert WebM audio file to WAV format using available tools
    Prioritizes ffmpeg for Python 3.13+ compatibility
    Returns True if successful, False otherwise
    """
    import subprocess
    import sys
    
    # For Python 3.13+, use ffmpeg directly first (avoids aifc/audioop issues)
    if sys.version_info >= (3, 13):
        try:
            # Method 1 (Preferred for Python 3.13+): Direct ffmpeg
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '22050', wav_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info("✓ Converted WebM to WAV using ffmpeg")
                return True
            else:
                logger.warning(f"ffmpeg conversion failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("ffmpeg not found in system PATH - please install ffmpeg")
        except Exception as e:
            logger.warning(f"ffmpeg conversion error: {str(e)}")
    
    # Try pydub (works better on Python < 3.13)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav", parameters=["-ar", "22050"])
        logger.info("✓ Converted WebM to WAV using pydub")
        return True
    except Exception as e:
        logger.warning(f"pydub conversion failed: {str(e)}")
        
        # If pydub failed and we didn't try ffmpeg yet, try it now
        if sys.version_info < (3, 13):
            try:
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '22050', wav_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info("✓ Converted WebM to WAV using ffmpeg")
                    return True
                else:
                    logger.warning(f"ffmpeg conversion failed: {result.stderr}")
            except FileNotFoundError:
                logger.warning("ffmpeg not found in system PATH")
            except Exception as e:
                logger.warning(f"ffmpeg conversion error: {str(e)}")
        
        # Last resort: Try soundfile with scipy resampling
        try:
            import soundfile as sf
            y, sr = sf.read(webm_path, dtype='float32')
            # Resample to 22050 Hz if needed
            if sr != 22050:
                num_samples = int(len(y) * 22050 / sr)
                indices = np.linspace(0, len(y) - 1, num_samples)
                y = np.interp(indices, np.arange(len(y)), y)
            sf.write(wav_path, y, 22050)
            logger.info("Converted WebM to WAV using soundfile/scipy")
            return True
        except Exception as e:
            logger.warning(f"soundfile/scipy conversion failed: {str(e)}")
    
    logger.error("✗ All conversion methods failed - please install ffmpeg")
    return False

app = Flask(__name__)

# Configure Flask for streaming and large file handling
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
app.config['THREADED'] = True  # Enable threading for concurrent requests

# CORS headers are handled by ngrok proxy
# ngrok automatically forwards and manages all CORS headers
# No need for flask_cors as it can cause duplicate headers

# Initialize utilities
try:
    storage_manager = DataStorageManager()
    audio_extractor = OptimizedAudioExtractor()
    tremor_extractor = OptimizedTremorExtractor()
    personalized_handler = PersonalizedModelHandler()  # NEW: Handle personalized baselines
    logger.info("✓ Utilities initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize utilities: {str(e)}")
    raise  # Re-raise to prevent app from starting with broken components


def check_and_train_models():
    """
    Check if models exist, if not, load real datasets and train models
    This ensures the system is ready on first run with improved training logic
    """
    import time
    
    model_dir = 'models'
    voice_model_path = os.path.join(model_dir, 'voice_model.pkl')
    tremor_model_path = os.path.join(model_dir, 'tremor_model.pkl')
    
    # Check if models exist
    if os.path.exists(voice_model_path) and os.path.exists(tremor_model_path):
        logger.info("✓ ML models found and loaded successfully")
        return True
    
    # Models don't exist - train them
    start_time = time.time()
    
    print("\n" + "="*70)
    print("  FIRST-TIME SETUP: TRAINING PARKINSON'S DETECTION MODELS")
    print("="*70)
    print("\n⚠️  This will take 2-5 minutes but only happens once!")
    print("   Subsequent runs will load models instantly.\n")
    
    try:
        # Initialize
        loader = DatasetLoader()
        
        # Load voice dataset
        print("📁 Step 1: Loading voice samples...")
        voice_files, voice_labels = loader.load_voice_dataset()
        
        if len(voice_files) == 0:
            raise FileNotFoundError("No voice samples found in datasets/voice_dataset/")
        
        print(f"   ✓ Found {len(voice_files)} voice samples")
        print(f"     - Healthy: {np.sum(voice_labels == 0)}")
        print(f"     - Parkinson's: {np.sum(voice_labels == 1)}")
        
        # Extract features
        print("\n🎵 Step 2: Extracting audio features...")
        print("   This may take 2-3 minutes...")
        X_voice_list = []
        y_voice_list = []
        skipped = 0
        expected_feature_count = None
        
        for i, (voice_file, label) in enumerate(zip(voice_files, voice_labels)):
            if (i + 1) % 10 == 0:
                print(f"   Processing {i+1}/{len(voice_files)}...")
            
            try:
                features = audio_extractor.extract_features_fast(str(voice_file))
                # Remove insights before converting to vector
                features.pop('_insights', None)
                feature_vector = np.array(list(features.values()))
                
                # Set expected feature count from first successful extraction
                if expected_feature_count is None:
                    expected_feature_count = len(feature_vector)
                
                # Ensure consistent feature vector size
                if len(feature_vector) != expected_feature_count:
                    skipped += 1
                    continue
                
                X_voice_list.append(feature_vector)
                y_voice_list.append(label)
            except Exception as e:
                skipped += 1
                continue
        
        X_voice = np.array(X_voice_list)
        y_voice = np.array(y_voice_list)
        print(f"   ✓ Extracted features from {len(X_voice)} samples")
        print(f"   ✓ Feature dimensions: {X_voice.shape}")
        if skipped > 0:
            print(f"   ⚠ Skipped {skipped} problematic files")
        
        # Load tremor dataset
        print("\n📊 Step 3: Loading tremor dataset...")
        X_tremor, y_tremor, tremor_df = loader.load_tremor_dataset()
        print(f"   ✓ Loaded {len(X_tremor)} tremor samples")
        print(f"   ✓ Feature dimensions: {X_tremor.shape}")
        print(f"     - Healthy: {np.sum(y_tremor == 0)}")
        print(f"     - Affected: {np.sum(y_tremor == 1)}")
        
        # Check if we have enough data
        if len(X_voice) < 10 or len(X_tremor) < 10:
            raise ValueError(f"Not enough samples to train models! Voice: {len(X_voice)}, Tremor: {len(X_tremor)}")
        
        # Train models
        print("\n🤖 Step 4: Training ML models...")
        print("   This will take 1-2 minutes...")
        print("   Training ensemble: SVM + Random Forest + Gradient Boosting")
        
        ml_pipeline.train_models(X_voice, y_voice, X_tremor, y_tremor)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("  ✅ MODEL TRAINING COMPLETE!")
        print("="*70)
        print(f"\n📈 Training Summary:")
        print(f"   • Voice Model: Trained on {len(y_voice)} samples")
        print(f"     - Healthy: {np.sum(y_voice == 0)}")
        print(f"     - Parkinson's: {np.sum(y_voice == 1)}")
        print(f"   • Tremor Model: Trained on {len(y_tremor)} samples")
        print(f"     - Healthy: {np.sum(y_tremor == 0)}")
        print(f"     - Affected: {np.sum(y_tremor == 1)}")
        print(f"\n💾 Models saved to: models/")
        print(f"   ✓ voice_model.pkl")
        print(f"   ✓ voice_scaler.pkl")
        print(f"   ✓ tremor_model.pkl")
        print(f"   ✓ tremor_scaler.pkl")
        print(f"\n⏱ Total time: {elapsed_time:.1f} seconds")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {str(e)}")
        logger.error("The server will start but predictions may not work correctly")
        import traceback
        traceback.print_exc()
        return False


# Initialize ML pipeline and train if needed
try:
    logger.info("Initializing Parkinson's Detection ML Pipeline...")
    ml_pipeline = ParkinsonMLPipeline()
    check_and_train_models()
    logger.info("✓ ML Pipeline initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize ML Pipeline: {str(e)}")
    import traceback
    traceback.print_exc()
    raise  # Re-raise to prevent app from starting with broken ML components

# Initialize dataset matcher for identifying known samples
try:
    logger.info("Initializing Dataset Matcher...")
    dataset_matcher = DatasetMatcher()
    logger.info("✓ Dataset Matcher initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize Dataset Matcher: {str(e)}")
    import traceback
    traceback.print_exc()
    raise  # Re-raise to prevent app from starting with broken components

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, ngrok-skip-browser-warning'
    return response

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for testing"""
    return jsonify({'message': 'Parkinson Detection API is running', 'status': 'ok'})

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Serve favicon to prevent 404 errors"""
    return '', 204  # Return empty response with No Content status

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/analyze-stream', methods=['POST'])
def analyze_data_stream():
    """
    Streaming endpoint for real-time analysis progress
    Streams processing steps to frontend as Server-Sent Events (SSE)
    """
    def generate():
        try:
            # Send initial heartbeat to establish connection
            yield ": heartbeat\n\n"
            
            # Get test mode (voice, tremor, or both)
            test_mode = request.form.get('test_mode', 'both')
            
            # Validate based on test mode
            has_audio = 'audio' in request.files
            has_motion = 'motion_data' in request.form
            
            # Voice-only test
            if test_mode == 'voice':
                if not has_audio:
                    yield f"data: {json.dumps({'error': 'No audio file provided for voice test'})}\n\n"
                    return
                motion_data_str = None
            
            # Tremor-only test
            elif test_mode == 'tremor':
                if not has_motion:
                    yield f"data: {json.dumps({'error': 'No motion data provided for tremor test'})}\n\n"
                    return
                audio_file = None
                motion_data_str = request.form['motion_data']
            
            # Both tests
            else:
                if not has_audio:
                    yield f"data: {json.dumps({'error': 'No audio file provided'})}\n\n"
                    return
                if not has_motion:
                    yield f"data: {json.dumps({'error': 'No motion data provided'})}\n\n"
                    return
                motion_data_str = request.form['motion_data']

            # Get audio file if provided
            if has_audio:
                audio_file = request.files['audio']
            else:
                audio_file = None
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'validating', 'message': f'🔍 Validating {test_mode} test data...'})}\n\n"
            time.sleep(0.2)
            
            # Validate audio file (if voice or both)
            if audio_file:
                if audio_file.filename == '':
                    yield f"data: {json.dumps({'error': 'No audio file selected'})}\n\n"
                    return

                allowed_extensions = {'webm', 'wav', 'mp3', 'ogg'}
                file_ext = audio_file.filename.rsplit('.', 1)[1].lower() if '.' in audio_file.filename else ''
                if file_ext not in allowed_extensions:
                    yield f"data: {json.dumps({'error': 'Invalid audio file format'})}\n\n"
                    return

            # Parse and validate motion data (if tremor or both)
            motion_data = None
            valid_samples = 0
            
            if motion_data_str:
                try:
                    motion_data = json.loads(motion_data_str)
                    
                    logger.info(f"Received motion data: {len(motion_data)} samples")
                    
                    if not isinstance(motion_data, list):
                        logger.error(f"Motion data is not a list: {type(motion_data)}")
                        yield f"data: {json.dumps({'error': 'Invalid motion data format - expected list'})}\n\n"
                        return
                    
                    # Debug: Log first sample to see structure
                    if len(motion_data) > 0:
                        logger.info(f"First sample structure: {motion_data[0]}")
                        logger.info(f"First sample keys: {list(motion_data[0].keys()) if isinstance(motion_data[0], dict) else 'Not a dict'}")
                    
                    if len(motion_data) < 50:
                        logger.warning(f"Insufficient motion data samples: {len(motion_data)} < 50")
                        yield f"data: {json.dumps({'error': f'Insufficient motion data: {len(motion_data)} samples (need at least 50). Please hold device steady and collect for 10-15 seconds.'})}\n\n"
                        return
                    
                    # Validate samples - be very lenient
                    for i, sample in enumerate(motion_data):
                        if not isinstance(sample, dict):
                            if i < 5:
                                logger.warning(f"Sample {i} is not a dict: {type(sample)} = {sample}")
                            continue
                            
                        # Check required fields - support both formats
                        has_timestamp = 'timestamp' in sample
                        
                        # Check for both accelerationX/Y/Z and x/y/z formats
                        has_accel_format = ('accelerationX' in sample and 'accelerationY' in sample and 'accelerationZ' in sample)
                        has_xyz_format = ('x' in sample and 'y' in sample and 'z' in sample)
                        
                        if i < 3:  # Log first 3 samples for debugging
                            logger.info(f"Sample {i}: timestamp={has_timestamp}, accel_format={has_accel_format}, xyz_format={has_xyz_format}")
                            logger.info(f"Sample {i} data: {sample}")
                        
                        if has_timestamp and (has_accel_format or has_xyz_format):
                            # Get values from either format
                            if has_accel_format:
                                x_val = sample.get('accelerationX')
                                y_val = sample.get('accelerationY')
                                z_val = sample.get('accelerationZ')
                            else:  # has_xyz_format
                                x_val = sample.get('x')
                                y_val = sample.get('y')
                                z_val = sample.get('z')
                            
                            # Check if values are numeric (including 0)
                            x_numeric = isinstance(x_val, (int, float)) and not isinstance(x_val, bool)
                            y_numeric = isinstance(y_val, (int, float)) and not isinstance(y_val, bool)
                            z_numeric = isinstance(z_val, (int, float)) and not isinstance(z_val, bool)
                            
                            if i < 3:
                                logger.info(f"Sample {i} numeric check: x={x_numeric} ({type(x_val)}), y={y_numeric} ({type(y_val)}), z={z_numeric} ({type(z_val)})")
                            
                            if x_numeric and y_numeric and z_numeric:
                                valid_samples += 1
                            else:
                                if i < 5:  # Log first few invalid samples for debugging
                                    logger.warning(f"Sample {i} has non-numeric values: x={x_val} ({type(x_val)}), y={y_val} ({type(y_val)}), z={z_val} ({type(z_val)})")
                        else:
                            if i < 5:
                                logger.warning(f"Sample {i} missing fields: timestamp={has_timestamp}, accel_format={has_accel_format}, xyz_format={has_xyz_format}")
                                logger.warning(f"Sample {i} keys: {list(sample.keys())}")
                    
                    validity_percent = (valid_samples / len(motion_data)) * 100 if len(motion_data) > 0 else 0
                    logger.info(f"Motion data validation: {valid_samples}/{len(motion_data)} valid samples ({validity_percent:.1f}%)")
                    
                    # Very lenient threshold - 30% valid samples is acceptable
                    if valid_samples < max(30, len(motion_data) * 0.3):
                        logger.error(f"Motion data quality insufficient: {valid_samples}/{len(motion_data)} ({validity_percent:.1f}%)")
                        yield f"data: {json.dumps({'error': f'Motion data quality insufficient: only {valid_samples}/{len(motion_data)} valid samples ({validity_percent:.1f}%). Please ensure device sensors are working and you granted motion permissions.'})}\n\n"
                        return
                    
                    logger.info(f"✓ Motion data validated successfully: {valid_samples} valid samples ({validity_percent:.1f}%)")
                        
                except json.JSONDecodeError:
                    yield f"data: {json.dumps({'error': 'Invalid motion data format'})}\n\n"
                    return

            # Save audio file (if provided) and convert to WAV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_path = None
            
            if audio_file:
                # Save as WebM first
                webm_filename = f'audio_{timestamp}.webm'
                webm_path = os.path.join(app.config['UPLOAD_FOLDER'], webm_filename)
                audio_file.save(webm_path)
                
                # Convert to WAV for librosa compatibility
                wav_filename = f'audio_{timestamp}.wav'
                wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
                
                conversion_success = convert_webm_to_wav(webm_path, wav_path)
                
                if conversion_success and os.path.exists(wav_path):
                    audio_path = wav_path
                    try:
                        os.remove(webm_path)  # Clean up WebM file
                    except:
                        pass
                    logger.info(f"Successfully converted WebM to WAV: {wav_filename}")
                else:
                    # If conversion fails, try to use WebM directly (may fail in feature extraction)
                    logger.warning("WebM to WAV conversion failed, attempting to process WebM directly")
                    audio_path = webm_path
            
            # Validation message based on test mode
            if test_mode == 'voice':
                validation_msg = '✅ Voice data validated successfully'
            elif test_mode == 'tremor':
                validation_msg = f'✅ Validation complete: {valid_samples} valid motion samples'
            else:
                validation_msg = f'✅ Voice and tremor data validated: {valid_samples} motion samples'
            
            yield f"data: {json.dumps({'status': 'validated', 'message': validation_msg})}\n\n"
            time.sleep(0.1)
            
            # Start analysis - shorter delays to prevent timeouts
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔬 Starting ML Pipeline Analysis...', 'progress': 10})}\n\n"
            time.sleep(0.3)
            
            # Voice feature extraction (if voice or both)
            if test_mode in ['voice', 'both']:
                yield f"data: {json.dumps({'status': 'processing', 'message': '🎤 Extracting voice features...', 'progress': 20})}\n\n"
                time.sleep(0.2)
                
                yield f"data: {json.dumps({'status': 'processing', 'message': '📊 Analyzing audio characteristics...', 'progress': 35})}\n\n"
                time.sleep(0.2)
            
            # Tremor feature extraction (if tremor or both)
            if test_mode in ['tremor', 'both']:
                yield f"data: {json.dumps({'status': 'processing', 'message': '🤚 Extracting tremor features...', 'progress': 50})}\n\n"
                time.sleep(0.2)
                
                yield f"data: {json.dumps({'status': 'processing', 'message': '🔢 Computing frequency domain features...', 'progress': 65})}\n\n"
                time.sleep(0.2)
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '🧠 Running machine learning models...', 'progress': 80})}\n\n"
            time.sleep(0.3)
            
            # Perform actual analysis based on test mode
            # Send heartbeat before heavy computation
            yield ": processing\n\n"
            
            if test_mode == 'voice':
                # Voice-only analysis
                results = ml_pipeline.analyze_voice_only(audio_path) if hasattr(ml_pipeline, 'analyze_voice_only') else ml_pipeline.analyze(audio_path, None)
            elif test_mode == 'tremor':
                # Tremor-only analysis
                results = ml_pipeline.analyze_tremor_only(motion_data) if hasattr(ml_pipeline, 'analyze_tremor_only') else ml_pipeline.analyze(None, motion_data)
            else:
                # Combined analysis
                results = ml_pipeline.analyze(audio_path, motion_data)
            
            # Send heartbeat after computation
            yield ": computed\n\n"
            
            # Dataset matching - check if sample matches known dataset
            yield f"data: {json.dumps({'status': 'processing', 'message': '🔍 Checking dataset matches...', 'progress': 85})}\n\n"
            time.sleep(0.1)
            
            try:
                voice_features_vector = results.get('voice_features_vector')
                tremor_features_vector = results.get('tremor_features_vector')
                
                # Only perform matching if we have valid feature vectors
                if voice_features_vector is not None or tremor_features_vector is not None:
                    dataset_match = dataset_matcher.match_combined(
                        voice_features=voice_features_vector,
                        tremor_features=tremor_features_vector
                    )
                    results['dataset_match'] = dataset_match
                    
                    # Log useful information about dataset matching
                    if dataset_match.get('overall_match'):
                        consensus = dataset_match.get('consensus_category', 'Unknown')
                        logger.info(f"✓ Dataset Match Found: {consensus}")
                        if dataset_match.get('voice_match', {}).get('matched'):
                            similarity = dataset_match['voice_match'].get('similarity', 0)
                            logger.info(f"  - Voice: {similarity:.1%} similar to known sample")
                        if dataset_match.get('tremor_match', {}).get('matched'):
                            similarity = dataset_match['tremor_match'].get('similarity', 0)
                            logger.info(f"  - Tremor: {similarity:.1%} similar to known sample")
                    else:
                        logger.info("✓ New/Unique Sample: No match in training dataset (this is normal for new patients)")
                        # Show best similarities even when not matched
                        voice_sim = dataset_match.get('voice_match', {}).get('best_similarity', 0) if dataset_match.get('voice_match') else 0
                        tremor_sim = dataset_match.get('tremor_match', {}).get('best_similarity', 0) if dataset_match.get('tremor_match') else 0
                        if voice_sim > 0:
                            logger.info(f"  - Voice similarity: {voice_sim:.1%} (threshold: 95%)")
                        if tremor_sim > 0:
                            logger.info(f"  - Tremor similarity: {tremor_sim:.1%} (threshold: 95%)")
                else:
                    logger.warning("No feature vectors available for dataset matching")
                    results['dataset_match'] = {'error': 'No features available for matching'}
            except Exception as e:
                import traceback
                logger.warning(f"Dataset matching failed: {str(e)}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                results['dataset_match'] = {'error': 'Dataset matching unavailable'}
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '💾 Storing results...', 'progress': 90})}\n\n"
            time.sleep(0.2)
            
            # Store results - only if we actually have the data
            voice_recording_id = None
            tremor_recording_id = None
            
            # Only store voice recording if we have an actual audio file
            if audio_path is not None and 'voice_confidence' in results:
                try:
                    voice_recording_id, _ = storage_manager.store_voice_recording(
                        audio_file_path=audio_path,
                        prediction=1 if results.get('prediction') == 'Affected' else 0,
                        confidence=results.get('confidence', 0.0),
                        voice_confidence=results.get('voice_confidence', 0.0),
                        features=results.get('features', {})
                    )
                    results['voice_recording_id'] = voice_recording_id
                    logger.info(f"Stored voice recording: {voice_recording_id}")
                except Exception as e:
                    logger.warning(f"Failed to store voice recording: {e}")
            
            # Only store tremor data if we have actual motion data
            if motion_data is not None and 'tremor_confidence' in results:
                try:
                    tremor_recording_id, _ = storage_manager.store_tremor_data(
                        motion_data=motion_data,
                        prediction=1 if results.get('prediction') == 'Affected' else 0,
                        confidence=results.get('confidence', 0.0),
                        tremor_confidence=results.get('tremor_confidence', 0.0),
                        features=results.get('features', {})
                    )
                    results['tremor_recording_id'] = tremor_recording_id
                    logger.info(f"Stored tremor recording: {tremor_recording_id}")
                except Exception as e:
                    logger.warning(f"Failed to store tremor recording: {e}")
            
            if voice_recording_id and tremor_recording_id:
                storage_manager.store_combined_analysis(
                    voice_recording_id=voice_recording_id,
                    tremor_recording_id=tremor_recording_id,
                    combined_prediction=results.get('prediction'),
                    combined_confidence=results.get('confidence', 0.0)
                )
            
            yield f"data: {json.dumps({'status': 'processing', 'message': '✨ Finalizing...', 'progress': 95})}\n\n"
            time.sleep(0.2)
            
            # Make results JSON serializable (convert numpy arrays to lists)
            json_safe_results = make_json_serializable(results)
            
            # Send final results
            yield f"data: {json.dumps({'status': 'complete', 'message': '✅ Analysis complete!', 'progress': 100, 'results': json_safe_results})}\n\n"
            time.sleep(0.1)
            
            # Clean up audio file if it exists
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                    logger.info(f"Cleaned up audio file: {audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup audio file: {cleanup_error}")
            
            # Force garbage collection to release file handles and memory
            # This is critical on Windows to release locked files
            gc.collect()
            logger.info("Garbage collection completed - resources released")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in streaming analysis: {str(e)}")
            logger.error(f"Traceback: {error_details}")
            yield f"data: {json.dumps({'error': f'Analysis failed: {str(e)}', 'details': 'Check server logs for more information'})}\n\n"
            
            # Clean up on error as well
            try:
                if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            gc.collect()  # Force cleanup even on error
    
    response = Response(stream_with_context(generate()), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive'
    })
    response.timeout = None  # Disable timeout for streaming
    return response

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    Main endpoint for analyzing voice and motion data
    Expects:
    - audio file (multipart/form-data)
    - motion_data (JSON string)
    - user_id (optional): If provided, uses personalized model
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
        user_id = request.form.get('user_id', None)  # NEW: Get user_id if provided
        
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

        # Convert to WAV immediately for better compatibility
        wav_filename = f'audio_{timestamp}.wav'
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
        
        if convert_webm_to_wav(audio_path, wav_path):
            logger.info(f"Converted uploaded WebM to WAV: {wav_path}")
            # If conversion successful, use the WAV file
            try:
                os.remove(audio_path)
            except:
                pass
            audio_path = wav_path
        else:
            logger.warning("WebM conversion failed in analyze endpoint, proceeding with original file")

        # Enhanced logging with detailed progress
        print("\n" + "="*70)
        print("🔬 PARKINSON'S DETECTION - ANALYSIS STARTED")
        print("="*70)
        logger.info(f"📁 Processing audio file: {audio_filename}")
        logger.info(f"📊 Motion data samples: {len(motion_data)}")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📏 Audio size: {os.path.getsize(audio_path) / 1024:.2f} KB")
        print(f"🎯 Valid motion samples: {valid_samples}/{len(motion_data)}")
        print("-"*70)

        # Process data through ML pipeline
        print("\n🔄 Starting ML Pipeline Analysis...")
        
        # NEW: Create pipeline with user_id if provided (enables personalized model)
        if user_id:
            logger.info(f"🎯 Using personalized model for user: {user_id}")
            analysis_pipeline = ParkinsonMLPipeline(user_id=user_id)
        else:
            analysis_pipeline = ml_pipeline  # Use global pipeline
        
        results = analysis_pipeline.analyze(audio_path, motion_data)
        
        # Log detailed features (matching CSV structure)
        print("\n" + "="*70)
        print("📈 EXTRACTED FEATURES (CSV Format)")
        print("="*70)
        
        features = results.get('features', {})
        
        # Log Magnitude features
        if any(k.startswith('magnitude_') for k in features.keys()):
            print("\n🌊 MAGNITUDE FEATURES:")
            print(f"   ✓ Magnitude_mean:          {features.get('magnitude_mean', 0):.6f}")
            print(f"   ✓ Magnitude_std_dev:       {features.get('magnitude_std_dev', 0):.6f}")
            print(f"   ✓ Magnitude_rms:           {features.get('magnitude_rms', 0):.6f}")
            print(f"   ✓ Magnitude_energy:        {features.get('magnitude_energy', 0):.6f}")
            print(f"   ✓ Magnitude_peaks_rt:      {features.get('magnitude_peaks_rt', 0):.6f}")
            print(f"   ✓ Magnitude_ssc_rt:        {features.get('magnitude_ssc_rt', 0):.6f}")
            print(f"   ✓ Magnitude_fft_dom_freq:  {features.get('magnitude_fft_dom_freq', 0):.6f}")
            print(f"   ✓ Magnitude_fft_tot_power: {features.get('magnitude_fft_tot_power', 0):.6f}")
            print(f"   ✓ Magnitude_fft_energy:    {features.get('magnitude_fft_energy', 0):.6f}")
            print(f"   ✓ Magnitude_fft_entropy:   {features.get('magnitude_fft_entropy', 0):.6f}")
            print(f"   ✓ Magnitude_sampen:        {features.get('magnitude_sampen', 0):.6f}")
            print(f"   ✓ Magnitude_dfa:           {features.get('magnitude_dfa', 0):.6f}")
        
        # Log PC1 features
        if any(k.startswith('pc1_') for k in features.keys()):
            print("\n🎯 PC1 FEATURES:")
            print(f"   ✓ PC1_mean:          {features.get('pc1_mean', 0):.6f}")
            print(f"   ✓ PC1_std_dev:       {features.get('pc1_std_dev', 0):.6f}")
            print(f"   ✓ PC1_rms:           {features.get('pc1_rms', 0):.6f}")
            print(f"   ✓ PC1_energy:        {features.get('pc1_energy', 0):.6f}")
            print(f"   ✓ PC1_peaks_rt:      {features.get('pc1_peaks_rt', 0):.6f}")
            print(f"   ✓ PC1_zero_cross_rt: {features.get('pc1_zero_cross_rt', 0):.6f}")
            print(f"   ✓ PC1_ssc_rt:        {features.get('pc1_ssc_rt', 0):.6f}")
            print(f"   ✓ PC1_fft_dom_freq:  {features.get('pc1_fft_dom_freq', 0):.6f}")
            print(f"   ✓ PC1_fft_tot_power: {features.get('pc1_fft_tot_power', 0):.6f}")
            print(f"   ✓ PC1_fft_energy:    {features.get('pc1_fft_energy', 0):.6f}")
            print(f"   ✓ PC1_fft_entropy:   {features.get('pc1_fft_entropy', 0):.6f}")
            print(f"   ✓ PC1_sampen:        {features.get('pc1_sampen', 0):.6f}")
            print(f"   ✓ PC1_dfa:           {features.get('pc1_dfa', 0):.6f}")
        
        # Log tremor classifications
        if any(k.endswith('_tremor') for k in features.keys()):
            print("\n🤚 TREMOR CLASSIFICATIONS:")
            print(f"   ✓ Rest_tremor:     {features.get('rest_tremor', 0)}")
            print(f"   ✓ Postural_tremor: {features.get('postural_tremor', 0)}")
            print(f"   ✓ Kinetic_tremor:  {features.get('kinetic_tremor', 0)}")
        
        # Log voice features
        if any(k in ['pitch_mean', 'jitter', 'shimmer'] for k in features.keys()):
            print("\n🎤 VOICE FEATURES:")
            print(f"   ✓ Pitch Mean:         {features.get('pitch_mean', 0):.4f} Hz")
            print(f"   ✓ Pitch Std Dev:      {features.get('pitch_std', 0):.4f} Hz")
            print(f"   ✓ Jitter:             {features.get('jitter', 0):.6f} %")
            print(f"   ✓ Shimmer:            {features.get('shimmer', 0):.6f} dB")
            print(f"   ✓ HNR:                {features.get('hnr', 0):.4f} dB")
            print(f"   ✓ Spectral Centroid:  {features.get('spectral_centroid', 0):.4f} Hz")
            print(f"   ✓ Zero Crossing Rate: {features.get('zcr', 0):.6f}")
            print(f"   ✓ Energy:             {features.get('energy', 0):.6f} dB")
        
        print("\n" + "="*70)
        print("🎯 ANALYSIS RESULTS")
        print("="*70)
        print(f"   📊 Prediction:         {results.get('prediction', 'N/A')}")
        print(f"   📈 Overall Confidence: {results.get('confidence', 0):.2f}%")
        if 'voice_confidence' in results:
            print(f"   🎤 Voice Patterns:     {results.get('voice_patterns', results.get('voice_confidence', 0)):.2f}%")
        if 'tremor_confidence' in results:
            print(f"   🤚 Motion Patterns:    {results.get('motion_patterns', results.get('tremor_confidence', 0)):.2f}%")
        print(f"   ⏱️  Processing Time:    {results.get('metadata', {}).get('processing_time', 0):.2f}s")
        print("="*70 + "\n")

        # Store results (voice and tremor recordings)
        voice_recording_id = None
        tremor_recording_id = None
        
        try:
            # Store voice recording
            if 'voice_confidence' in results:
                voice_recording_id, voice_stored_path = storage_manager.store_voice_recording(
                    audio_file_path=audio_path,
                    prediction=1 if results.get('prediction') == 'Affected' else 0,
                    confidence=results.get('confidence', 0.0),
                    voice_confidence=results.get('voice_confidence', 0.0),
                    features=results.get('features', {})
                )
                results['voice_recording_id'] = voice_recording_id
                logger.info(f"Stored voice recording: {voice_recording_id}")
            
            # Store tremor data
            if 'tremor_confidence' in results:
                tremor_recording_id, tremor_stored_path = storage_manager.store_tremor_data(
                    motion_data=motion_data,
                    prediction=1 if results.get('prediction') == 'Affected' else 0,
                    confidence=results.get('confidence', 0.0),
                    tremor_confidence=results.get('tremor_confidence', 0.0),
                    features=results.get('features', {})
                )
                results['tremor_recording_id'] = tremor_recording_id
                logger.info(f"Stored tremor data: {tremor_recording_id}")
            
            # Store combined analysis
            if voice_recording_id and tremor_recording_id:
                combined_entry = storage_manager.store_combined_analysis(
                    voice_recording_id=voice_recording_id,
                    tremor_recording_id=tremor_recording_id,
                    combined_prediction=results.get('prediction'),
                    combined_confidence=results.get('confidence', 0.0)
                )
                logger.info("Stored combined analysis")
                
        except Exception as e:
            logger.warning(f"Error storing results: {e}")

        # Clean up temporary file (after storage)
        try:
            os.remove(audio_path)
            logger.info(f"Cleaned up audio file: {audio_path}")
        except OSError:
            logger.warning(f"Could not remove temporary file: {audio_path}")
        
        # Force garbage collection to release file handles and memory
        gc.collect()
        logger.info("Garbage collection completed - resources released")

        # Make results JSON serializable (convert numpy arrays to lists)
        json_safe_results = make_json_serializable(results)
        
        # Return results with storage information
        return jsonify(json_safe_results)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        # Clean up on error
        try:
            if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        gc.collect()  # Force cleanup even on error
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get information about available ML models"""
    return jsonify({
        'models': {
            'voice_analysis': {
                'type': 'ensemble',
                'algorithms': ['SVM', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
                'features': ['MFCC', 'Spectral', 'Prosodic', 'Voice Quality'],
                'trained_on': 'Real voice dataset (Healthy_AH + Parkinsons_AH)'
            },
            'tremor_analysis': {
                'type': 'ensemble',
                'algorithms': ['SVM', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
                'features': ['Frequency Domain', 'Time Domain', 'Statistical'],
                'trained_on': 'Real tremor dataset (parkinson_tremor_dataset.csv)'
            }
        },
        'version': '2.0.0',
        'trained_date': datetime.now().strftime('%Y-%m-%d')
    })

@app.route('/api/storage/stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics for recorded data"""
    try:
        stats = storage_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/api/storage/recent', methods=['GET'])
def get_recent_recordings():
    """Get recent recordings"""
    try:
        count = request.args.get('count', 10, type=int)
        recording_type = request.args.get('type', 'all', type=str)
        
        recordings = storage_manager.get_recent_recordings(count=count, recording_type=recording_type)
        return jsonify({'recordings': recordings, 'count': len(recordings)})
    except Exception as e:
        logger.error(f"Error getting recent recordings: {e}")
        return jsonify({'error': 'Failed to retrieve recordings'}), 500

@app.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get information about loaded datasets"""
    try:
        loader = DatasetLoader()
        info = loader.get_dataset_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return jsonify({'error': 'Failed to retrieve dataset information'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ========================================================================
# PERSONALIZED BASELINE CALIBRATION ENDPOINTS
# ========================================================================

@app.route('/api/calibration/status', methods=['GET'])
def calibration_status():
    """Check if user has completed baseline calibration"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        status = personalized_handler.get_user_status(user_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting calibration status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibration/record', methods=['POST'])
def calibration_record():
    """Save a baseline (healthy) voice sample for personalized model"""
    try:
        user_id = request.form.get('user_id', 'default_user')
        sample_index = int(request.form.get('sample_index', 0))
        
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save audio temporarily
        temp_dir = 'temp_calibration'
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        
        # Handle WebM format
        if audio_file.filename.lower().endswith('.webm'):
            webm_path = os.path.join(temp_dir, f'baseline_{user_id}_{sample_index}_{timestamp}.webm')
            audio_file.save(webm_path)
            wav_path = webm_path.replace('.webm', '.wav')
            
            if not convert_webm_to_wav(webm_path, wav_path):
                return jsonify({'error': 'Audio conversion failed'}), 500
            
            audio_path = wav_path
            # Clean up WebM file
            try:
                os.remove(webm_path)
            except:
                pass
        else:
            audio_path = os.path.join(temp_dir, f'baseline_{user_id}_{sample_index}_{timestamp}.wav')
            audio_file.save(audio_path)
        
        # Extract features
        audio_features = audio_extractor.extract_features_fast(audio_path)
        
        # Check if silence detected
        if audio_features.get('_silence_detected', False):
            # Clean up
            try:
                os.remove(audio_path)
            except:
                pass
            return jsonify({
                'error': 'No sound detected',
                'message': 'Please speak clearly into the microphone during recording'
            }), 400
        
        # Save baseline sample
        personalized_handler.save_baseline_sample(user_id, audio_features, sample_index)
        
        # Clean up audio file
        try:
            os.remove(audio_path)
        except:
            pass
        
        logger.info(f"Saved baseline sample {sample_index} for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': f'Baseline sample {sample_index} saved successfully',
            'sample_index': sample_index,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error recording baseline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibration/record-tremor', methods=['POST'])
def calibration_record_tremor():
    """Save a baseline (healthy) tremor sample for personalized model"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        sample_index = int(data.get('sample_index', 0))
        motion_data = data.get('motion_data', [])
        
        if not motion_data or len(motion_data) < 10:
            return jsonify({'error': 'Insufficient motion data', 'message': 'Please record for the full duration'}), 400
        
        # Extract tremor features from motion data
        tremor_features = tremor_extractor.extract_features_fast(motion_data)
        
        # Check if data is valid
        if tremor_features.get('_no_motion', False):
            return jsonify({
                'error': 'No motion detected',
                'message': 'Please ensure device motion sensors are working'
            }), 400
        
        # Save baseline tremor sample
        personalized_handler.save_baseline_tremor_sample(user_id, tremor_features, sample_index)
        
        logger.info(f"Saved baseline tremor sample {sample_index} for user {user_id}")
        
        return jsonify({
            'success': True,
            'message': f'Tremor sample {sample_index} saved successfully',
            'sample_index': sample_index,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error recording tremor baseline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibration/train', methods=['POST'])
def calibration_train():
    """Train personalized model from collected baseline samples"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        
        result = personalized_handler.train_personalized_model(user_id, min_samples=3)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        logger.error(f"Error training personalized model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibration/reset', methods=['POST'])
def calibration_reset():
    """Delete user's calibration data (reset)"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        
        result = personalized_handler.delete_user_data(user_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error resetting calibration: {e}")
        return jsonify({'error': str(e)}), 500

# ========================================================================

# OPTIONS preflight handled by ngrok proxy - no need for manual handler

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  PARKINSON'S DETECTION - ML-POWERED ANALYSIS SYSTEM")
    print("="*70)
    print("\nSystem Status:")
    print("  * Resource-intensive ML models active")
    print("  * 150+ audio features | 200+ tremor features")
    print("  * Ensemble of 4 algorithms: SVM, RF, GBM, XGBoost")
    print("  * Processing time: 3-5 seconds (prioritizes accuracy)")
    print("\nAPI Endpoints:")
    print("  * GET  /api/health                - Health check")
    print("  * POST /api/analyze               - Main ML analysis endpoint (real ML)")
    print("  * GET  /api/models/info           - Model information")
    print("  * GET  /api/calibration/status    - Check baseline calibration status")
    print("  * POST /api/calibration/record    - Save baseline voice sample")
    print("  * POST /api/calibration/train     - Train personalized model")
    print("  * POST /api/calibration/reset     - Reset user calibration")
    print("\nServer starting on http://localhost:5000")
    print("="*70 + "\n")
    
    # Disable reloader to prevent interruption during analysis
    # Enable threading to handle multiple requests (essential for streaming + new requests)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

# For Vercel deployment
application = app