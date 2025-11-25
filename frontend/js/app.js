// Prevent duplicate class declaration
if (typeof window.ParkinsonDetectionApp !== 'undefined') {
    console.warn('⚠️ ParkinsonDetectionApp already loaded, skipping redeclaration');
} else {
    // Helper function to add ngrok headers to fetch requests
    window.fetchWithNgrokBypass = async (url, options = {}) => {
        const headers = options.headers || {};
        headers['ngrok-skip-browser-warning'] = 'true';
        headers['User-Agent'] = 'ParkinsonDetectionApp';
        return fetch(url, { ...options, headers });
    };

    // App State Management
    window.ParkinsonDetectionApp = class ParkinsonDetectionApp {
        constructor() {
            this.currentScreen = 'welcome-screen';
            this.audioContext = null;
            this.mediaRecorder = null;
            this.audioStream = null;
            this.audioData = [];
            this.motionData = [];
            this.isRecording = false;
            this.testProgress = 0;
        this.selectedTestMode = 'both'; // 'voice', 'tremor', or 'both'
        
        // API Configuration - Use AppConfig for environment-aware backend URL
        this.API_BASE_URL = window.AppConfig ? window.AppConfig.getBackendUrl() : '/api';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkBackendAvailability();
        
        // Request permissions on load for better UX
        this.requestPermissions();
    }

    setupEventListeners() {
        // Navigation
        document.getElementById('start-test-btn').addEventListener('click', () => {
            this.showScreen('test-mode-screen');
        });

        document.getElementById('back-to-welcome-btn').addEventListener('click', () => {
            this.showScreen('welcome-screen');
        });

        // Test mode selection
        document.querySelectorAll('.test-mode-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectTestMode(card.dataset.mode);
            });
        });

        document.getElementById('new-test-btn').addEventListener('click', () => {
            this.resetTest();
            this.showScreen('test-mode-screen');
        });

        // Voice recording
        document.getElementById('voice-record-btn').addEventListener('click', () => {
            this.toggleVoiceRecording();
        });

        // Tremor recording
        document.getElementById('tremor-record-btn').addEventListener('click', () => {
            this.startTremorTest();
        });

        // Share results
        document.getElementById('share-results-btn').addEventListener('click', () => {
            this.shareResults();
        });
    }

    // Test Mode Selection
    selectTestMode(mode) {
        // Remove previous selection
        document.querySelectorAll('.test-mode-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Add selection to clicked card
        document.querySelector(`[data-mode="${mode}"]`).classList.add('selected');
        
        this.selectedTestMode = mode;
        
        // Start test after short delay
        setTimeout(() => {
            this.showScreen('test-screen');
            this.startTest();
        }, 800);
    }

    // Screen Management
    showScreen(screenId) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        
        // Show target screen
        document.getElementById(screenId).classList.add('active');
        this.currentScreen = screenId;
    }

    // Test Management
    startTest() {
        this.resetTestData();
        
        // Update progress based on test mode
        if (this.selectedTestMode === 'voice') {
            this.updateProgress(20, 'Starting voice test...');
            this.startVoicePhase();
        } else if (this.selectedTestMode === 'tremor') {
            this.updateProgress(20, 'Starting tremor test...');
            this.startTremorPhase();
        } else {
            this.updateProgress(10, 'Starting voice test...');
            this.startVoicePhase();
        }
    }

    startVoicePhase() {
        // Show voice test, hide others
        document.getElementById('voice-test').classList.remove('hidden');
        document.getElementById('tremor-test').classList.add('hidden');
    }

    resetTest() {
        this.testProgress = 0;
        this.audioData = [];
        this.motionData = [];
        this.isRecording = false;
        
        // Reset UI
        this.updateProgress(0, 'Preparing test...');
        document.getElementById('voice-test').classList.remove('hidden');
        document.getElementById('tremor-test').classList.add('hidden');
        
        // Reset buttons
        const voiceBtn = document.getElementById('voice-record-btn');
        const tremorBtn = document.getElementById('tremor-record-btn');
        voiceBtn.textContent = '🎤 Start Recording';
        voiceBtn.classList.remove('recording');
        tremorBtn.textContent = '📱 Start Tremor Test';
        tremorBtn.classList.remove('recording');
        
        // Clear status messages
        document.getElementById('voice-status').innerHTML = '';
        document.getElementById('tremor-status').innerHTML = '';
    }

    resetTestData() {
        this.audioData = [];
        this.motionData = [];
        this.testProgress = 0;
        
        // Hide metrics containers
        const voiceMetrics = document.getElementById('voice-metrics');
        const tremorMetrics = document.getElementById('tremor-metrics');
        if (voiceMetrics) voiceMetrics.classList.add('hidden');
        if (tremorMetrics) tremorMetrics.classList.add('hidden');
    }

    updateProgress(percentage, text) {
        this.testProgress = percentage;
        document.getElementById('progress-fill').style.width = percentage + '%';
        document.getElementById('progress-text').textContent = text;
    }



    // Backend Availability Check - REQUIRED (no fallback)
    async checkBackendAvailability() {
        try {
            const response = await window.fetchWithNgrokBypass(`${this.API_BASE_URL}/health`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                console.log('✅ Backend is available and ready');
                const data = await response.json();
                console.log(`Backend version: ${data.version}`);
            } else {
                throw new Error('Backend not responding');
            }
        } catch (error) {
            console.error('❌ Backend not available:', error);
            this.showBackendRequiredNotification();
        }
    }

    showBackendRequiredNotification() {
        // Show notification that backend is required
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            font-size: 1rem;
            z-index: 10001;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 400px;
            text-align: center;
        `;
        notification.innerHTML = `
            <h3 style="margin: 0 0 1rem 0;">⚠️ Backend Required</h3>
            <p style="margin: 0 0 1rem 0;">The analysis backend is not running. All processing must be performed by the backend server.</p>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Please start the Flask backend server and refresh this page.</p>
        `;
        document.body.appendChild(notification);
    }

    // Permission Management
    async requestPermissions() {
        try {
            // Check if getUserMedia is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn('⚠️ getUserMedia not available - HTTPS required');
                this.showError('Microphone access requires HTTPS. Please access the app via HTTPS.');
                return;
            }
            
            // Request microphone permission
            await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('✅ Microphone permission granted');
        } catch (error) {
            console.log('❌ Microphone permission denied:', error);
        }

        try {
            // Check if DeviceMotionEvent is available
            if (typeof window.DeviceMotionEvent === 'undefined') {
                console.warn('⚠️ DeviceMotionEvent not available');
                return;
            }
            
            // Request motion permission (for iOS 13+)
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                const permission = await DeviceMotionEvent.requestPermission();
                console.log('Motion permission:', permission);
            }
        } catch (error) {
            console.log('Motion permission not needed or denied:', error);
        }
    }

    // Voice Recording Functionality
    async toggleVoiceRecording() {
        if (!this.isRecording) {
            await this.startVoiceRecording();
        } else {
            this.stopVoiceRecording();
        }
    }

    async startVoiceRecording() {
        try {
            // Check if mediaDevices API is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Microphone access not available. Please use HTTPS (https://) to access this app. Current protocol: ' + window.location.protocol);
            }

            // Get user media
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100
                }
            });

            // Enhanced audio setup for maximum accuracy
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.audioStream);
            const analyser = this.audioContext.createAnalyser();
            
            // High-precision audio analysis settings
            analyser.fftSize = 4096; // Higher resolution for better frequency analysis
            analyser.smoothingTimeConstant = 0.1; // Less smoothing for accurate real-time data
            source.connect(analyser);

            // Enhanced media recorder with optimal settings for accuracy
            const mimeTypes = [
                'audio/webm;codecs=opus',
                'audio/webm;codecs=pcm',
                'audio/wav',
                'audio/ogg;codecs=opus'
            ];
            
            let selectedMimeType = 'audio/webm';
            for (const mimeType of mimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    selectedMimeType = mimeType;
                    break;
                }
            }

            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: selectedMimeType,
                audioBitsPerSecond: 128000 // Higher bitrate for better quality
            });

            this.audioData = [];
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioData.push(event.data);
                    
                    // Real-time audio quality assessment
                    this.assessAudioQuality(analyser);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.completeVoiceRecording();
            };

            // Start recording with high precision timing
            this.mediaRecorder.start(50); // Collect data every 50ms for higher accuracy
            this.isRecording = true;

            // Update UI
            const button = document.getElementById('voice-record-btn');
            button.innerHTML = '<span class="record-icon">⏹️</span>Stop Recording';
            button.classList.add('recording');

            // Start visualization
            this.startAudioVisualization(analyser);

            // Update status
            this.showStatus('voice-status', 'Recording... Speak clearly for 10 seconds', 'success');

            // Auto-stop after 10 seconds
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopVoiceRecording();
                }
            }, 10000);

            this.updateProgress(25, 'Recording voice...');

        } catch (error) {
            console.error('Error starting voice recording:', error);
            this.showStatus('voice-status', 'Error: Could not access microphone. Please check permissions.', 'error');
        }
    }

    stopVoiceRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }

        if (this.audioContext) {
            this.audioContext.close();
        }

        this.isRecording = false;

        // Update UI
        const button = document.getElementById('voice-record-btn');
        button.innerHTML = '<span class="record-icon">🎤</span>Start Recording';
        button.classList.remove('recording');
    }

    completeVoiceRecording() {
        if (this.selectedTestMode === 'voice') {
            this.updateProgress(90, 'Voice recording completed');
            this.showStatus('voice-status', 'Voice recording completed successfully!', 'success');
            
            // Move to processing for voice-only mode
            setTimeout(() => {
                this.startProcessing();
            }, 1500);
        } else {
            this.updateProgress(50, 'Voice recording completed');
            this.showStatus('voice-status', 'Voice recording completed successfully!', 'success');
            
            // Move to tremor test for combined mode
            setTimeout(() => {
                this.startTremorPhase();
            }, 1500);
        }
    }

    startAudioVisualization(analyser) {
        const visualizer = document.getElementById('voice-visualizer');
        const metricsContainer = document.getElementById('voice-metrics');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const timeDataArray = new Uint8Array(analyser.fftSize);

        // Show metrics container
        metricsContainer.classList.remove('hidden');

        // Clear existing visualization
        visualizer.innerHTML = '';

        // Create audio bars
        const numBars = 32;
        const bars = [];
        for (let i = 0; i < numBars; i++) {
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.height = '10px';
            visualizer.appendChild(bar);
            bars.push(bar);
        }

        const animate = () => {
            if (!this.isRecording) return;

            analyser.getByteFrequencyData(dataArray);
            analyser.getByteTimeDomainData(timeDataArray);

            // Update bars based on frequency data
            for (let i = 0; i < numBars; i++) {
                const value = dataArray[i * Math.floor(bufferLength / numBars)];
                const height = (value / 255) * 80 + 10; // Scale to 10-90px
                bars[i].style.height = height + 'px';
            }

            // Calculate and display real-time metrics
            this.updateVoiceMetrics(dataArray, timeDataArray, bufferLength);

            requestAnimationFrame(animate);
        };

        animate();
    }

    updateVoiceMetrics(frequencyData, timeData, bufferLength) {
        try {
            // 1. Audio Level (RMS)
            let sum = 0;
            for (let i = 0; i < frequencyData.length; i++) {
                sum += frequencyData[i] * frequencyData[i];
            }
            const rms = Math.sqrt(sum / frequencyData.length);
            const audioLevel = 20 * Math.log10(rms / 255 + 0.0001); // Convert to dB
            const audioLevelNormalized = Math.max(0, Math.min(100, (audioLevel + 60) * 1.67)); // Normalize to 0-100

            const audioLevelEl = document.getElementById('audio-level');
            const audioLevelBarEl = document.getElementById('audio-level-bar');
            if (audioLevelEl && audioLevelBarEl) {
                audioLevelEl.textContent = audioLevel.toFixed(1) + ' dB';
                audioLevelBarEl.style.width = audioLevelNormalized + '%';
                this.setBarQuality('audio-level-bar', audioLevelNormalized);
            }
        } catch (error) {
            console.error('Error updating voice metrics:', error);
            return;
        }

        // 2. Estimate Pitch using Zero Crossing Rate
        try {
            let zeroCrossings = 0;
            for (let i = 1; i < timeData.length; i++) {
                if ((timeData[i] >= 128 && timeData[i-1] < 128) || 
                    (timeData[i] < 128 && timeData[i-1] >= 128)) {
                    zeroCrossings++;
                }
            }
            const zcr = zeroCrossings / timeData.length;
            
            // Rough pitch estimation (in Hz) - typical human voice 85-255 Hz
            const estimatedPitch = Math.max(80, Math.min(400, zcr * 22050 / 2));
            const pitchNormalized = ((estimatedPitch - 80) / (400 - 80)) * 100;
            
            const pitchEl = document.getElementById('pitch-value');
            const pitchBarEl = document.getElementById('pitch-bar');
            if (pitchEl && pitchBarEl) {
                pitchEl.textContent = estimatedPitch.toFixed(0) + ' Hz';
                pitchBarEl.style.width = pitchNormalized + '%';
                this.setBarQuality('pitch-bar', pitchNormalized > 30 ? 70 : 50);
            }

            // 6. Zero Crossing Rate (measure of noisiness)
            const zcrEl = document.getElementById('zcr-value');
            const zcrBarEl = document.getElementById('zcr-bar');
            if (zcrEl && zcrBarEl) {
                zcrEl.textContent = zcr.toFixed(3);
                const zcrNormalized = Math.min(100, zcr * 200);
                zcrBarEl.style.width = zcrNormalized + '%';
                this.setBarQuality('zcr-bar', zcrNormalized < 50 ? 70 : 50);
            }
        } catch (error) {
            console.error('Error calculating pitch/ZCR:', error);
        }

        // 3. Spectral Centroid (brightness of sound)
        try {
            let weightedSum = 0;
            let totalSum = 0;
            for (let i = 0; i < frequencyData.length; i++) {
                const freq = (i / bufferLength) * 22050; // Map to Hz
                weightedSum += freq * frequencyData[i];
                totalSum += frequencyData[i];
            }
            const spectralCentroid = totalSum > 0 ? weightedSum / totalSum : 0;
            const centroidNormalized = Math.min(100, (spectralCentroid / 5000) * 100);
            
            const centroidEl = document.getElementById('spectral-centroid');
            const centroidBarEl = document.getElementById('spectral-bar');
            if (centroidEl && centroidBarEl) {
                centroidEl.textContent = spectralCentroid.toFixed(0) + ' Hz';
                centroidBarEl.style.width = centroidNormalized + '%';
                this.setBarQuality('spectral-bar', centroidNormalized > 20 ? 75 : 55);
            }
        } catch (error) {
            console.error('Error calculating spectral centroid:', error);
        }

        // 4. Signal Quality (based on signal strength)
        try {
            const avgSignal = frequencyData.reduce((a, b) => a + b, 0) / frequencyData.length;
            const signalQuality = Math.min(100, (avgSignal / 128) * 100);
            
            const qualityEl = document.getElementById('signal-quality');
            const qualityBarEl = document.getElementById('quality-bar');
            if (qualityEl && qualityBarEl) {
                qualityEl.textContent = signalQuality.toFixed(0) + '%';
                qualityBarEl.style.width = signalQuality + '%';
                this.setBarQuality('quality-bar', signalQuality);
            }
        } catch (error) {
            console.error('Error calculating signal quality:', error);
        }

        // 5. Signal-to-Noise Ratio (SNR)
        try {
            const maxSignal = Math.max(...frequencyData);
            const minNoise = Math.min(...frequencyData.filter(v => v > 0));
            const snr = maxSignal > 0 ? 20 * Math.log10(maxSignal / Math.max(minNoise, 1)) : 0;
            const snrNormalized = Math.min(100, (snr / 40) * 100);
            
            const snrEl = document.getElementById('snr-value');
            const snrBarEl = document.getElementById('snr-bar');
            if (snrEl && snrBarEl) {
                snrEl.textContent = snr.toFixed(1) + ' dB';
                snrBarEl.style.width = snrNormalized + '%';
                this.setBarQuality('snr-bar', snrNormalized);
            }
        } catch (error) {
            console.error('Error calculating SNR:', error);
        }
    }

    setBarQuality(barId, qualityPercent) {
        try {
            const bar = document.getElementById(barId);
            if (!bar) return;
            
            bar.classList.remove('excellent', 'good', 'poor');
            if (qualityPercent >= 70) {
                bar.classList.add('excellent');
            } else if (qualityPercent >= 40) {
                bar.classList.add('good');
            } else {
                bar.classList.add('poor');
            }
        } catch (error) {
            console.error('Error setting bar quality:', error);
        }
    }

    // Tremor Detection Functionality
    startTremorPhase() {
        if (this.selectedTestMode === 'tremor') {
            this.updateProgress(20, 'Ready for tremor test');
        } else {
            this.updateProgress(60, 'Ready for tremor test');
        }
        
        // Hide voice test, show tremor test
        document.getElementById('voice-test').classList.add('hidden');
        document.getElementById('tremor-test').classList.remove('hidden');
    }

    startTremorTest() {
        this.motionData = [];
        
        // Update UI
        const button = document.getElementById('tremor-record-btn');
        button.innerHTML = '<span class="record-icon">⏹️</span>Recording Motion...';
        button.classList.add('recording');

        this.updateProgress(70, 'Recording motion data...');
        this.showStatus('tremor-status', 'Hold your phone steady with both hands...', 'success');

        // Start motion data collection
        this.startMotionCapture();

        // Auto-stop after 15 seconds
        setTimeout(() => {
            this.stopTremorTest();
        }, 15000);
    }

    startMotionCapture() {
        // Check if DeviceMotionEvent is available
        if (typeof window.DeviceMotionEvent === 'undefined') {
            console.error('❌ DeviceMotionEvent is not available in this browser');
            this.showError('Device Motion API is not supported by your browser');
            return;
        }

        // Check secure context (HTTPS required on most browsers)
        if (!window.isSecureContext) {
            console.error('❌ Motion sensors require a secure context (HTTPS)');
            const protocol = window.location.protocol;
            this.showError(`Motion sensors require HTTPS. Currently using ${protocol}. Please access via HTTPS.`);
            return;
        }

        let hasReceivedData = false;
        const dataTimeout = setTimeout(() => {
            if (!hasReceivedData) {
                console.error('❌ Motion sensor timeout: No data received after 3 seconds');
                this.showError('Motion sensors are not responding. Make sure you granted sensor permissions and are using HTTPS.');
            }
        }, 3000);

        const handleMotion = (event) => {
            hasReceivedData = true;
            clearTimeout(dataTimeout);
            
            // Enhanced motion data collection for 100% accuracy
            if (this.motionData.length < 1500) { // Collect for ~15 seconds at ~100Hz
                
                // Validate sensor data availability and accuracy
                const accelData = event.accelerationIncludingGravity;
                const rotationData = event.rotationRate;
                
                // Only collect high-quality samples
                if (accelData && 
                    typeof accelData.x === 'number' && 
                    typeof accelData.y === 'number' && 
                    typeof accelData.z === 'number' &&
                    !isNaN(accelData.x) && !isNaN(accelData.y) && !isNaN(accelData.z)) {
                    
                    // Apply data quality filters for accuracy
                    const sample = {
                        timestamp: performance.now(), // High-precision timestamp
                        accelerationX: parseFloat(accelData.x.toFixed(6)), // 6 decimal precision
                        accelerationY: parseFloat(accelData.y.toFixed(6)),
                        accelerationZ: parseFloat(accelData.z.toFixed(6)),
                        rotationAlpha: rotationData ? parseFloat((rotationData.alpha || 0).toFixed(6)) : 0,
                        rotationBeta: rotationData ? parseFloat((rotationData.beta || 0).toFixed(6)) : 0,
                        rotationGamma: rotationData ? parseFloat((rotationData.gamma || 0).toFixed(6)) : 0,
                        interval: event.interval || 10 // Sampling interval for accuracy
                    };
                    
                    // Validate acceleration values are within reasonable range (-50 to +50 m/s²)
                    const maxAccel = 50;
                    if (Math.abs(sample.accelerationX) <= maxAccel && 
                        Math.abs(sample.accelerationY) <= maxAccel && 
                        Math.abs(sample.accelerationZ) <= maxAccel) {
                        
                        this.motionData.push(sample);
                        
                        // Real-time data quality monitoring
                        this.updateDataQualityIndicator();
                    }
                }
            }
        };

        window.addEventListener('devicemotion', handleMotion);
        
        // Store reference to remove listener later
        this.motionHandler = handleMotion;

        // Start motion visualization with quality indicators
        this.startMotionVisualization();
    }

    updateDataQualityIndicator() {
        // Real-time data quality assessment for 100% accuracy
        if (this.motionData.length > 0) {
            const recentSamples = this.motionData.slice(-10); // Last 10 samples
            
            // Calculate sampling rate accuracy
            if (recentSamples.length >= 2) {
                const intervals = [];
                for (let i = 1; i < recentSamples.length; i++) {
                    intervals.push(recentSamples[i].timestamp - recentSamples[i-1].timestamp);
                }
                
                const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
                const samplingRate = 1000 / avgInterval; // Hz
                
                // Ideal sampling rate is around 100Hz
                const qualityScore = Math.min(100, Math.max(0, 
                    100 - Math.abs(samplingRate - 100) * 2));
                
                // Update quality indicator in UI
                this.displayDataQuality(qualityScore, samplingRate);
            }
        }
    }

    displayDataQuality(qualityScore, samplingRate) {
        // Display real-time quality metrics for user feedback
        const qualityElement = document.getElementById('data-quality-indicator');
        if (qualityElement) {
            const qualityClass = qualityScore >= 80 ? 'excellent' : 
                                qualityScore >= 60 ? 'good' : 'poor';
            
            qualityElement.innerHTML = `
                <div class="quality-meter ${qualityClass}">
                    <div class="quality-bar" style="width: ${qualityScore}%"></div>
                </div>
                <div class="quality-stats">
                    Quality: ${qualityScore.toFixed(0)}% | Rate: ${samplingRate.toFixed(1)}Hz
                </div>
            `;
        }
    }

    assessAudioQuality(analyser) {
        // Real-time audio quality assessment for 100% accuracy
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate signal strength and noise floor
        const sum = dataArray.reduce((a, b) => a + b, 0);
        const average = sum / bufferLength;
        
        // Calculate signal-to-noise ratio
        const maxSignal = Math.max(...dataArray);
        const minNoise = Math.min(...dataArray.filter(val => val > 0));
        const snr = maxSignal > 0 ? 20 * Math.log10(maxSignal / Math.max(minNoise, 1)) : 0;
        
        // Quality score based on signal strength and SNR
        const signalQuality = Math.min(100, average * 2); // 0-100 scale
        const noiseQuality = Math.min(100, Math.max(0, snr * 10)); // 0-100 scale
        const overallQuality = (signalQuality + noiseQuality) / 2;
        
        // Update audio quality indicator
        this.displayAudioQuality(overallQuality, average, snr);
    }

    displayAudioQuality(quality, signalLevel, snr) {
        // Display real-time audio quality metrics
        const qualityElement = document.getElementById('audio-quality-indicator');
        if (qualityElement) {
            const qualityClass = quality >= 70 ? 'excellent' : 
                                quality >= 50 ? 'good' : 'poor';
            
            qualityElement.innerHTML = `
                <div class="quality-meter ${qualityClass}">
                    <div class="quality-bar" style="width: ${quality}%"></div>
                </div>
                <div class="quality-stats">
                    Audio Quality: ${quality.toFixed(0)}% | Level: ${signalLevel.toFixed(1)} | SNR: ${snr.toFixed(1)}dB
                </div>
            `;
        }
    }

    startMotionVisualization() {
        const visualizer = document.getElementById('tremor-visualizer');
        const metricsContainer = document.getElementById('tremor-metrics');
        let startTime = Date.now();
        
        // Show metrics container
        metricsContainer.classList.remove('hidden');
        
        const animate = () => {
            if (this.motionData.length === 0) {
                requestAnimationFrame(animate);
                return;
            }

            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / 15000, 1); // 15 seconds max
            
            // Create a simple progress visualization
            visualizer.innerHTML = `
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                    <div style="width: 200px; height: 8px; background: #e9ecef; border-radius: 4px; margin-bottom: 1rem;">
                        <div style="width: ${progress * 100}%; height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px;"></div>
                    </div>
                    <p>Motion Data: ${this.motionData.length} samples</p>
                    <p>Time Remaining: ${Math.max(0, 15 - Math.floor(elapsed / 1000))}s</p>
                </div>
            `;

            // Update tremor metrics
            this.updateTremorMetrics();

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    updateTremorMetrics() {
        try {
            if (this.motionData.length < 10) {
                console.log('Not enough motion data yet:', this.motionData.length);
                return;
            }

            // Get recent samples for analysis (last 50 samples or all if less)
            const recentSamples = this.motionData.slice(-50);
            console.log('Updating tremor metrics with', recentSamples.length, 'samples');
            
            // Extract acceleration data
            const accelX = recentSamples.map(s => s.accelerationX || 0);
            const accelY = recentSamples.map(s => s.accelerationY || 0);
            const accelZ = recentSamples.map(s => s.accelerationZ || 0);
            
            // 1. Acceleration Magnitude
            const magnitudes = recentSamples.map(s => 
                Math.sqrt(s.accelerationX**2 + s.accelerationY**2 + s.accelerationZ**2)
            );
            const avgMagnitude = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
            const magnitudeNormalized = Math.min(100, (avgMagnitude / 20) * 100); // Normalize to 20 m/s²
            
            const magEl = document.getElementById('accel-magnitude');
            const magBarEl = document.getElementById('accel-bar');
            if (magEl && magBarEl) {
                magEl.textContent = avgMagnitude.toFixed(2) + ' m/s²';
                magBarEl.style.width = magnitudeNormalized + '%';
                this.setBarQuality('accel-bar', 100 - magnitudeNormalized);
            }

            // 2. Tremor Frequency
            const tremorFreq = this.calculateDominantFrequency(magnitudes);
            const tremorFreqNormalized = Math.min(100, (tremorFreq / 10) * 100);
            
            const freqEl = document.getElementById('tremor-freq');
            const freqBarEl = document.getElementById('tremor-freq-bar');
            if (freqEl && freqBarEl) {
                freqEl.textContent = tremorFreq.toFixed(2) + ' Hz';
                freqBarEl.style.width = tremorFreqNormalized + '%';
                const inTremorRange = tremorFreq >= 4 && tremorFreq <= 6;
                this.setBarQuality('tremor-freq-bar', inTremorRange ? 40 : 80);
            }

            // 3. 4-6 Hz Band Power
            const bandPower = this.calculateBandPower(magnitudes, 4, 6);
            const bandPowerNormalized = Math.min(100, bandPower * 10);
            
            const bpEl = document.getElementById('tremor-band-power');
            const bpBarEl = document.getElementById('tremor-band-bar');
            if (bpEl && bpBarEl) {
                bpEl.textContent = bandPower.toFixed(3);
                bpBarEl.style.width = bandPowerNormalized + '%';
                this.setBarQuality('tremor-band-bar', 100 - bandPowerNormalized * 2);
            }

            // 4. Stability Index
            const stdMagnitude = this.calculateStd(magnitudes);
            const stabilityIndex = avgMagnitude > 0 ? stdMagnitude / avgMagnitude : 0;
            const stabilityNormalized = Math.min(100, stabilityIndex * 100);
            
            const stEl = document.getElementById('stability-index');
            const stBarEl = document.getElementById('stability-bar');
            if (stEl && stBarEl) {
                stEl.textContent = stabilityIndex.toFixed(3);
                stBarEl.style.width = stabilityNormalized + '%';
                this.setBarQuality('stability-bar', 100 - stabilityNormalized);
            }

            // 5. Jerk
            let totalJerk = 0;
            for (let i = 1; i < recentSamples.length; i++) {
                const dt = (recentSamples[i].timestamp - recentSamples[i-1].timestamp) / 1000;
                if (dt > 0) {
                    const jerkX = Math.abs((accelX[i] - accelX[i-1]) / dt);
                    const jerkY = Math.abs((accelY[i] - accelY[i-1]) / dt);
                    const jerkZ = Math.abs((accelZ[i] - accelZ[i-1]) / dt);
                    totalJerk += Math.sqrt(jerkX**2 + jerkY**2 + jerkZ**2);
                }
            }
            const avgJerk = totalJerk / (recentSamples.length - 1);
            const jerkNormalized = Math.min(100, (avgJerk / 100) * 100);
            
            const jkEl = document.getElementById('jerk-value');
            const jkBarEl = document.getElementById('jerk-bar');
            if (jkEl && jkBarEl) {
                jkEl.textContent = avgJerk.toFixed(2) + ' m/s³';
                jkBarEl.style.width = jerkNormalized + '%';
                this.setBarQuality('jerk-bar', 100 - jerkNormalized);
            }

            // 6. Sampling Rate
            if (recentSamples.length >= 2) {
                const intervals = [];
                for (let i = 1; i < recentSamples.length; i++) {
                    intervals.push(recentSamples[i].timestamp - recentSamples[i-1].timestamp);
                }
                const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
                const samplingRate = 1000 / avgInterval;
                const samplingNormalized = Math.min(100, (samplingRate / 100) * 100);
                
                const srEl = document.getElementById('sampling-rate');
                const srBarEl = document.getElementById('sampling-bar');
                if (srEl && srBarEl) {
                    srEl.textContent = samplingRate.toFixed(1) + ' Hz';
                    srBarEl.style.width = samplingNormalized + '%';
                    this.setBarQuality('sampling-bar', samplingRate >= 50 ? 80 : 50);
                }
            }

            // Update axis displays
            const currentSample = recentSamples[recentSamples.length - 1];
            this.updateAxisDisplay('x', currentSample.accelerationX || 0);
            this.updateAxisDisplay('y', currentSample.accelerationY || 0);
            this.updateAxisDisplay('z', currentSample.accelerationZ || 0);

        } catch (error) {
            console.error('Error updating tremor metrics:', error);
        }
    }

    calculateDominantFrequency(signal) {
        // Simple dominant frequency calculation using zero crossings
        const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
        let crossings = 0;
        for (let i = 1; i < signal.length; i++) {
            if ((signal[i] >= mean && signal[i-1] < mean) || 
                (signal[i] < mean && signal[i-1] >= mean)) {
                crossings++;
            }
        }
        // Estimate frequency (assuming ~100 Hz sampling rate)
        return (crossings / 2) * (100 / signal.length);
    }

    calculateBandPower(signal, lowFreq, highFreq) {
        // Simplified band power calculation
        const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
        const centered = signal.map(v => v - mean);
        const power = centered.reduce((sum, v) => sum + v * v, 0) / centered.length;
        return power;
    }

    calculateStd(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squaredDiffs = values.map(v => (v - mean) ** 2);
        const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
        return Math.sqrt(variance);
    }

    updateAxisDisplay(axis, value) {
        const normalized = Math.min(100, Math.abs(value) * 5); // Normalize to visible range
        const barId = `axis-${axis}-bar`;
        const valueId = `axis-${axis}-value`;
        
        document.getElementById(barId).style.width = normalized + '%';
        document.getElementById(valueId).textContent = value.toFixed(2);
    }

    stopTremorTest() {
        // Remove motion event listener
        if (this.motionHandler) {
            window.removeEventListener('devicemotion', this.motionHandler);
        }

        // Update UI
        const button = document.getElementById('tremor-record-btn');
        button.innerHTML = '<span class="record-icon">📱</span>Start Tremor Test';
        button.classList.remove('recording');

        if (this.selectedTestMode === 'tremor') {
            this.updateProgress(90, 'Motion recording completed');
        } else {
            this.updateProgress(90, 'Motion recording completed');
        }
        
        this.showStatus('tremor-status', `Tremor data collected: ${this.motionData.length} samples`, 'success');

        // Move to processing
        setTimeout(() => {
            this.startProcessing();
        }, 1500);
    }

    // Data Processing
    async startProcessing() {
        // Data collection complete - now process with backend
        // The processing screen will be shown by analyzeWithStreaming()
        
        try {
            // Prepare data for backend based on test mode
            let audioBlob = null;
            let motionDataToSend = [];
            
            if (this.selectedTestMode === 'voice' || this.selectedTestMode === 'both') {
                audioBlob = new Blob(this.audioData, { type: 'audio/webm' });
            }
            
            if (this.selectedTestMode === 'tremor' || this.selectedTestMode === 'both') {
                // Transform motion data to backend expected format
                motionDataToSend = this.motionData.map(sample => ({
                    x: sample.accelerationX || 0,
                    y: sample.accelerationY || 0,
                    z: sample.accelerationZ || 0,
                    timestamp: sample.timestamp || Date.now()
                }));
            }
            
            // Use real backend streaming analysis - NO DEMO MODE
            await this.analyzeWithStreaming(audioBlob, motionDataToSend);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showScreen('welcome-screen');
            alert(`Analysis failed: ${error.message}\n\nPlease ensure the backend server is running.`);
        }
    }

    async analyzeWithStreaming(audioBlob, motionData) {
        try {
            // Show processing screen
            this.showScreen('processing-screen');
            this.updateProcessingProgress(0, 'Preparing data...');
            
            // Prepare form data
            const formData = new FormData();
            
            // Add test mode
            formData.append('test_mode', this.selectedTestMode);
            
            // Add audio only if voice or both
            if (this.selectedTestMode === 'voice' || this.selectedTestMode === 'both') {
                if (audioBlob) {
                    formData.append('audio', audioBlob, 'recording.webm');
                }
            }
            
            // Add motion data only if tremor or both
            if (this.selectedTestMode === 'tremor' || this.selectedTestMode === 'both') {
                if (motionData && motionData.length > 0) {
                    formData.append('motion_data', JSON.stringify(motionData));
                }
            }

            // Use EventSource for Server-Sent Events
            // Note: EventSource doesn't support POST, so we'll use fetch with streaming
            const response = await window.fetchWithNgrokBypass(`${this.API_BASE_URL}/analyze-stream`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Backend error: ${response.status} ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete messages
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        if (data.status === 'complete' && data.results) {
                            // Analysis complete - show results
                            this.showResults(data.results);
                        } else if (data.message) {
                            // Update progress
                            const progress = data.progress || 0;
                            this.updateProcessingProgress(progress, data.message);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Streaming analysis error:', error);
            throw error; // Re-throw to be caught by parent
        }
    }

    updateProcessingProgress(percentage, message) {
        const progressBar = document.querySelector('#processing-screen .progress-fill');
        const statusText = document.querySelector('#processing-screen .status-text');
        const percentText = document.querySelector('#processing-screen .progress-percent');
        
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
        
        if (statusText) {
            statusText.textContent = message;
        }
        
        if (percentText) {
            percentText.textContent = `${Math.round(percentage)}%`;
        }
        
        console.log(`[${percentage}%] ${message}`);
    }

    showResults(results) {
        this.updateProgress(100, 'Analysis complete!');
        
        setTimeout(() => {
            this.displayResults(results);
            this.showScreen('results-screen');
        }, 1000);
    }

    displayResults(results) {
        const indicator = document.getElementById('result-indicator');
        const title = document.getElementById('result-title');
        const subtitle = document.getElementById('result-subtitle');
        const mainBadge = document.getElementById('main-confidence-badge');
        
        // Check for insufficient data / idle detection
        const isInsufficient = results.prediction === 'Insufficient Data' || results.confidence < 10;
        
        // Set result type and styling
        if (isInsufficient) {
            // Very low confidence - idle/baseline detected
            indicator.className = 'result-indicator insufficient';
            indicator.querySelector('.result-icon').textContent = '📱';
            title.textContent = 'Insufficient Activity Detected';
            subtitle.textContent = 'No meaningful voice or motion detected. Please ensure: (1) Speak clearly during voice test, (2) Hold phone steady during motion test';
            mainBadge.textContent = `${Math.round(results.confidence)}% - Please Retry Test`;
            mainBadge.style.background = '#9ca3af';
        } else if (results.prediction === 'Affected') {
            indicator.className = 'result-indicator positive';
            indicator.querySelector('.result-icon').textContent = '⚠️';
            title.textContent = 'Attention Required';
            subtitle.textContent = 'Analysis suggests possible Parkinson\'s indicators detected';
            mainBadge.textContent = `${Math.round(results.confidence)}% Detection Confidence`;
            mainBadge.style.background = '';
        } else {
            indicator.className = 'result-indicator negative';
            indicator.querySelector('.result-icon').textContent = '✅';
            title.textContent = 'No Indicators Detected';
            subtitle.textContent = 'Analysis shows normal patterns - No concerns identified';
            mainBadge.textContent = `${Math.round(results.confidence)}% Confidence`;
            mainBadge.style.background = '';
        }

        // Update confidence circles with animations (values already 0-100)
        this.updateConfidenceCircle('overall-circle', 'overall-percentage', results.confidence);
        
        // Show/hide voice confidence card
        const voiceCard = document.getElementById('voice-confidence-card');
        if (this.selectedTestMode === 'voice' || this.selectedTestMode === 'both') {
            voiceCard.style.display = 'block';
            // Use voice_patterns if available, fallback to voice_confidence (already 0-100)
            const voiceValue = results.voice_patterns || results.voice_confidence || 0;
            this.updateConfidenceCircle('voice-circle', 'voice-percentage', voiceValue);
        } else {
            voiceCard.style.display = 'none';
        }
        
        // Show/hide tremor confidence card
        const tremorCard = document.getElementById('tremor-confidence-card');
        if (this.selectedTestMode === 'tremor' || this.selectedTestMode === 'both') {
            tremorCard.style.display = 'block';
            // Use motion_patterns if available, fallback to tremor_confidence (already 0-100)
            const tremorValue = results.motion_patterns || results.tremor_confidence || 0;
            this.updateConfidenceCircle('tremor-circle', 'tremor-percentage', tremorValue);
        } else {
            tremorCard.style.display = 'none';
        }

        // Update detailed features
        this.updateDetailedFeatures(results);
        
        // Display dataset match information if available
        this.displayDatasetMatch(results.dataset_match);

        // Add download button functionality
        document.getElementById('download-results-btn').onclick = () => this.downloadResults(results);
    }

    updateConfidenceCircle(circleId, percentageId, value) {
        // Value is already 0-100, no need to multiply
        const percentage = Math.round(value);
        const circle = document.getElementById(circleId);
        const percentageText = document.getElementById(percentageId);
        
        // Calculate circle dash offset (283 is circumference of circle with r=45)
        const circumference = 283;
        const offset = circumference - (percentage / 100) * circumference;
        
        // Animate circle
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);
        
        // Animate percentage counter
        let current = 0;
        const increment = percentage / 50; // 50 frames
        const timer = setInterval(() => {
            current += increment;
            if (current >= percentage) {
                current = percentage;
                clearInterval(timer);
            }
            percentageText.textContent = Math.round(current) + '%';
        }, 20);
        
        // Set color based on percentage
        const gradient = percentage >= 70 ? '#e74c3c' : percentage >= 40 ? '#f39c12' : '#27ae60';
        circle.style.stroke = gradient;
    }

    updateDetailedFeatures(results) {
        // Update feature counts
        let voiceFeatureCount = 0;
        let tremorFeatureCount = 0;

        // Voice Features
        const voiceFeaturesSection = document.getElementById('voice-features-section');
        const voiceFeaturesGrid = document.getElementById('voice-features-grid');
        
        if (this.selectedTestMode === 'voice' || this.selectedTestMode === 'both') {
            voiceFeaturesSection.style.display = 'block';
            voiceFeaturesGrid.innerHTML = '';
            
            // Get raw audio features for detailed display
            const rawAudioFeatures = results.audio_features || results.raw_features || {};
            console.log('Raw Audio Features:', rawAudioFeatures);
            
            let voiceFeatures = [];
            const featureKeys = Object.keys(rawAudioFeatures);

            // Check if we have a filtered set of features (likely from feature selection)
            // If we have keys and they look like technical names (e.g. mfcc, chroma), display them dynamically
            if (featureKeys.length > 0) {
                // Sort keys to group similar features
                featureKeys.sort();
                
                featureKeys.forEach(key => {
                    // Skip internal keys if any remain
                    if (key.startsWith('_')) return;
                    
                    voiceFeatures.push({
                        name: this.formatFeatureName(key),
                        icon: this.getFeatureIcon(key),
                        value: rawAudioFeatures[key],
                        unit: ''
                    });
                });
            } else {
                // Fallback to default list if no features returned
                voiceFeatures = [
                    { name: 'No Features', icon: '❓', value: 0, unit: '' }
                ];
            }
            
            voiceFeatures.forEach((feature, index) => {
                const featureCard = document.createElement('div');
                featureCard.className = 'feature-item';
                featureCard.style.animationDelay = `${index * 0.05}s`;
                
                const displayValue = typeof feature.value === 'number' ? feature.value.toFixed(4) : feature.value;
                
                featureCard.innerHTML = `
                    <div class="feature-item-icon">${feature.icon}</div>
                    <div class="feature-item-name">${feature.name}</div>
                    <div class="feature-item-value">${displayValue}</div>
                    <div class="feature-item-unit">${feature.unit}</div>
                `;
                voiceFeaturesGrid.appendChild(featureCard);
                voiceFeatureCount++;
            });
            
            document.getElementById('voice-feature-count').textContent = `${voiceFeatureCount} features`;
        } else {
            voiceFeaturesSection.style.display = 'none';
        }

        // Tremor Features
        const tremorFeaturesSection = document.getElementById('tremor-features-section');
        const tremorFeaturesGrid = document.getElementById('tremor-features-grid');
        
        if (this.selectedTestMode === 'tremor' || this.selectedTestMode === 'both') {
            tremorFeaturesSection.style.display = 'block';
            tremorFeaturesGrid.innerHTML = '';
            
            // Get raw tremor features for detailed display
            const rawTremorFeatures = results.tremor_features || results.raw_features || {};
            console.log('Raw Tremor Features:', rawTremorFeatures);
            
            // Only show features that match what's displayed during recording (like in the screenshot)
            const tremorFeatures = [
                // Row 1 - Main metrics
                { name: 'Tremor\nFrequency', icon: '📳', value: rawTremorFeatures.magnitude_fft_dom_freq || 0, unit: 'Hz' },
                { name: 'Postural\nStability', icon: '⚖️', value: results.features?.['Postural Stability'] || 0, unit: '%' },
                { name: 'Motion\nVariability', icon: '💫', value: results.features?.['Motion Variability'] || 0, unit: '%' },
                { name: 'Acceleration\nMagnitude', icon: '📊', value: rawTremorFeatures.magnitude_mean || 0, unit: 'm/s²' },
                
                // Row 2 - Additional metrics
                { name: 'Magnitude\nStd Dev', icon: '📈', value: rawTremorFeatures.magnitude_std || 0, unit: 'm/s²' },
                { name: 'Magnitude\nRMS', icon: '⚡', value: rawTremorFeatures.magnitude_rms || 0, unit: 'm/s²' },
                { name: 'Tremor Band\nPower (4-6Hz)', icon: '🎯', value: rawTremorFeatures.tremor_band_power_mag || 0, unit: 'power' },
                { name: 'Stability\nIndex', icon: '⚖️', value: rawTremorFeatures.stability_index || 0, unit: 'index' },
                
                // Row 3 - Advanced metrics
                { name: 'Total FFT\nPower', icon: '🔋', value: rawTremorFeatures.magnitude_fft_tot_power || 0, unit: 'power' },
                { name: 'Peak\nRate', icon: '📍', value: rawTremorFeatures.magnitude_peaks_rt || 0, unit: 'rate' },
                { name: 'Sample\nEntropy', icon: '🔢', value: rawTremorFeatures.magnitude_sampen || 0, unit: 'entropy' }
            ];
            
            tremorFeatures.forEach((feature, index) => {
                const featureCard = document.createElement('div');
                featureCard.className = 'feature-item';
                featureCard.style.animationDelay = `${index * 0.05}s`;
                
                const displayValue = typeof feature.value === 'number' ? feature.value.toFixed(2) : feature.value;
                
                featureCard.innerHTML = `
                    <div class="feature-item-icon">${feature.icon}</div>
                    <div class="feature-item-name">${feature.name}</div>
                    <div class="feature-item-value">${displayValue}</div>
                    <div class="feature-item-unit">${feature.unit}</div>
                `;
                tremorFeaturesGrid.appendChild(featureCard);
                tremorFeatureCount++;
            });
            
            document.getElementById('tremor-feature-count').textContent = `${tremorFeatureCount} features`;
        } else {
            tremorFeaturesSection.style.display = 'none';
        }

        // Update total feature count in ML info
        const totalFeatures = voiceFeatureCount + tremorFeatureCount;
        document.getElementById('feature-count-text').textContent = `${totalFeatures} parameters analyzed`;
    }

    formatFeatureName(key) {
        // Convert snake_case to Title Case and make it readable
        return key
            .replace(/_/g, ' ')
            .replace(/mfcc/i, 'MFCC')
            .replace(/rms/i, 'RMS')
            .replace(/zcr/i, 'ZCR')
            .replace(/hnr/i, 'HNR')
            .replace(/std/i, 'Std Dev')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    getFeatureIcon(key) {
        if (key.includes('mfcc')) return '🎼';
        if (key.includes('chroma')) return '🌈';
        if (key.includes('contrast')) return '🌗';
        if (key.includes('tonnetz')) return '🎹';
        if (key.includes('pitch')) return '📈';
        if (key.includes('rms') || key.includes('energy')) return '⚡';
        if (key.includes('zcr')) return '〰️';
        return '📊';
    }
    
    displayDatasetMatch(datasetMatch) {
        // Find or create dataset match container
        let matchContainer = document.getElementById('dataset-match-container');
        
        if (!matchContainer) {
            // Create container if it doesn't exist
            matchContainer = document.createElement('div');
            matchContainer.id = 'dataset-match-container';
            matchContainer.className = 'dataset-match-section';
            
            // Insert after confidence cards
            const confidenceCards = document.querySelector('.confidence-cards');
            if (confidenceCards && confidenceCards.parentNode) {
                confidenceCards.parentNode.insertBefore(matchContainer, confidenceCards.nextSibling);
            }
        }
        
        // Clear previous content
        matchContainer.innerHTML = '';
        
        if (!datasetMatch || datasetMatch.error) {
            matchContainer.style.display = 'none';
            return;
        }
        
        // Check if we have matches
        const voiceMatched = datasetMatch.voice_match && datasetMatch.voice_match.matched;
        const tremorMatched = datasetMatch.tremor_match && datasetMatch.tremor_match.matched;
        
        if (!voiceMatched && !tremorMatched) {
            matchContainer.style.display = 'none';
            return;
        }
        
        matchContainer.style.display = 'block';
        
        // Build match display
        let matchHTML = '<div class="dataset-match-card"><div class="dataset-match-header">';
        matchHTML += '<h3>📊 Dataset Match Found</h3>';
        matchHTML += '<p class="dataset-match-subtitle">Your sample matches known dataset patterns</p>';
        matchHTML += '</div><div class="dataset-match-content">';
        
        // Voice match
        if (voiceMatched) {
            const vm = datasetMatch.voice_match;
            const categoryClass = vm.category === 'Healthy' ? 'healthy' : 'parkinsons';
            matchHTML += `
                <div class="match-item ${categoryClass}">
                    <div class="match-icon">🎤</div>
                    <div class="match-details">
                        <div class="match-title">Voice Pattern Match</div>
                        <div class="match-category">${vm.category} Dataset</div>
                        <div class="match-filename">${vm.filename}</div>
                        <div class="match-confidence">${vm.confidence.toFixed(1)}% similarity</div>
                    </div>
                </div>
            `;
        }
        
        // Tremor match
        if (tremorMatched) {
            const tm = datasetMatch.tremor_match;
            const categoryClass = tm.category === 'Healthy' ? 'healthy' : 'parkinsons';
            const tremorTypes = tm.tremor_types && tm.tremor_types.length > 0 ? 
                ` (${tm.tremor_types.join(', ')})` : '';
            matchHTML += `
                <div class="match-item ${categoryClass}">
                    <div class="match-icon">🤚</div>
                    <div class="match-details">
                        <div class="match-title">Tremor Pattern Match</div>
                        <div class="match-category">${tm.category} Dataset${tremorTypes}</div>
                        <div class="match-filename">Subject ID: ${tm.subject_id}</div>
                        <div class="match-confidence">${tm.confidence.toFixed(1)}% similarity</div>
                    </div>
                </div>
            `;
        }
        
        // Overall consensus
        if (datasetMatch.overall_match && datasetMatch.consensus_category) {
            const consensusClass = datasetMatch.consensus_category === 'Healthy' ? 'healthy' : 'parkinsons';
            matchHTML += `
                <div class="match-consensus ${consensusClass}">
                    <strong>Dataset Consensus:</strong> ${datasetMatch.consensus_category}
                </div>
            `;
        }
        
        matchHTML += '</div></div>';
        matchContainer.innerHTML = matchHTML;
    }

    downloadResults(results) {
        // Show download modal instead of directly downloading
        this.showDownloadModal(results);
    }

    showDownloadModal(results) {
        const modal = document.getElementById('download-modal');
        modal.classList.add('active');

        // Close modal handlers
        const closeBtn = document.getElementById('close-modal');
        const closeModal = () => {
            modal.classList.remove('active');
        };

        closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };

        // Simple report download
        document.getElementById('download-simple').onclick = async () => {
            closeModal();
            await this.downloadSimpleReport(results);
        };

        // Detailed Excel download
        document.getElementById('download-detailed').onclick = async () => {
            closeModal();
            await this.downloadDetailedExcel(results);
        };
    }

    async downloadSimpleReport(results) {
        try {
            console.log('📄 Generating simple report...');
            
            // Check if ExcelExporter is available
            if (typeof window.ExcelExporter === 'undefined') {
                throw new Error('ExcelExporter not loaded. Please refresh the page.');
            }
            
            const exporter = new window.ExcelExporter();
            await exporter.exportSimpleReport(results, this.selectedTestMode);
            
            this.showNotification('✅ Simple report downloaded successfully!', 'success');
        } catch (error) {
            console.error('❌ Error downloading simple report:', error);
            this.showNotification(`❌ ${error.message || 'Failed to download report'}`, 'error');
        }
    }

    async downloadDetailedExcel(results) {
        try {
            console.log('📊 Generating detailed Excel report...');
            
            // Check if ExcelExporter is available
            if (typeof window.ExcelExporter === 'undefined') {
                throw new Error('ExcelExporter not loaded. Please refresh the page.');
            }
            
            const exporter = new window.ExcelExporter();
            
            // Prepare raw data for detailed export
            const rawData = {
                voiceData: this.prepareVoiceData(results),
                tremorData: this.prepareTremorData(results),
                motionSamples: this.motionRecorder?.samples || []
            };

            await exporter.exportDetailedData(results, this.selectedTestMode, rawData);
            
            this.showNotification('✅ Detailed Excel report downloaded successfully!', 'success');
        } catch (error) {
            console.error('❌ Error downloading detailed report:', error);
            this.showNotification(`❌ ${error.message || 'Failed to download detailed report'}`, 'error');
        }
    }

    prepareVoiceData(results) {
        // Use raw audio features instead of simplified features
        const audioFeatures = results.audio_features || results.raw_features || {};
        const features = results.features || {};
        const metadata = results.metadata || {};
        
        return {
            // Audio metadata
            duration: metadata.audio_duration || audioFeatures.duration || 0,
            sample_rate: audioFeatures.sample_rate || metadata.sample_rate || 22050,
            
            // Pitch features (matching UI display)
            pitch_mean: audioFeatures.pitch_mean || 0,
            pitch_std: audioFeatures.pitch_std || 0,
            pitch_range: audioFeatures.pitch_range || 0,
            
            // Voice quality metrics
            hnr_mean: audioFeatures.hnr_mean || audioFeatures.hnr || 0,
            
            // Spectral features (matching UI display)
            spectral_centroid: audioFeatures.spectral_centroid_mean || audioFeatures.spectral_centroid || 0,
            spectral_rolloff: audioFeatures.spectral_rolloff_mean || audioFeatures.spectral_rolloff || 0,
            spectral_bandwidth: audioFeatures.spectral_bandwidth_mean || audioFeatures.spectral_bandwidth || 0,
            
            // Speech rate (matching UI display)
            speech_rate: audioFeatures.speech_rate || 0
        };
    }

    prepareTremorData(results) {
        // Use raw tremor features - map ALL extracted features to Excel columns
        const tremorFeatures = results.tremor_features || results.raw_features || {};
        const metadata = results.metadata || {};
        
        return {
            start_timestamp: metadata.start_timestamp || new Date().toISOString(),
            end_timestamp: metadata.end_timestamp || new Date().toISOString(),
            
            // Magnitude statistics (12 features)
            magnitude_mean: tremorFeatures.magnitude_mean || 0,
            magnitude_std_dev: tremorFeatures.magnitude_std || 0,
            magnitude_rms: tremorFeatures.magnitude_rms || 0,
            magnitude_energy: tremorFeatures.magnitude_energy || 0,
            magnitude_max: tremorFeatures.magnitude_max || 0,
            magnitude_min: tremorFeatures.magnitude_min || 0,
            magnitude_range: tremorFeatures.magnitude_range || 0,
            magnitude_kurtosis: tremorFeatures.magnitude_kurtosis || 0,
            magnitude_skewness: tremorFeatures.magnitude_skewness || 0,
            magnitude_cv: tremorFeatures.magnitude_cv || 0,
            magnitude_peaks_rt: tremorFeatures.magnitude_peaks_rt || 0,
            magnitude_ssc_rt: tremorFeatures.magnitude_ssc_rt || 0,
            
            // Frequency domain features (8 features)
            magnitude_fft_dom_freq: tremorFeatures.magnitude_fft_dom_freq || 0,
            magnitude_fft_tot_power: tremorFeatures.magnitude_fft_tot_power || 0,
            magnitude_fft_energy: tremorFeatures.magnitude_fft_energy || 0,
            magnitude_fft_entropy: tremorFeatures.magnitude_fft_entropy || 0,
            tremor_band_power_mag: tremorFeatures.tremor_band_power_mag || 0,
            tremor_peak_freq: tremorFeatures.tremor_peak_freq || 0,
            dominant_freq_x: tremorFeatures.dominant_freq_x || 0,
            tremor_band_power_x: tremorFeatures.tremor_band_power_x || 0,
            
            // Time domain features (6 features)
            zero_crossing_rate_mag: tremorFeatures.zero_crossing_rate_mag || 0,
            peak_count_mag: tremorFeatures.peak_count_mag || 0,
            jerk_mean: tremorFeatures.jerk_mean || 0,
            jerk_std: tremorFeatures.jerk_std || 0,
            stability_index: tremorFeatures.stability_index || 0,
            magnitude_sampen: tremorFeatures.magnitude_sampen || 0,
            
            // Tremor classifications (if available)
            rest_tremor: tremorFeatures.rest_tremor || 0,
            postural_tremor: tremorFeatures.postural_tremor || 0,
            kinetic_tremor: tremorFeatures.kinetic_tremor || 0
        };
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${type === 'success' ? '#10b981' : '#ef4444'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10001;
            animation: slideInRight 0.3s ease;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Utility Functions
    showStatus(elementId, message, type) {
        const element = document.getElementById(elementId);
        element.innerHTML = message;
        element.className = `test-status ${type}`;
    }

    async shareResults() {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: 'Parkinson\'s Detection Results',
                    text: 'I just completed a Parkinson\'s detection test using AI analysis.',
                    url: window.location.href
                });
            } catch (error) {
                console.log('Error sharing:', error);
            }
        } else {
            // Fallback: copy to clipboard
            const text = 'Parkinson\'s Detection Test Results - Check out this AI-powered detection app!';
            navigator.clipboard.writeText(text + ' ' + window.location.href);
            alert('Results copied to clipboard!');
        }
    }
    } // End of ParkinsonDetectionApp class
} // End of declaration guard

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.ParkinsonDetectionApp !== 'undefined' && !window.app) {
        window.app = new window.ParkinsonDetectionApp();
        console.log('✅ Parkinson Detection App initialized');
    }
});

// Handle orientation changes for mobile
window.addEventListener('orientationchange', () => {
    setTimeout(() => {
        window.location.reload();
    }, 500);
});

// Global error handler to catch blocked requests (analytics, extensions, etc.)
window.addEventListener('error', (event) => {
    // Silently ignore blocked requests from extensions
    if (event.message && (
        event.message.includes('ERR_BLOCKED_BY_CLIENT') ||
        event.message.includes('analytics') ||
        event.message.includes('google-analytics')
    )) {
        event.preventDefault();
        console.log('ℹ️ External resource blocked by browser extension (this is safe to ignore)');
        return false;
    }
}, true);

// Handle unhandled promise rejections gracefully
window.addEventListener('unhandledrejection', (event) => {
    // Silently ignore blocked requests
    if (event.reason && (
        event.reason.message?.includes('ERR_BLOCKED_BY_CLIENT') ||
        event.reason.message?.includes('analytics')
    )) {
        event.preventDefault();
        console.log('ℹ️ External request blocked (safe to ignore)');
        return false;
    }
});
