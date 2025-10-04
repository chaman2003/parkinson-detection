// Sensor Test Page - Complete Implementation
// Tests microphone, accelerometer, and gyroscope functionality

class SensorTestApp {
    constructor() {
        this.micStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.micTesting = false;
        this.accelTesting = false;
        this.gyroTesting = false;
        this.accelHandler = null;
        this.gyroHandler = null;
        this.accelCount = 0;
        this.accelStartTime = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.displaySystemInfo();
        this.log('INFO', 'Sensor test page loaded');
    }

    setupEventListeners() {
        // Permission buttons
        document.getElementById('request-motion-btn')?.addEventListener('click', () => {
            this.requestMotionPermission();
        });

        document.getElementById('request-mic-btn')?.addEventListener('click', () => {
            this.requestMicrophonePermission();
        });

        document.getElementById('request-all-btn')?.addEventListener('click', () => {
            this.requestAllPermissions();
        });

        // Microphone test buttons
        document.getElementById('mic-test-btn')?.addEventListener('click', () => {
            this.startMicrophoneTest();
        });

        document.getElementById('mic-stop-btn')?.addEventListener('click', () => {
            this.stopMicrophoneTest();
        });

        // Accelerometer test buttons
        document.getElementById('accel-test-btn')?.addEventListener('click', () => {
            this.startAccelerometerTest();
        });

        document.getElementById('accel-stop-btn')?.addEventListener('click', () => {
            this.stopAccelerometerTest();
        });

        // Gyroscope test buttons
        document.getElementById('gyro-test-btn')?.addEventListener('click', () => {
            this.startGyroscopeTest();
        });

        document.getElementById('gyro-stop-btn')?.addEventListener('click', () => {
            this.stopGyroscopeTest();
        });
    }

    // =========================================================================
    // Permission Management
    // =========================================================================

    async requestMotionPermission() {
        this.log('INFO', 'Requesting motion sensor permission...');
        this.setStatus('permission-status', 'warning');
        
        try {
            // Check if DeviceMotionEvent exists
            if (typeof DeviceMotionEvent === 'undefined') {
                throw new Error('DeviceMotionEvent not supported by this browser');
            }

            // iOS 13+ requires explicit permission
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                const permission = await DeviceMotionEvent.requestPermission();
                if (permission === 'granted') {
                    this.log('SUCCESS', 'Motion sensor permission granted');
                    this.setStatus('permission-status', 'success');
                    this.showMessage('permission-message', 'Motion sensors enabled!', 'success');
                } else {
                    throw new Error('Motion sensor permission denied');
                }
            } else {
                // Non-iOS or older browsers - permission not needed
                this.log('SUCCESS', 'Motion sensors available (no permission needed)');
                this.setStatus('permission-status', 'success');
                this.showMessage('permission-message', 'Motion sensors available!', 'success');
            }
        } catch (error) {
            this.log('ERROR', 'Motion permission failed: ' + error.message);
            this.setStatus('permission-status', 'error');
            this.showMessage('permission-message', 'Failed: ' + error.message, 'error');
        }
    }

    async requestMicrophonePermission() {
        this.log('INFO', 'Requesting microphone permission...');
        this.setStatus('permission-status', 'warning');
        
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Microphone access not available (HTTPS required)');
            }

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop()); // Stop immediately
            
            this.log('SUCCESS', 'Microphone permission granted');
            this.setStatus('permission-status', 'success');
            this.showMessage('permission-message', 'Microphone enabled!', 'success');
        } catch (error) {
            this.log('ERROR', 'Microphone permission failed: ' + error.message);
            this.setStatus('permission-status', 'error');
            this.showMessage('permission-message', 'Failed: ' + error.message, 'error');
        }
    }

    async requestAllPermissions() {
        this.log('INFO', 'Requesting all permissions...');
        this.setStatus('permission-status', 'warning');
        
        try {
            // Request microphone
            await this.requestMicrophonePermission();
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Request motion
            await this.requestMotionPermission();
            
            this.log('SUCCESS', 'All permissions granted');
            this.setStatus('permission-status', 'success');
            this.showMessage('permission-message', 'All permissions granted!', 'success');
        } catch (error) {
            this.log('ERROR', 'Permission request failed: ' + error.message);
            this.setStatus('permission-status', 'error');
            this.showMessage('permission-message', 'Some permissions failed', 'error');
        }
    }

    // =========================================================================
    // Microphone Test
    // =========================================================================

    async startMicrophoneTest() {
        if (this.micTesting) return;

        this.log('INFO', 'Starting microphone test...');
        this.setStatus('mic-status', 'warning');

        try {
            // Get microphone stream
            this.micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(this.micStream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            source.connect(this.analyser);

            // Update UI
            document.getElementById('mic-test-btn').disabled = true;
            document.getElementById('mic-stop-btn').disabled = false;
            document.getElementById('mic-status-text').textContent = 'Recording';
            document.getElementById('mic-sample-rate').textContent = this.audioContext.sampleRate + ' Hz';

            this.micTesting = true;
            this.setStatus('mic-status', 'success');
            this.showMessage('mic-message', 'Microphone active - speak into your device', 'success');
            this.log('SUCCESS', 'Microphone test started');

            // Start visualization
            this.visualizeMicrophone();

        } catch (error) {
            this.log('ERROR', 'Microphone test failed: ' + error.message);
            this.setStatus('mic-status', 'error');
            this.showMessage('mic-message', 'Error: ' + error.message, 'error');
        }
    }

    stopMicrophoneTest() {
        if (!this.micTesting) return;

        this.log('INFO', 'Stopping microphone test...');
        
        // Stop stream
        if (this.micStream) {
            this.micStream.getTracks().forEach(track => track.stop());
            this.micStream = null;
        }

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        // Update UI
        document.getElementById('mic-test-btn').disabled = false;
        document.getElementById('mic-stop-btn').disabled = true;
        document.getElementById('mic-status-text').textContent = 'Stopped';

        this.micTesting = false;
        this.setStatus('mic-status', 'success');
        this.showMessage('mic-message', 'Microphone test stopped', 'success');
        this.log('SUCCESS', 'Microphone test stopped');
    }

    visualizeMicrophone() {
        if (!this.micTesting || !this.analyser) return;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const update = () => {
            if (!this.micTesting) return;

            this.analyser.getByteFrequencyData(dataArray);

            // Calculate audio level (RMS)
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i] * dataArray[i];
            }
            const rms = Math.sqrt(sum / dataArray.length);
            const dbLevel = 20 * Math.log10(rms / 255 + 0.0001);

            // Calculate dominant frequency
            let maxValue = 0;
            let maxIndex = 0;
            for (let i = 0; i < dataArray.length; i++) {
                if (dataArray[i] > maxValue) {
                    maxValue = dataArray[i];
                    maxIndex = i;
                }
            }
            const frequency = maxIndex * this.audioContext.sampleRate / this.analyser.fftSize;

            // Update UI
            document.getElementById('mic-level').textContent = dbLevel.toFixed(1);
            document.getElementById('mic-freq').textContent = frequency.toFixed(0);

            // Create visualization bars
            this.createAudioBars(dataArray);

            requestAnimationFrame(update);
        };

        update();
    }

    createAudioBars(dataArray) {
        const container = document.getElementById('audio-bars');
        if (!container) return;

        // Clear existing bars
        container.innerHTML = '';

        // Create 32 bars
        const numBars = 32;
        const step = Math.floor(dataArray.length / numBars);

        for (let i = 0; i < numBars; i++) {
            const value = dataArray[i * step];
            const height = Math.max(10, (value / 255) * 80); // 10-80px
            
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.height = height + 'px';
            bar.style.backgroundColor = this.getBarColor(value);
            container.appendChild(bar);
        }
    }

    getBarColor(value) {
        if (value > 200) return '#667eea';
        if (value > 100) return '#764ba2';
        return '#a0a0a0';
    }

    // =========================================================================
    // Accelerometer Test
    // =========================================================================

    startAccelerometerTest() {
        if (this.accelTesting) return;

        this.log('INFO', 'Starting accelerometer test...');
        this.setStatus('accel-status', 'warning');

        try {
            // Check if DeviceMotionEvent is available
            if (typeof DeviceMotionEvent === 'undefined') {
                throw new Error('DeviceMotionEvent not supported');
            }

            // Check secure context
            if (!window.isSecureContext) {
                throw new Error('Secure context (HTTPS) required');
            }

            this.accelCount = 0;
            this.accelStartTime = Date.now();

            // Add event listener
            this.accelHandler = (event) => {
                const accel = event.accelerationIncludingGravity;
                
                if (accel && accel.x !== null && accel.y !== null && accel.z !== null) {
                    this.accelCount++;
                    
                    const x = accel.x.toFixed(2);
                    const y = accel.y.toFixed(2);
                    const z = accel.z.toFixed(2);
                    const magnitude = Math.sqrt(accel.x**2 + accel.y**2 + accel.z**2).toFixed(2);

                    // Update UI
                    document.getElementById('accel-x').textContent = x;
                    document.getElementById('accel-y').textContent = y;
                    document.getElementById('accel-z').textContent = z;
                    document.getElementById('accel-mag').textContent = magnitude;
                    document.getElementById('accel-count').textContent = this.accelCount;

                    // Calculate sampling rate
                    const elapsed = (Date.now() - this.accelStartTime) / 1000;
                    const rate = this.accelCount / elapsed;
                    document.getElementById('accel-rate').textContent = rate.toFixed(1);
                }
            };

            window.addEventListener('devicemotion', this.accelHandler);

            // Update UI
            document.getElementById('accel-test-btn').disabled = true;
            document.getElementById('accel-stop-btn').disabled = false;

            this.accelTesting = true;
            this.setStatus('accel-status', 'success');
            this.showMessage('accel-message', 'Accelerometer active - move your device', 'success');
            this.log('SUCCESS', 'Accelerometer test started');

            // Check if data is coming in
            setTimeout(() => {
                if (this.accelCount === 0) {
                    this.log('WARNING', 'No accelerometer data received');
                    this.showMessage('accel-message', 'Warning: No data received. Check permissions.', 'warning');
                }
            }, 3000);

        } catch (error) {
            this.log('ERROR', 'Accelerometer test failed: ' + error.message);
            this.setStatus('accel-status', 'error');
            this.showMessage('accel-message', 'Error: ' + error.message, 'error');
        }
    }

    stopAccelerometerTest() {
        if (!this.accelTesting) return;

        this.log('INFO', 'Stopping accelerometer test...');
        
        // Remove event listener
        if (this.accelHandler) {
            window.removeEventListener('devicemotion', this.accelHandler);
            this.accelHandler = null;
        }

        // Update UI
        document.getElementById('accel-test-btn').disabled = false;
        document.getElementById('accel-stop-btn').disabled = true;

        this.accelTesting = false;
        this.setStatus('accel-status', 'success');
        this.showMessage('accel-message', `Accelerometer test stopped. Collected ${this.accelCount} samples.`, 'success');
        this.log('SUCCESS', 'Accelerometer test stopped');
    }

    // =========================================================================
    // Gyroscope Test
    // =========================================================================

    startGyroscopeTest() {
        if (this.gyroTesting) return;

        this.log('INFO', 'Starting gyroscope test...');
        this.setStatus('gyro-status', 'warning');

        try {
            // Check if DeviceOrientationEvent is available
            if (typeof DeviceOrientationEvent === 'undefined') {
                throw new Error('DeviceOrientationEvent not supported');
            }

            // Add event listener
            this.gyroHandler = (event) => {
                const alpha = (event.alpha || 0).toFixed(2);
                const beta = (event.beta || 0).toFixed(2);
                const gamma = (event.gamma || 0).toFixed(2);

                // Update UI
                document.getElementById('gyro-alpha').textContent = alpha;
                document.getElementById('gyro-beta').textContent = beta;
                document.getElementById('gyro-gamma').textContent = gamma;
                document.getElementById('gyro-status-text').textContent = 'Recording';
            };

            window.addEventListener('deviceorientation', this.gyroHandler);

            // Update UI
            document.getElementById('gyro-test-btn').disabled = true;
            document.getElementById('gyro-stop-btn').disabled = false;

            this.gyroTesting = true;
            this.setStatus('gyro-status', 'success');
            this.showMessage('gyro-message', 'Gyroscope active - rotate your device', 'success');
            this.log('SUCCESS', 'Gyroscope test started');

        } catch (error) {
            this.log('ERROR', 'Gyroscope test failed: ' + error.message);
            this.setStatus('gyro-status', 'error');
            this.showMessage('gyro-message', 'Error: ' + error.message, 'error');
        }
    }

    stopGyroscopeTest() {
        if (!this.gyroTesting) return;

        this.log('INFO', 'Stopping gyroscope test...');
        
        // Remove event listener
        if (this.gyroHandler) {
            window.removeEventListener('deviceorientation', this.gyroHandler);
            this.gyroHandler = null;
        }

        // Update UI
        document.getElementById('gyro-test-btn').disabled = false;
        document.getElementById('gyro-stop-btn').disabled = true;
        document.getElementById('gyro-status-text').textContent = 'Stopped';

        this.gyroTesting = false;
        this.setStatus('gyro-status', 'success');
        this.showMessage('gyro-message', 'Gyroscope test stopped', 'success');
        this.log('SUCCESS', 'Gyroscope test stopped');
    }

    // =========================================================================
    // System Information
    // =========================================================================

    displaySystemInfo() {
        const info = [];
        
        // Browser information
        info.push(`Browser: ${navigator.userAgent}`);
        info.push(`Platform: ${navigator.platform}`);
        info.push(`Language: ${navigator.language}`);
        info.push(`Online: ${navigator.onLine}`);
        info.push(`Cookies Enabled: ${navigator.cookieEnabled}`);
        
        // Protocol information
        info.push(`Protocol: ${window.location.protocol}`);
        info.push(`Secure Context: ${window.isSecureContext}`);
        
        // API support
        info.push(`\nAPI Support:`);
        info.push(`- getUserMedia: ${!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)}`);
        info.push(`- DeviceMotionEvent: ${typeof DeviceMotionEvent !== 'undefined'}`);
        info.push(`- DeviceOrientationEvent: ${typeof DeviceOrientationEvent !== 'undefined'}`);
        info.push(`- AudioContext: ${!!(window.AudioContext || window.webkitAudioContext)}`);
        
        // Permission API
        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            info.push(`- Motion Permission Required: Yes (iOS 13+)`);
        } else {
            info.push(`- Motion Permission Required: No`);
        }
        
        // Screen information
        info.push(`\nScreen:`);
        info.push(`- Resolution: ${window.screen.width} x ${window.screen.height}`);
        info.push(`- Available: ${window.screen.availWidth} x ${window.screen.availHeight}`);
        info.push(`- Color Depth: ${window.screen.colorDepth}-bit`);
        info.push(`- Pixel Ratio: ${window.devicePixelRatio}`);
        
        // Display in console
        const container = document.getElementById('system-info');
        if (container) {
            container.innerHTML = info.map(line => 
                `<div class="log-line">${line}</div>`
            ).join('');
        }
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

    setStatus(elementId, status) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.classList.remove('success', 'error', 'warning');
        if (status) {
            element.classList.add(status);
        }
    }

    showMessage(containerId, message, type) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const className = type === 'success' ? 'success-message' : 
                         type === 'error' ? 'error-message' : 'permission-warning';
        
        container.innerHTML = `<div class="${className}">${message}</div>`;
    }

    log(level, message) {
        const timestamp = new Date().toLocaleTimeString();
        const logLine = `[${timestamp}] [${level}] ${message}`;
        
        console.log(logLine);
        
        const container = document.getElementById('console-log');
        if (container) {
            const line = document.createElement('div');
            line.className = 'log-line';
            line.textContent = logLine;
            container.appendChild(line);
            
            // Keep only last 50 lines
            while (container.children.length > 50) {
                container.removeChild(container.firstChild);
            }
            
            // Auto-scroll to bottom
            container.scrollTop = container.scrollHeight;
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sensorTestApp = new SensorTestApp();
    console.log('✅ Sensor Test App initialized');
});
