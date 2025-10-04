/**
 * Excel Export Utility for Parkinson's Detection Test Results
 * Creates detailed Excel files matching the dataset format
 */

class ExcelExporter {
    constructor() {
        this.sheetJSLoaded = false;
    }

    /**
     * Load SheetJS library dynamically
     */
    async loadSheetJS() {
        if (this.sheetJSLoaded) return true;

        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.sheetjs.com/xlsx-0.20.1/package/dist/xlsx.full.min.js';
            script.onload = () => {
                this.sheetJSLoaded = true;
                console.log('✅ SheetJS loaded successfully');
                resolve(true);
            };
            script.onerror = () => {
                console.error('❌ Failed to load SheetJS');
                reject(new Error('Failed to load Excel export library'));
            };
            document.head.appendChild(script);
        });
    }

    /**
     * Generate simple report (PDF-style formatted Excel)
     */
    async exportSimpleReport(results, testMode) {
        await this.loadSheetJS();

        const timestamp = new Date().toISOString();
        const data = [
            ['Parkinson\'s Detection - Analysis Report'],
            [''],
            ['Test Information'],
            ['Date & Time:', new Date().toLocaleString()],
            ['Test Mode:', testMode.toUpperCase()],
            ['Test ID:', this.generateTestId()],
            [''],
            ['Results Summary'],
            ['Prediction:', results.prediction || 'N/A'],
            ['Overall Confidence:', `${Math.round(results.confidence || 0)}%`],
            [''],
        ];

        if (testMode === 'voice' || testMode === 'both') {
            data.push(
                ['Voice Analysis'],
                ['Voice Patterns:', `${Math.round(results.voice_patterns || results.voice_confidence || 0)}%`],
                ['Voice Confidence:', `${Math.round(results.voice_confidence || 0)}%`],
                ['Audio Duration:', `${results.metadata?.audio_duration || 0} seconds`],
                ['']
            );
            
            // Add key voice features if available
            if (results.audio_features) {
                data.push(
                    ['Key Voice Features'],
                    ['Pitch Mean:', `${results.audio_features.pitch_mean?.toFixed(2) || 'N/A'} Hz`],
                    ['Pitch Std Dev:', `${results.audio_features.pitch_std?.toFixed(2) || 'N/A'} Hz`],
                    ['Jitter:', `${results.audio_features.jitter_local?.toFixed(4) || 'N/A'}`],
                    ['Shimmer:', `${results.audio_features.shimmer_local?.toFixed(4) || 'N/A'}`],
                    ['HNR:', `${results.audio_features.hnr_mean?.toFixed(2) || 'N/A'} dB`],
                    ['']
                );
            }
        }

        if (testMode === 'tremor' || testMode === 'both') {
            data.push(
                ['Tremor Analysis'],
                ['Motion Patterns:', `${Math.round(results.motion_patterns || results.tremor_confidence || 0)}%`],
                ['Tremor Confidence:', `${Math.round(results.tremor_confidence || 0)}%`],
                ['Motion Samples:', results.metadata?.motion_samples || 0],
                ['']
            );
            
            // Add key tremor features if available
            if (results.tremor_features) {
                data.push(
                    ['Key Tremor Features'],
                    ['Acceleration Magnitude:', `${results.tremor_features.magnitude_mean?.toFixed(2) || 'N/A'} m/s²`],
                    ['Tremor Frequency:', `${results.tremor_features.magnitude_fft_dom_freq?.toFixed(2) || 'N/A'} Hz`],
                    ['Tremor Band Power (4-6Hz):', `${results.tremor_features.tremor_band_power_mag?.toFixed(4) || 'N/A'}`],
                    ['Motion Variability:', `${results.tremor_features.magnitude_std?.toFixed(2) || 'N/A'}`],
                    ['Stability Index:', `${results.tremor_features.stability_index?.toFixed(4) || 'N/A'}`],
                    ['Sample Entropy:', `${results.tremor_features.magnitude_sampen?.toFixed(2) || 'N/A'}`],
                    ['']
                );
            }
        }

        data.push(
            ['Machine Learning Details'],
            ['Processing Time:', `${results.metadata?.processing_time || 0} seconds`],
            ['Model Version:', results.metadata?.model_version || 'N/A'],
            ['Algorithms Used:', 'SVM, Random Forest, Gradient Boosting, XGBoost'],
            [''],
            ['Disclaimer'],
            ['This is a research tool and not a medical diagnosis.'],
            ['Please consult healthcare professionals for proper evaluation.']
        );

        const ws = XLSX.utils.aoa_to_sheet(data);
        
        // Set column widths
        ws['!cols'] = [
            { wch: 25 },
            { wch: 40 }
        ];

        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Report');

        XLSX.writeFile(wb, `Parkinson_Report_${this.getFilenameTimestamp()}.xlsx`);
        console.log('✅ Simple report exported');
    }

    /**
     * Generate detailed Excel with all sensor attributes (like CSV dataset)
     */
    async exportDetailedData(results, testMode, rawData) {
        await this.loadSheetJS();

        const wb = XLSX.utils.book_new();

        // Summary Sheet
        this.addSummarySheet(wb, results, testMode);

        // Voice Features Sheet (if applicable)
        if ((testMode === 'voice' || testMode === 'both') && rawData.voiceData) {
            this.addVoiceFeaturesSheet(wb, rawData.voiceData, results);
        }

        // Tremor Features Sheet (if applicable)
        if ((testMode === 'tremor' || testMode === 'both') && rawData.tremorData) {
            this.addTremorFeaturesSheet(wb, rawData.tremorData, results);
        }

        // Raw Data Sheet
        if (rawData.motionSamples) {
            this.addRawMotionDataSheet(wb, rawData.motionSamples);
        }

        XLSX.writeFile(wb, `Parkinson_Detailed_${this.getFilenameTimestamp()}.xlsx`);
        console.log('✅ Detailed data exported');
    }

    /**
     * Add summary sheet
     */
    addSummarySheet(wb, results, testMode) {
        const data = [
            ['Parkinson\'s Detection - Detailed Analysis'],
            [''],
            ['Test ID', this.generateTestId()],
            ['Timestamp', new Date().toISOString()],
            ['Test Mode', testMode],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A'],
            ['Overall Confidence', `${(results.confidence || 0).toFixed(2)}%`],
            ['Voice Patterns', `${(results.voice_patterns || results.voice_confidence || 0).toFixed(2)}%`],
            ['Voice Confidence', `${(results.voice_confidence || 0).toFixed(2)}%`],
            ['Motion Patterns', `${(results.motion_patterns || results.tremor_confidence || 0).toFixed(2)}%`],
            ['Tremor Confidence', `${(results.tremor_confidence || 0).toFixed(2)}%`],
            [''],
            ['METADATA'],
            ['Processing Time (s)', results.metadata?.processing_time || 0],
            ['Audio Duration (s)', results.metadata?.audio_duration || 0],
            ['Motion Samples Count', results.metadata?.motion_samples || 0],
            ['Model Version', results.metadata?.model_version || 'N/A'],
            ['Sample Rate (Hz)', results.metadata?.sampling_rate || 100]
        ];

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 30 }, { wch: 30 }];
        XLSX.utils.book_append_sheet(wb, ws, 'Summary');
    }

    /**
     * Add voice features sheet
     */
    addVoiceFeaturesSheet(wb, voiceData, results) {
        const headers = [
            'Feature Name',
            'Value',
            'Unit',
            'Confidence',
            'Description'
        ];

        const features = [
            ['Pitch Mean', voiceData.pitch_mean, 'Hz', voiceData.pitch_confidence, 'Average fundamental frequency'],
            ['Pitch Std Dev', voiceData.pitch_std, 'Hz', voiceData.pitch_confidence, 'Pitch variation'],
            ['Jitter', voiceData.jitter, '%', voiceData.jitter_confidence, 'Frequency variation'],
            ['Shimmer', voiceData.shimmer, 'dB', voiceData.shimmer_confidence, 'Amplitude variation'],
            ['HNR', voiceData.hnr, 'dB', voiceData.hnr_confidence, 'Harmonic-to-Noise Ratio'],
            ['Spectral Centroid', voiceData.spectral_centroid, 'Hz', voiceData.spectral_confidence, 'Frequency center of mass'],
            ['Zero Crossing Rate', voiceData.zcr, 'rate', voiceData.zcr_confidence, 'Signal polarity changes'],
            ['Energy', voiceData.energy, 'dB', voiceData.energy_confidence, 'Signal power'],
            ['MFCC Mean', voiceData.mfcc_mean, '', voiceData.mfcc_confidence, 'Mel-frequency cepstral coefficients'],
            ['Spectral Rolloff', voiceData.spectral_rolloff, 'Hz', voiceData.rolloff_confidence, 'Frequency below 85% energy'],
            ['Spectral Flux', voiceData.spectral_flux, '', voiceData.flux_confidence, 'Spectral change rate'],
            ['RMS Energy', voiceData.rms_energy, 'dB', voiceData.rms_confidence, 'Root mean square energy']
        ];

        const data = [headers, ...features];
        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [
            { wch: 25 },
            { wch: 15 },
            { wch: 10 },
            { wch: 12 },
            { wch: 40 }
        ];

        XLSX.utils.book_append_sheet(wb, ws, 'Voice Features');
    }

    /**
     * Add tremor features sheet (comprehensive with all features shown in UI)
     */
    addTremorFeaturesSheet(wb, tremorData, results) {
        // Create human-readable feature list matching UI display
        const readableHeaders = [
            'Feature Name',
            'Value',
            'Unit',
            'Category',
            'Description'
        ];
        
        const readableFeatures = [
            // Pattern Features (0-100 scale)
            ['Tremor Frequency Score', results.features?.['Tremor Frequency'] || 0, '%', 'Pattern', 'Overall tremor frequency pattern strength'],
            ['Postural Stability Score', results.features?.['Postural Stability'] || 0, '%', 'Pattern', 'Postural stability assessment'],
            ['Motion Variability Score', results.features?.['Motion Variability'] || 0, '%', 'Pattern', 'Motion variation pattern'],
            
            // Raw Acceleration Features
            ['Acceleration Magnitude', tremorData.magnitude_mean || 0, 'm/s²', 'Acceleration', 'Mean acceleration magnitude'],
            ['Magnitude Std Dev', tremorData.magnitude_std_dev || 0, 'm/s²', 'Acceleration', 'Standard deviation of magnitude'],
            ['Magnitude RMS', tremorData.magnitude_rms || 0, 'm/s²', 'Acceleration', 'Root mean square of magnitude'],
            ['Magnitude Energy', tremorData.magnitude_energy || 0, 'energy', 'Acceleration', 'Total energy in magnitude signal'],
            
            // Frequency Analysis
            ['Tremor Frequency', tremorData.magnitude_fft_dom_freq || 0, 'Hz', 'Frequency', 'Dominant frequency in motion'],
            ['Tremor Band Power (4-6Hz)', tremorData.tremor_band_power_mag || 0, 'power', 'Frequency', 'Power in Parkinson\'s tremor band'],
            ['Total FFT Power', tremorData.magnitude_fft_tot_power || 0, 'power', 'Frequency', 'Total frequency domain power'],
            ['FFT Energy', tremorData.magnitude_fft_energy || 0, 'energy', 'Frequency', 'Energy in frequency domain'],
            ['FFT Entropy', tremorData.magnitude_fft_entropy || 0, 'entropy', 'Frequency', 'Frequency domain entropy'],
            
            // Stability Metrics
            ['Stability Index', tremorData.stability_index || 0, 'index', 'Stability', 'Overall stability measurement'],
            ['Peak Rate', tremorData.magnitude_peaks_rt || 0, 'rate', 'Stability', 'Rate of signal peaks'],
            ['Sample Entropy', tremorData.magnitude_sampen || 0, 'entropy', 'Stability', 'Signal regularity measure'],
            ['DFA', tremorData.magnitude_dfa || 0, 'coefficient', 'Stability', 'Detrended fluctuation analysis'],
            
            // Tremor Classifications
            ['Rest Tremor', tremorData.rest_tremor || 0, 'binary', 'Classification', 'Rest tremor detection'],
            ['Postural Tremor', tremorData.postural_tremor || 0, 'binary', 'Classification', 'Postural tremor detection'],
            ['Kinetic Tremor', tremorData.kinetic_tremor || 0, 'binary', 'Classification', 'Kinetic tremor detection']
        ];
        
        const readableData = [readableHeaders, ...readableFeatures];
        const ws1 = XLSX.utils.aoa_to_sheet(readableData);
        ws1['!cols'] = [
            { wch: 30 },
            { wch: 15 },
            { wch: 10 },
            { wch: 15 },
            { wch: 50 }
        ];
        XLSX.utils.book_append_sheet(wb, ws1, 'Tremor Features Detail');
        
        // CSV-format sheet - All features matching actual extracted data
        const headers = [
            'subject_id',
            'start_timestamp',
            'end_timestamp',
            // Magnitude statistics (12)
            'magnitude_mean',
            'magnitude_std_dev',
            'magnitude_rms',
            'magnitude_energy',
            'magnitude_max',
            'magnitude_min',
            'magnitude_range',
            'magnitude_kurtosis',
            'magnitude_skewness',
            'magnitude_cv',
            'magnitude_peaks_rt',
            'magnitude_ssc_rt',
            // Frequency features (8)
            'magnitude_fft_dom_freq',
            'magnitude_fft_tot_power',
            'magnitude_fft_energy',
            'magnitude_fft_entropy',
            'tremor_band_power_mag',
            'tremor_peak_freq',
            'dominant_freq_x',
            'tremor_band_power_x',
            // Time features (6)
            'zero_crossing_rate_mag',
            'peak_count_mag',
            'jerk_mean',
            'jerk_std',
            'stability_index',
            'magnitude_sampen',
            // Classification (3)
            'rest_tremor',
            'postural_tremor',
            'kinetic_tremor',
            // Results (3)
            'Prediction',
            'Tremor_Confidence',
            'Motion_Patterns'
        ];

        const row = [
            this.generateTestId(),
            tremorData.start_timestamp || new Date().toISOString(),
            tremorData.end_timestamp || new Date().toISOString(),
            // Magnitude statistics
            tremorData.magnitude_mean || 0,
            tremorData.magnitude_std_dev || 0,
            tremorData.magnitude_rms || 0,
            tremorData.magnitude_energy || 0,
            tremorData.magnitude_max || 0,
            tremorData.magnitude_min || 0,
            tremorData.magnitude_range || 0,
            tremorData.magnitude_kurtosis || 0,
            tremorData.magnitude_skewness || 0,
            tremorData.magnitude_cv || 0,
            tremorData.magnitude_peaks_rt || 0,
            tremorData.magnitude_ssc_rt || 0,
            // Frequency features
            tremorData.magnitude_fft_dom_freq || 0,
            tremorData.magnitude_fft_tot_power || 0,
            tremorData.magnitude_fft_energy || 0,
            tremorData.magnitude_fft_entropy || 0,
            tremorData.tremor_band_power_mag || 0,
            tremorData.tremor_peak_freq || 0,
            tremorData.dominant_freq_x || 0,
            tremorData.tremor_band_power_x || 0,
            // Time features
            tremorData.zero_crossing_rate_mag || 0,
            tremorData.peak_count_mag || 0,
            tremorData.jerk_mean || 0,
            tremorData.jerk_std || 0,
            tremorData.stability_index || 0,
            tremorData.magnitude_sampen || 0,
            // Classification
            tremorData.rest_tremor || 0,
            tremorData.postural_tremor || 0,
            tremorData.kinetic_tremor || 0,
            // Results
            results.prediction || 'N/A',
            `${(results.tremor_confidence || 0).toFixed(2)}%`,
            `${(results.motion_patterns || results.tremor_confidence || 0).toFixed(2)}%`
        ];

        const data = [headers, row];
        const ws2 = XLSX.utils.aoa_to_sheet(data);
        
        // Set column widths
        ws2['!cols'] = Array(headers.length).fill({ wch: 18 });

        XLSX.utils.book_append_sheet(wb, ws2, 'Tremor CSV Format');
    }

    /**
     * Add raw motion data sheet
     */
    addRawMotionDataSheet(wb, motionSamples) {
        const headers = [
            'Timestamp',
            'Acceleration X (m/s²)',
            'Acceleration Y (m/s²)',
            'Acceleration Z (m/s²)',
            'Magnitude (m/s²)',
            'Rotation Alpha (°/s)',
            'Rotation Beta (°/s)',
            'Rotation Gamma (°/s)',
            'Interval (ms)'
        ];

        const rows = motionSamples.map(sample => [
            sample.timestamp || 0,
            sample.accelerationX || 0,
            sample.accelerationY || 0,
            sample.accelerationZ || 0,
            Math.sqrt(
                Math.pow(sample.accelerationX || 0, 2) +
                Math.pow(sample.accelerationY || 0, 2) +
                Math.pow(sample.accelerationZ || 0, 2)
            ),
            sample.rotationAlpha || 0,
            sample.rotationBeta || 0,
            sample.rotationGamma || 0,
            sample.interval || 0
        ]);

        const data = [headers, ...rows];
        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = Array(headers.length).fill({ wch: 20 });

        XLSX.utils.book_append_sheet(wb, ws, 'Raw Motion Data');
    }

    /**
     * Generate unique test ID
     */
    generateTestId() {
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 10000);
        return `TEST_${timestamp}_${random}`;
    }

    /**
     * Get formatted timestamp for filenames
     */
    getFilenameTimestamp() {
        const now = new Date();
        return now.toISOString()
            .replace(/:/g, '-')
            .replace(/\..+/, '')
            .replace('T', '_');
    }
}

// Export for use in app.js
window.ExcelExporter = ExcelExporter;
