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

        // Raw Motion Data Sheet REMOVED - not needed in export

        XLSX.writeFile(wb, `Parkinson_Detailed_${this.getFilenameTimestamp()}.xlsx`);
        console.log('✅ Detailed data exported');
    }

    /**
     * Add summary sheet (filtered based on test mode)
     */
    addSummarySheet(wb, results, testMode) {
        const data = [
            ['Parkinson\'s Detection - Detailed Analysis'],
            [''],
            ['Test ID', this.generateTestId()],
            ['Timestamp', new Date().toISOString()],
            ['Test Mode', testMode.toUpperCase()],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A'],
            ['Overall Confidence', `${(results.confidence || 0).toFixed(2)}%`]
        ];

        // Add voice-specific results only if voice or both
        if (testMode === 'voice' || testMode === 'both') {
            data.push(
                ['Voice Patterns', `${(results.voice_patterns || results.voice_confidence || 0).toFixed(2)}%`],
                ['Voice Confidence', `${(results.voice_confidence || 0).toFixed(2)}%`]
            );
        }

        // Add tremor-specific results only if tremor or both
        if (testMode === 'tremor' || testMode === 'both') {
            data.push(
                ['Motion Patterns', `${(results.motion_patterns || results.tremor_confidence || 0).toFixed(2)}%`],
                ['Tremor Confidence', `${(results.tremor_confidence || 0).toFixed(2)}%`]
            );
        }

        data.push([''], ['METADATA'], ['Processing Time (s)', results.metadata?.processing_time || 0]);

        // Add voice metadata only if voice or both
        if (testMode === 'voice' || testMode === 'both') {
            data.push(['Audio Duration (s)', results.metadata?.audio_duration || 0]);
        }

        // Add tremor metadata only if tremor or both
        if (testMode === 'tremor' || testMode === 'both') {
            data.push(
                ['Motion Samples Count', results.metadata?.motion_samples || 0],
                ['Sample Rate (Hz)', results.metadata?.sampling_rate || 100]
            );
        }

        data.push(['Model Version', results.metadata?.model_version || 'N/A']);

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 30 }, { wch: 30 }];
        
        // Apply left alignment to all cells
        const range = XLSX.utils.decode_range(ws['!ref']);
        for (let R = range.s.r; R <= range.e.r; ++R) {
            for (let C = range.s.c; C <= range.e.c; ++C) {
                const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
                if (!ws[cellAddress]) continue;
                if (!ws[cellAddress].s) ws[cellAddress].s = {};
                ws[cellAddress].s.alignment = { horizontal: 'left' };
            }
        }
        
        XLSX.utils.book_append_sheet(wb, ws, 'Summary');
    }

    /**
     * Add voice features sheet (row-wise with linked results)
     */
    addVoiceFeaturesSheet(wb, voiceData, results) {
        // ROW-WISE FORMAT: Each category becomes a row with value
        const data = [
            ['Parkinson\'s Voice Analysis - Detailed Results'],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A'],
            ['Voice Confidence', `${(results.voice_confidence || 0).toFixed(2)}%`],
            ['Voice Patterns', `${(results.voice_patterns || results.voice_confidence || 0).toFixed(2)}%`],
            [''],
            ['AUDIO METADATA'],
            ['Audio Duration', `${(voiceData.duration || results.audio_features?.duration || results.metadata?.audio_duration || 0).toFixed(2)} seconds`],
            ['Sample Rate', `${voiceData.sample_rate || results.audio_features?.sample_rate || 'N/A'} Hz`],
            ['Test ID', this.generateTestId()],
            ['Timestamp', new Date().toISOString()],
            [''],
            ['VOICE FEATURES (matching results display)'],
            ['Voice Quality', `${(results.features?.['Voice Quality'] || 0).toFixed(2)}%`],
            ['Pitch Mean', `${(voiceData.pitch_mean || 0).toFixed(2)} Hz`],
            ['Pitch Std Deviation', `${(voiceData.pitch_std || 0).toFixed(2)} Hz`],
            ['Pitch Range', `${(voiceData.pitch_range || 0).toFixed(2)} Hz`],
            ['HNR (Harmonics)', `${(voiceData.hnr_mean || 0).toFixed(2)} dB`],
            ['Spectral Centroid', `${(voiceData.spectral_centroid || 0).toFixed(2)} Hz`],
            ['Spectral Rolloff', `${(voiceData.spectral_rolloff || 0).toFixed(2)} Hz`],
            ['Speech Rate', `${(voiceData.speech_rate || 0).toFixed(2)} rate`],
            ['Spectral Bandwidth', `${(voiceData.spectral_bandwidth || 0).toFixed(2)} Hz`],
            [''],
            ['DISCLAIMER'],
            ['This is a research tool and not a medical diagnosis.'],
            ['Please consult healthcare professionals for proper evaluation.']
        ];

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [
            { wch: 30 },  // Category/Feature name column
            { wch: 40 }   // Value column
        ];

        // Apply left alignment to all cells
        const range = XLSX.utils.decode_range(ws['!ref']);
        for (let R = range.s.r; R <= range.e.r; ++R) {
            for (let C = range.s.c; C <= range.e.c; ++C) {
                const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
                if (!ws[cellAddress]) continue;
                if (!ws[cellAddress].s) ws[cellAddress].s = {};
                ws[cellAddress].s.alignment = { horizontal: 'left' };
            }
        }

        XLSX.utils.book_append_sheet(wb, ws, 'Voice Features');
    }

    /**
     * Add tremor features sheet (row-wise format, no tremor type columns)
     */
    addTremorFeaturesSheet(wb, tremorData, results) {
        // ROW-WISE FORMAT: Each category becomes a row with value
        const data = [
            ['Parkinson\'s Tremor Analysis - Detailed Results'],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A'],
            ['Tremor Confidence', `${(results.tremor_confidence || 0).toFixed(2)}%`],
            ['Motion Patterns', `${(results.motion_patterns || results.tremor_confidence || 0).toFixed(2)}%`],
            [''],
            ['TEST METADATA'],
            ['Test ID', this.generateTestId()],
            ['Start Timestamp', tremorData.start_timestamp || new Date().toISOString()],
            ['End Timestamp', tremorData.end_timestamp || new Date().toISOString()],
            ['Motion Samples Count', results.metadata?.motion_samples || 0],
            ['Sample Rate', `${results.metadata?.sampling_rate || 100} Hz`],
            [''],
            ['KEY METRICS'],
            ['Tremor Frequency', `${(tremorData.magnitude_fft_dom_freq || 0).toFixed(2)} Hz`],
            ['Tremor Frequency Score', `${(results.features?.['Tremor Frequency'] || 0).toFixed(2)}%`],
            ['Postural Stability Score', `${(results.features?.['Postural Stability'] || 0).toFixed(2)}%`],
            ['Motion Variability Score', `${(results.features?.['Motion Variability'] || 0).toFixed(2)}%`],
            [''],
            ['ACCELERATION FEATURES'],
            ['Magnitude Mean', `${(tremorData.magnitude_mean || 0).toFixed(4)} m/s²`],
            ['Magnitude Std Dev', `${(tremorData.magnitude_std_dev || 0).toFixed(4)} m/s²`],
            ['Magnitude RMS', `${(tremorData.magnitude_rms || 0).toFixed(4)} m/s²`],
            ['Magnitude Energy', `${(tremorData.magnitude_energy || 0).toFixed(4)}`],
            ['Magnitude Max', `${(tremorData.magnitude_max || 0).toFixed(4)} m/s²`],
            ['Magnitude Min', `${(tremorData.magnitude_min || 0).toFixed(4)} m/s²`],
            ['Magnitude Range', `${(tremorData.magnitude_range || 0).toFixed(4)} m/s²`],
            ['Magnitude Kurtosis', `${(tremorData.magnitude_kurtosis || 0).toFixed(4)}`],
            ['Magnitude Skewness', `${(tremorData.magnitude_skewness || 0).toFixed(4)}`],
            ['Coefficient of Variation', `${(tremorData.magnitude_cv || 0).toFixed(4)}`],
            ['Peak Rate', `${(tremorData.magnitude_peaks_rt || 0).toFixed(4)}`],
            ['Slope Sign Changes Rate', `${(tremorData.magnitude_ssc_rt || 0).toFixed(4)}`],
            [''],
            ['FREQUENCY FEATURES'],
            ['Dominant Frequency', `${(tremorData.magnitude_fft_dom_freq || 0).toFixed(4)} Hz`],
            ['Total FFT Power', `${(tremorData.magnitude_fft_tot_power || 0).toFixed(4)}`],
            ['FFT Energy', `${(tremorData.magnitude_fft_energy || 0).toFixed(4)}`],
            ['FFT Entropy', `${(tremorData.magnitude_fft_entropy || 0).toFixed(4)}`],
            ['Tremor Band Power (4-6Hz)', `${(tremorData.tremor_band_power_mag || 0).toFixed(4)}`],
            ['Tremor Peak Frequency', `${(tremorData.tremor_peak_freq || 0).toFixed(4)} Hz`],
            ['Dominant Frequency X-axis', `${(tremorData.dominant_freq_x || 0).toFixed(4)} Hz`],
            ['Tremor Band Power X-axis', `${(tremorData.tremor_band_power_x || 0).toFixed(4)}`],
            [''],
            ['TIME DOMAIN FEATURES'],
            ['Zero Crossing Rate', `${(tremorData.zero_crossing_rate_mag || 0).toFixed(4)}`],
            ['Peak Count', `${(tremorData.peak_count_mag || 0).toFixed(0)}`],
            ['Jerk Mean', `${(tremorData.jerk_mean || 0).toFixed(4)} m/s³`],
            ['Jerk Std Dev', `${(tremorData.jerk_std || 0).toFixed(4)} m/s³`],
            [''],
            ['STABILITY METRICS'],
            ['Stability Index', `${(tremorData.stability_index || 0).toFixed(4)}`],
            ['Sample Entropy', `${(tremorData.magnitude_sampen || 0).toFixed(4)}`],
            [''],
            ['DISCLAIMER'],
            ['This is a research tool and not a medical diagnosis.'],
            ['Please consult healthcare professionals for proper evaluation.']
        ];

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [
            { wch: 35 },  // Feature name column
            { wch: 40 }   // Value column
        ];

        // Apply left alignment to all cells
        const range = XLSX.utils.decode_range(ws['!ref']);
        for (let R = range.s.r; R <= range.e.r; ++R) {
            for (let C = range.s.c; C <= range.e.c; ++C) {
                const cellAddress = XLSX.utils.encode_cell({ r: R, c: C });
                if (!ws[cellAddress]) continue;
                if (!ws[cellAddress].s) ws[cellAddress].s = {};
                ws[cellAddress].s.alignment = { horizontal: 'left' };
            }
        }

        XLSX.utils.book_append_sheet(wb, ws, 'Tremor Features');
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
