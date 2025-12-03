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
     * Helper: Check if value is valid (not 0, null, undefined, NaN, or 'N/A')
     */
    isValidValue(value) {
        if (value === null || value === undefined || value === 'N/A') return false;
        if (typeof value === 'number' && (isNaN(value) || value === 0)) return false;
        if (typeof value === 'string' && value.trim() === '') return false;
        return true;
    }

    /**
     * Helper: Format feature name for display
     */
    formatFeatureName(key) {
        return key
            .replace(/_/g, ' ')
            .replace(/mfcc/gi, 'MFCC')
            .replace(/rms/gi, 'RMS')
            .replace(/zcr/gi, 'ZCR')
            .replace(/hnr/gi, 'HNR')
            .replace(/fft/gi, 'FFT')
            .replace(/std/gi, 'Std Dev')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Helper: Add row only if value is valid
     */
    addRowIfValid(data, label, value, unit = '') {
        if (this.isValidValue(value)) {
            const formattedValue = typeof value === 'number' ? 
                (Math.abs(value) < 0.01 ? value.toExponential(2) : value.toFixed(4)) : value;
            data.push([label, unit ? `${formattedValue} ${unit}` : formattedValue]);
        }
    }

    /**
     * Generate simple report (PDF-style formatted Excel)
     */
    async exportSimpleReport(results, testMode) {
        await this.loadSheetJS();

        const data = [
            ['Parkinson\'s Detection - Analysis Report'],
            [''],
            ['TEST INFORMATION'],
            ['Date & Time', new Date().toLocaleString()],
            ['Test Mode', testMode.toUpperCase()],
            ['Test ID', this.generateTestId()],
            [''],
            ['RESULTS SUMMARY'],
            ['Prediction', results.prediction || 'N/A']
        ];

        // Add confidence only if valid
        this.addRowIfValid(data, 'Overall Confidence', results.confidence, '%');
        this.addRowIfValid(data, 'Risk Score', results.risk_score, '%');
        data.push(['']);

        // Voice Analysis Section
        if (testMode === 'voice' || testMode === 'both') {
            const hasVoiceData = this.isValidValue(results.voice_patterns) || 
                                 this.isValidValue(results.voice_confidence) ||
                                 (results.audio_features && Object.keys(results.audio_features).length > 0);
            
            if (hasVoiceData) {
                data.push(['VOICE ANALYSIS']);
                this.addRowIfValid(data, 'Voice Patterns', results.voice_patterns, '%');
                this.addRowIfValid(data, 'Voice Confidence', results.voice_confidence, '%');
                
                // Add ALL voice features from audio_features
                if (results.audio_features) {
                    data.push([''], ['Voice Features']);
                    const audioFeatures = results.audio_features;
                    
                    // Sort keys and add all valid features
                    Object.keys(audioFeatures).sort().forEach(key => {
                        const value = audioFeatures[key];
                        if (this.isValidValue(value)) {
                            const unit = key.includes('freq') || key.includes('pitch') || key.includes('centroid') ? 'Hz' :
                                        key.includes('energy') || key.includes('power') ? '' :
                                        key.includes('hnr') ? 'dB' : '';
                            this.addRowIfValid(data, this.formatFeatureName(key), value, unit);
                        }
                    });
                }
                data.push(['']);
            }
        }

        // Tremor Analysis Section
        if (testMode === 'tremor' || testMode === 'both') {
            const hasTremorData = this.isValidValue(results.motion_patterns) || 
                                  this.isValidValue(results.tremor_confidence) ||
                                  (results.tremor_features && Object.keys(results.tremor_features).length > 0);
            
            if (hasTremorData) {
                data.push(['TREMOR ANALYSIS']);
                this.addRowIfValid(data, 'Motion Patterns', results.motion_patterns, '%');
                this.addRowIfValid(data, 'Tremor Confidence', results.tremor_confidence, '%');
                
                // Add ALL tremor features
                if (results.tremor_features) {
                    data.push([''], ['Tremor Features']);
                    const tremorFeatures = results.tremor_features;
                    
                    // Sort keys and add all valid features
                    Object.keys(tremorFeatures).sort().forEach(key => {
                        if (key.startsWith('_')) return; // Skip internal keys
                        const value = tremorFeatures[key];
                        if (this.isValidValue(value)) {
                            const unit = key.includes('freq') ? 'Hz' :
                                        key.includes('power') || key.includes('energy') ? '' :
                                        key.includes('jerk') ? 'm/s³' :
                                        key.includes('magnitude') && !key.includes('fft') ? 'm/s²' : '';
                            this.addRowIfValid(data, this.formatFeatureName(key), value, unit);
                        }
                    });
                }
                data.push(['']);
            }
        }

        // Metadata
        data.push(['PROCESSING DETAILS']);
        this.addRowIfValid(data, 'Processing Time', results.metadata?.processing_time, 'seconds');
        this.addRowIfValid(data, 'Audio Features Count', results.metadata?.audio_features_count);
        this.addRowIfValid(data, 'Tremor Features Count', results.metadata?.tremor_features_count);
        this.addRowIfValid(data, 'Motion Samples', results.metadata?.motion_samples);
        data.push(['Model Version', results.metadata?.model_version || '1.0.0']);
        
        data.push(
            [''],
            ['DISCLAIMER'],
            ['This is a research tool and not a medical diagnosis.'],
            ['Please consult healthcare professionals for proper evaluation.']
        );

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 35 }, { wch: 45 }];

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

        // Summary Sheet (always included)
        this.addSummarySheet(wb, results, testMode);

        // Voice Features Sheet (if applicable)
        if (testMode === 'voice' || testMode === 'both') {
            this.addVoiceFeaturesSheet(wb, results);
        }

        // Tremor Features Sheet (if applicable)
        if (testMode === 'tremor' || testMode === 'both') {
            this.addTremorFeaturesSheet(wb, results);
        }

        // All Features Sheet (complete raw data)
        this.addAllFeaturesSheet(wb, results, testMode);

        XLSX.writeFile(wb, `Parkinson_Detailed_${this.getFilenameTimestamp()}.xlsx`);
        console.log('✅ Detailed data exported');
    }

    /**
     * Add summary sheet (filtered based on test mode)
     */
    addSummarySheet(wb, results, testMode) {
        const data = [
            ['Parkinson\'s Detection - Detailed Analysis Summary'],
            [''],
            ['TEST INFORMATION'],
            ['Test ID', this.generateTestId()],
            ['Timestamp', new Date().toISOString()],
            ['Test Mode', testMode.toUpperCase()],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A']
        ];

        this.addRowIfValid(data, 'Overall Confidence', results.confidence, '%');
        this.addRowIfValid(data, 'Risk Score', results.risk_score, '%');

        // Voice results
        if (testMode === 'voice' || testMode === 'both') {
            this.addRowIfValid(data, 'Voice Patterns', results.voice_patterns, '%');
            this.addRowIfValid(data, 'Voice Confidence', results.voice_confidence, '%');
        }

        // Tremor results
        if (testMode === 'tremor' || testMode === 'both') {
            this.addRowIfValid(data, 'Motion Patterns', results.motion_patterns, '%');
            this.addRowIfValid(data, 'Tremor Confidence', results.tremor_confidence, '%');
        }

        data.push([''], ['METADATA']);
        this.addRowIfValid(data, 'Processing Time', results.metadata?.processing_time, 'seconds');
        this.addRowIfValid(data, 'Audio Features Count', results.metadata?.audio_features_count);
        this.addRowIfValid(data, 'Tremor Features Count', results.metadata?.tremor_features_count);
        this.addRowIfValid(data, 'Motion Samples', results.metadata?.motion_samples);
        data.push(['Model Version', results.metadata?.model_version || '1.0.0']);

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 30 }, { wch: 35 }];
        XLSX.utils.book_append_sheet(wb, ws, 'Summary');
    }

    /**
     * Add voice features sheet with ALL audio features
     */
    addVoiceFeaturesSheet(wb, results) {
        const data = [
            ['Voice Analysis - All Features'],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A']
        ];

        this.addRowIfValid(data, 'Voice Confidence', results.voice_confidence, '%');
        this.addRowIfValid(data, 'Voice Patterns', results.voice_patterns, '%');

        // Add ALL audio features
        const audioFeatures = results.audio_features || {};
        const featureCount = Object.keys(audioFeatures).filter(k => this.isValidValue(audioFeatures[k])).length;

        if (featureCount > 0) {
            data.push([''], [`VOICE FEATURES (${featureCount} total)`]);
            
            // Group features by category
            const categories = {
                'Pitch Features': [],
                'MFCC Features': [],
                'Spectral Features': [],
                'Energy Features': [],
                'Other Features': []
            };

            Object.keys(audioFeatures).sort().forEach(key => {
                const value = audioFeatures[key];
                if (!this.isValidValue(value)) return;

                const unit = key.includes('freq') || key.includes('pitch') || key.includes('centroid') || key.includes('rolloff') || key.includes('bandwidth') ? 'Hz' :
                            key.includes('hnr') ? 'dB' : '';
                const row = [this.formatFeatureName(key), unit ? `${this.formatNumber(value)} ${unit}` : this.formatNumber(value)];

                if (key.includes('pitch')) categories['Pitch Features'].push(row);
                else if (key.includes('mfcc')) categories['MFCC Features'].push(row);
                else if (key.includes('spectral') || key.includes('centroid') || key.includes('bandwidth') || key.includes('rolloff') || key.includes('contrast')) categories['Spectral Features'].push(row);
                else if (key.includes('energy') || key.includes('rms') || key.includes('power')) categories['Energy Features'].push(row);
                else categories['Other Features'].push(row);
            });

            // Add each category
            Object.keys(categories).forEach(category => {
                if (categories[category].length > 0) {
                    data.push([''], [category]);
                    categories[category].forEach(row => data.push(row));
                }
            });
        }

        data.push(
            [''],
            ['DISCLAIMER'],
            ['This is a research tool and not a medical diagnosis.']
        );

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 35 }, { wch: 40 }];
        XLSX.utils.book_append_sheet(wb, ws, 'Voice Features');
    }

    /**
     * Add tremor features sheet with ALL tremor features
     */
    addTremorFeaturesSheet(wb, results) {
        const data = [
            ['Tremor Analysis - All Features'],
            [''],
            ['ANALYSIS RESULTS'],
            ['Prediction', results.prediction || 'N/A']
        ];

        this.addRowIfValid(data, 'Tremor Confidence', results.tremor_confidence, '%');
        this.addRowIfValid(data, 'Motion Patterns', results.motion_patterns, '%');

        // Add ALL tremor features
        const tremorFeatures = results.tremor_features || {};
        const featureCount = Object.keys(tremorFeatures).filter(k => !k.startsWith('_') && this.isValidValue(tremorFeatures[k])).length;

        if (featureCount > 0) {
            data.push([''], [`TREMOR FEATURES (${featureCount} total)`]);
            
            // Group features by category
            const categories = {
                'Magnitude Statistics': [],
                'Frequency Domain': [],
                'Time Domain': [],
                'Stability Metrics': [],
                'Axis-Specific Features': [],
                'Other Features': []
            };

            Object.keys(tremorFeatures).sort().forEach(key => {
                if (key.startsWith('_')) return;
                const value = tremorFeatures[key];
                if (!this.isValidValue(value)) return;

                const unit = key.includes('freq') ? 'Hz' :
                            key.includes('jerk') ? 'm/s³' :
                            (key.includes('magnitude') && !key.includes('fft') && !key.includes('power')) ? 'm/s²' : '';
                const row = [this.formatFeatureName(key), unit ? `${this.formatNumber(value)} ${unit}` : this.formatNumber(value)];

                if (key.includes('magnitude') && !key.includes('fft')) categories['Magnitude Statistics'].push(row);
                else if (key.includes('fft') || key.includes('freq') || key.includes('power') || key.includes('band')) categories['Frequency Domain'].push(row);
                else if (key.includes('zcr') || key.includes('peak') || key.includes('jerk') || key.includes('crossing')) categories['Time Domain'].push(row);
                else if (key.includes('stability') || key.includes('entropy') || key.includes('sampen') || key.includes('dfa')) categories['Stability Metrics'].push(row);
                else if (key.includes('_x') || key.includes('_y') || key.includes('_z')) categories['Axis-Specific Features'].push(row);
                else categories['Other Features'].push(row);
            });

            // Add each category
            Object.keys(categories).forEach(category => {
                if (categories[category].length > 0) {
                    data.push([''], [category]);
                    categories[category].forEach(row => data.push(row));
                }
            });
        }

        data.push(
            [''],
            ['DISCLAIMER'],
            ['This is a research tool and not a medical diagnosis.']
        );

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 35 }, { wch: 40 }];
        XLSX.utils.book_append_sheet(wb, ws, 'Tremor Features');
    }

    /**
     * Add all features sheet (complete raw data in one place)
     */
    addAllFeaturesSheet(wb, results, testMode) {
        const data = [
            ['Complete Feature Export - All Raw Data'],
            [''],
            ['Test Mode', testMode.toUpperCase()],
            ['Prediction', results.prediction || 'N/A'],
            ['Timestamp', new Date().toISOString()],
            ['']
        ];

        // Voice features
        if (testMode === 'voice' || testMode === 'both') {
            const audioFeatures = results.audio_features || {};
            const validFeatures = Object.keys(audioFeatures).filter(k => this.isValidValue(audioFeatures[k]));
            
            if (validFeatures.length > 0) {
                data.push([`VOICE FEATURES (${validFeatures.length})`]);
                validFeatures.sort().forEach(key => {
                    data.push([key, this.formatNumber(audioFeatures[key])]);
                });
                data.push(['']);
            }
        }

        // Tremor features
        if (testMode === 'tremor' || testMode === 'both') {
            const tremorFeatures = results.tremor_features || {};
            const validFeatures = Object.keys(tremorFeatures).filter(k => !k.startsWith('_') && this.isValidValue(tremorFeatures[k]));
            
            if (validFeatures.length > 0) {
                data.push([`TREMOR FEATURES (${validFeatures.length})`]);
                validFeatures.sort().forEach(key => {
                    data.push([key, this.formatNumber(tremorFeatures[key])]);
                });
                data.push(['']);
            }
        }

        // Key features (normalized 0-100)
        if (results.features && Object.keys(results.features).length > 0) {
            const keyFeatures = results.features;
            const validFeatures = Object.keys(keyFeatures).filter(k => this.isValidValue(keyFeatures[k]));
            
            if (validFeatures.length > 0) {
                data.push(['KEY FEATURES (Normalized 0-100)']);
                validFeatures.sort().forEach(key => {
                    data.push([key, `${this.formatNumber(keyFeatures[key])}%`]);
                });
            }
        }

        const ws = XLSX.utils.aoa_to_sheet(data);
        ws['!cols'] = [{ wch: 40 }, { wch: 30 }];
        XLSX.utils.book_append_sheet(wb, ws, 'All Features');
    }

    /**
     * Format number for display
     */
    formatNumber(value) {
        if (typeof value !== 'number') return value;
        if (Math.abs(value) < 0.0001 && value !== 0) return value.toExponential(2);
        if (Math.abs(value) >= 1000) return value.toFixed(2);
        if (Number.isInteger(value)) return value.toString();
        return value.toFixed(4);
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
