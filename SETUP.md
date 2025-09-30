# Setup Instructions for Parkinson's Detection PWA

## Quick Start

### Frontend Setup (Required)
The frontend can run standalone with demo mode if the backend is not available.

1. **Open the frontend directory in VS Code or your preferred editor**
2. **Serve the files using a local web server** (required for PWA features):

   **Option A: Using Python (if installed):**
   ```bash
   cd frontend
   python -m http.server 8080
   ```

   **Option B: Using Node.js (if installed):**
   ```bash
   cd frontend
   npx serve -s . -l 8080
   ```

   **Option C: Using VS Code Live Server extension:**
   - Install the "Live Server" extension
   - Right-click on `index.html` and select "Open with Live Server"

3. **Access the app:**
   - Open `http://localhost:8080` in your browser
   - **Important:** Use HTTPS for mobile testing (Chrome DevTools can simulate mobile)

### Backend Setup (Optional - for full ML analysis)

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```

3. **The backend will be available at:** `http://localhost:5000`

## Testing the Application

### Desktop Testing
1. Open the app in Chrome/Edge
2. Test microphone recording
3. Test motion simulation (limited on desktop)
4. Check PWA install prompt

### Mobile Testing (Recommended)
1. **Enable HTTPS for mobile access:**
   - Use ngrok: `ngrok http 8080`
   - Or deploy to GitHub Pages/Netlify
   
2. **Access on mobile device:**
   - Open the HTTPS URL on your smartphone
   - Grant microphone and motion permissions
   - Test full functionality including motion sensors

### PWA Installation
1. **On Chrome (Android/Desktop):**
   - Look for "Install" button in address bar
   - Or use the in-app install banner

2. **On Safari (iOS):**
   - Tap "Share" → "Add to Home Screen"

## Features to Test

### ✅ Voice Recording
- Microphone access permission
- 10-second voice recording
- Audio visualization
- Recording controls

### ✅ Motion Detection
- Device motion permission (iOS 13+)
- 15-second tremor test
- Motion data collection
- Real-time feedback

### ✅ Analysis & Results
- Processing animation
- Results display with confidence scores
- Feature breakdown
- Share functionality

### ✅ PWA Features
- App installation
- Offline functionality
- Service worker caching
- Responsive design

## Demo Mode
- App automatically falls back to demo mode if backend is unavailable
- Generates realistic mock results
- Shows demo notification
- Full frontend functionality maintained

## Troubleshooting

### Microphone Issues
- **Permission denied:** Check browser permissions in settings
- **No audio:** Ensure microphone is not muted
- **Poor quality:** Test in quiet environment

### Motion Sensor Issues
- **No motion data:** Only works on mobile devices
- **Permission denied (iOS):** Must be on HTTPS
- **Limited data:** Hold device steady during test

### PWA Installation Issues
- **No install prompt:** Ensure HTTPS and valid manifest
- **Icons missing:** App works without icons but may affect install prompt
- **Service worker errors:** Check browser console for details

### Backend Connection Issues
- **API errors:** Check if backend server is running
- **CORS errors:** Backend includes CORS headers
- **Demo mode:** App automatically falls back to demo results

## Development Notes

### File Structure
```
parkinson/
├── frontend/              # PWA Frontend
│   ├── index.html        # Main HTML
│   ├── styles.css        # Styling
│   ├── app.js           # Main logic
│   ├── manifest.json    # PWA manifest
│   ├── sw.js           # Service worker
│   └── assets/         # Icons (add your own)
└── backend/             # Python Backend
    ├── app.py          # Flask server
    ├── ml_models.py    # ML pipeline
    └── requirements.txt # Dependencies
```

### Browser Compatibility
- **Chrome/Edge:** Full support
- **Firefox:** Good support (limited PWA features)
- **Safari:** Good support (requires user gesture for permissions)
- **Mobile browsers:** Recommended for full experience

### Security Considerations
- **HTTPS required** for microphone, motion sensors, and PWA features
- **Demo mode** provides safe fallback for testing
- **No sensitive data storage** in current implementation

## Next Steps for Production

1. **Add proper icons** (see assets/README_ICONS.md)
2. **Deploy to HTTPS hosting** (Netlify, Vercel, GitHub Pages)
3. **Train ML models** with real medical data
4. **Add user authentication** if needed
5. **Implement data privacy measures**
6. **Add medical disclaimers** and professional review

## Support

For issues or questions:
1. Check browser console for errors
2. Verify permissions are granted
3. Test on HTTPS (required for mobile)
4. Try demo mode first (frontend only)

Remember: This is a research prototype and should not be used for medical diagnosis.