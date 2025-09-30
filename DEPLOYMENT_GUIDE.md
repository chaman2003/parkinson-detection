# Vercel Deployment Guide

This guide explains how to deploy both the frontend and backend of the Parkinson's Detection app to Vercel manually through the website.

## Overview

The project is structured to support two separate Vercel deployments:
- **Frontend**: Main PWA application (deployed from root directory)
- **Backend**: Simplified API server (deployed from backend directory)

## Prerequisites

1. GitHub repository with your code
2. Vercel account (free tier is sufficient)
3. Access to the repository at: https://github.com/chaman2003/parkinson-detection

## Deployment Steps

### 1. Deploy Backend API

1. **Go to Vercel Dashboard**
   - Visit https://vercel.com/dashboard
   - Click "Add New..." → "Project"

2. **Import Repository**
   - Search for "parkinson-detection" repository
   - Click "Import"

3. **Configure Backend Deployment**
   - **Project Name**: `parkinson-detection-backend`
   - **Framework Preset**: Other
   - **Root Directory**: `backend`
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements-vercel.txt`

4. **Environment Variables** (if needed)
   - Click "Environment Variables"
   - Add any environment variables (none required for current setup)

5. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Note the deployment URL (e.g., `https://parkinson-detection-backend.vercel.app`)

### 2. Deploy Frontend PWA

1. **Create New Project**
   - In Vercel dashboard, click "Add New..." → "Project"
   - Import the same repository again

2. **Configure Frontend Deployment**
   - **Project Name**: `parkinson-detection-frontend` (or just `parkinson-detection`)
   - **Framework Preset**: Other
   - **Root Directory**: `.` (root directory)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - **Install Command**: Leave empty

3. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Note the deployment URL (e.g., `https://parkinson-detection.vercel.app`)

### 3. Update API Configuration (Important!)

After deploying the backend, you need to update the frontend to use the correct backend URL:

1. **Get Backend URL**
   - From your backend Vercel deployment, copy the production URL
   - Example: `https://parkinson-detection-backend.vercel.app`

2. **Update Frontend Code**
   - Edit `app.js` in your repository
   - Find the `getApiBaseUrl()` method
   - Update the Vercel condition to use your actual backend URL:

```javascript
// Replace this line in getApiBaseUrl method:
if (hostname.includes('vercel.app')) {
    return `${protocol}//${hostname}/api`;
}

// With your actual backend URL:
if (hostname.includes('vercel.app')) {
    return 'https://parkinson-detection-backend.vercel.app/api';
}
```

3. **Redeploy Frontend**
   - Commit and push changes to GitHub
   - Vercel will automatically redeploy your frontend

## File Structure for Deployment

```
parkinson-detection/
├── vercel.json                 # Frontend Vercel config
├── index.html                  # Main PWA entry point
├── app.js                      # Frontend application
├── styles.css                  # PWA styles
├── sw.js                      # Service worker
├── manifest.json              # PWA manifest
└── backend/
    ├── vercel.json            # Backend Vercel config
    ├── api.py                 # Simplified backend API
    ├── requirements-vercel.txt # Minimal dependencies
    ├── app.py                 # Full backend (for local dev)
    └── requirements.txt       # Full dependencies (for local dev)
```

## Deployment Features

### Backend (Serverless Functions)
- **Simplified ML**: Uses statistical analysis instead of heavy ML libraries
- **Lightweight**: Only Flask and Flask-CORS dependencies
- **Fast**: Quick cold start times
- **Realistic Results**: Generates meaningful results based on input data patterns

### Frontend (Static PWA)
- **Progressive Web App**: Installable, offline-capable
- **Responsive Design**: Works on mobile and desktop
- **Environment Aware**: Automatically detects deployment environment
- **Fallback Support**: Demo mode when backend is unavailable

## Testing Your Deployment

1. **Test Backend**
   - Visit: `https://your-backend-url.vercel.app/api/health`
   - Should return JSON with status information

2. **Test Frontend**
   - Visit: `https://your-frontend-url.vercel.app`
   - Try the voice and motion tests
   - Check if it connects to your backend

3. **Test Integration**
   - Complete a full test in your deployed frontend
   - Verify results are generated correctly

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Backend includes Flask-CORS
   - Should work automatically

2. **API Connection Issues**
   - Check that frontend is using correct backend URL
   - Verify backend deployment is successful

3. **Function Timeout**
   - Vercel free tier has 10-second timeout
   - Current backend is optimized for quick responses

4. **Import Errors in Backend**
   - Backend uses minimal dependencies (Flask, Flask-CORS only)
   - Heavy ML libraries are excluded

### Vercel Limits (Free Tier)
- **Bandwidth**: 100GB/month
- **Function Execution**: 100GB-hrs/month
- **Function Duration**: 10 seconds max
- **Function Size**: 50MB max

## Domain Configuration (Optional)

1. **Custom Domain**
   - In Vercel project settings
   - Go to "Domains"
   - Add your custom domain
   - Configure DNS records as instructed

2. **Environment Variables**
   - Set production environment variables
   - Configure API URLs if needed

## Monitoring

- View deployment logs in Vercel dashboard
- Monitor function usage and performance
- Set up notifications for deployment failures

## Local Development

To run locally while maintaining Vercel compatibility:

```bash
# Backend (full version with ML)
cd backend
python app.py

# Backend (Vercel-compatible version)
cd backend
python api.py

# Frontend
# Serve static files (use any HTTP server)
python -m http.server 8000
```

## Security Notes

- No sensitive data in environment variables
- CORS properly configured
- Input validation implemented
- File upload size limits enforced

---

**Note**: After deployment, remember to update the README.md with your live demo URLs!