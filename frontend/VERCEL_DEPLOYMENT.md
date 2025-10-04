# Vercel Deployment Guide

## Overview
This guide explains how to deploy the Parkinson Detection frontend to Vercel while keeping the backend running locally via ngrok.

## Architecture
- **Frontend**: Deployed on Vercel (static hosting)
- **Backend**: Running locally via `run.ps1` on your machine
- **Connection**: Frontend connects to backend through ngrok domain

## Prerequisites
1. Vercel account (free tier works)
2. Backend running locally with ngrok domain: `freezingly-nonsignificative-edison.ngrok-free.dev`
3. Git repository with the frontend code

## Deployment Steps

### Step 1: Prepare the Frontend for Vercel
The frontend is already configured with:
- ✅ `vercel.json` - Vercel configuration
- ✅ `js/config.js` - Environment-aware backend URL handling
- ✅ `.env.example` - Environment variable template

### Step 2: Deploy to Vercel

1. **Import your repository to Vercel:**
   - Go to https://vercel.com/new
   - Click "Import Git Repository"
   - Select your `parkinson-detection` repository
   - Choose the `frontend` directory as the root

2. **Configure the project:**
   - Framework Preset: `Other` (static site)
   - Root Directory: `frontend`
   - Build Command: (leave empty for static deployment)
   - Output Directory: `.` (current directory)

3. **Add Environment Variable:**
   - In Vercel dashboard, go to your project settings
   - Navigate to: **Settings** → **Environment Variables**
   - Add a new variable:
     - **Name**: `BACKEND_URL`
     - **Value**: `https://freezingly-nonsignificative-edison.ngrok-free.dev`
     - **Environment**: Select all (Production, Preview, Development)
   - Click "Save"

4. **Deploy:**
   - Click "Deploy"
   - Wait for deployment to complete
   - You'll get a Vercel URL like: `https://your-project.vercel.app`

### Step 3: Update Backend CORS Settings (if needed)

If you encounter CORS errors, update your Flask backend's CORS configuration to allow requests from your Vercel domain:

```python
# In backend/app.py
CORS(app, origins=[
    "https://your-project.vercel.app",
    "http://localhost:8000",
    # ... other origins
])
```

## Local Development Workflow

### Running Locally (Unchanged)
The local workflow remains exactly the same:

```powershell
# Run the project locally
.\run.ps1
```

This will:
1. Start the Flask backend on port 5000
2. Start the frontend proxy server on port 8000
3. Start ngrok tunnel pointing to port 8000
4. The frontend automatically detects local environment and uses the proxy

**Important**: Local development uses the proxy server (`server.py`) which routes `/api/*` requests to `localhost:5000`. This continues to work as before.

## How It Works

### Local Development (run.ps1)
```
Browser → http://localhost:8000 
       → server.py (proxy) 
       → /api/* → localhost:5000 (Flask backend)
```

### Production (Vercel)
```
Browser → https://your-project.vercel.app 
       → Frontend (Vercel CDN) 
       → Direct API calls → https://freezingly-nonsignificative-edison.ngrok-free.dev (Flask backend)
```

### Configuration Logic
The `js/config.js` file automatically detects the environment:

- **Local**: Uses `/api` (proxied by `server.py`)
- **Vercel**: Uses `BACKEND_URL` environment variable

## Testing the Deployment

1. **Start your local backend:**
   ```powershell
   .\run.ps1
   ```

2. **Ensure ngrok domain is active:**
   - Verify `freezingly-nonsignificative-edison.ngrok-free.dev` is accessible
   - Test: `https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health`

3. **Visit your Vercel URL:**
   - Open `https://your-project.vercel.app`
   - The app should load and connect to your local backend via ngrok

4. **Check browser console:**
   - Should see: `🚀 Production mode: Using configured backend URL`
   - Should see: `✅ Backend is available and ready`

## Troubleshooting

### Issue: "Backend not available"
- **Check**: Is `run.ps1` running?
- **Check**: Is the ngrok tunnel active?
- **Check**: Can you access `https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health` in your browser?

### Issue: CORS errors
- **Solution**: Update Flask CORS settings to include your Vercel domain
- **Location**: `backend/app.py` - CORS configuration

### Issue: Environment variable not working
- **Check**: Variable name is exactly `BACKEND_URL` (case-sensitive)
- **Check**: Value has no trailing slash: `https://domain.ngrok-free.dev` ✅ not `https://domain.ngrok-free.dev/` ❌
- **Solution**: Redeploy after changing environment variables

### Issue: Frontend shows old config
- **Solution**: Clear browser cache or open in incognito/private window
- **Solution**: Redeploy from Vercel dashboard

## Updating the ngrok Domain

If your ngrok domain changes:

1. **Update Vercel environment variable:**
   - Go to Vercel project settings
   - Edit `BACKEND_URL` with new domain
   - Redeploy (or wait for automatic deployment)

2. **Local development**: No changes needed - it uses the proxy

## Environment Variables Reference

| Variable | Required | Value | Example |
|----------|----------|-------|---------|
| `BACKEND_URL` | Yes (Vercel only) | Your ngrok domain | `https://freezingly-nonsignificative-edison.ngrok-free.dev` |

## Security Notes

- The ngrok domain is publicly accessible - ensure your backend has appropriate authentication if needed
- Environment variables in Vercel are secure and not exposed to the client
- CORS should be configured to only allow your Vercel domain in production

## Support

For issues:
1. Check the browser console for error messages
2. Verify backend is running: `run.ps1`
3. Test backend directly: `https://your-ngrok-domain.ngrok-free.dev/api/health`
4. Check Vercel deployment logs in the dashboard
