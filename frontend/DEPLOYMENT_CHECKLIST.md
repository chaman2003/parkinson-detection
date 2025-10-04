# ✅ Deployment Checklist

Use this checklist to ensure everything is ready for Vercel deployment.

---

## 📋 Pre-Deployment Checklist

### Local Setup Verification

- [ ] **Backend is working**
  ```powershell
  cd backend
  python app.py
  ```
  Expected: Backend starts on port 5000

- [ ] **Frontend proxy is working**
  ```powershell
  cd frontend
  python server.py 8000
  ```
  Expected: Frontend accessible at http://localhost:8000

- [ ] **run.ps1 script works**
  ```powershell
  .\run.ps1
  ```
  Expected: All services start, ngrok tunnel created

- [ ] **ngrok domain is active**
  - Check terminal output for ngrok URL
  - Verify: `https://freezingly-nonsignificative-edison.ngrok-free.dev`
  - Test: Visit `https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health`

### File Verification

- [ ] **Configuration files exist**
  - `frontend/vercel.json` ✓
  - `frontend/package.json` ✓
  - `frontend/vercel-build.js` ✓
  - `frontend/js/config.js` ✓
  - `frontend/.gitignore` ✓

- [ ] **Documentation exists**
  - `frontend/QUICKSTART_VERCEL.md` ✓
  - `frontend/VERCEL_DEPLOYMENT.md` ✓
  - `frontend/.env.example` ✓

- [ ] **Code is updated**
  - `frontend/index.html` - Loads config.js ✓
  - `frontend/js/app.js` - Uses AppConfig ✓

### Git Repository

- [ ] **Changes are committed**
  ```bash
  git status
  # Should show clean working tree or ready to commit
  ```

- [ ] **All files are added**
  ```bash
  git add frontend/
  git commit -m "Add Vercel deployment configuration"
  ```

- [ ] **Pushed to remote**
  ```bash
  git push origin main
  ```

---

## 🚀 Vercel Deployment Checklist

### Step 1: Create Vercel Project

- [ ] **Go to Vercel**
  - Visit: https://vercel.com/new
  - Sign in with GitHub/GitLab/Bitbucket

- [ ] **Import repository**
  - Select your `parkinson-detection` repository
  - Click "Import"

### Step 2: Configure Project

- [ ] **Set root directory**
  - Root Directory: `frontend`
  - Framework Preset: `Other`

- [ ] **Configure build**
  - Build Command: `npm run build`
  - Output Directory: `.` (current directory)
  - Install Command: `npm install` (or leave default)

### Step 3: Environment Variables

- [ ] **Add BACKEND_URL variable**
  - Click "Environment Variables"
  - Name: `BACKEND_URL`
  - Value: `https://freezingly-nonsignificative-edison.ngrok-free.dev`
  - ⚠️ **No trailing slash!**
  - Environments: Select ALL (Production, Preview, Development)

- [ ] **Verify variable is set**
  - Should see: `BACKEND_URL` in the list
  - Value should match your ngrok domain

### Step 4: Deploy

- [ ] **Click Deploy**
  - Wait for build to complete (usually 1-2 minutes)
  - Watch build logs for errors

- [ ] **Deployment succeeds**
  - Should see: "🎉 Congratulations! Your project has been deployed"
  - Note your Vercel URL: `https://your-project.vercel.app`

---

## 🧪 Testing Checklist

### Backend Verification

- [ ] **Backend is running locally**
  ```powershell
  .\run.ps1
  ```
  Expected: All services running

- [ ] **ngrok tunnel is active**
  - Check terminal: Should show ngrok URL
  - Test directly: `curl https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health`
  - Expected: `{"status": "healthy", ...}`

### Frontend Verification (Vercel)

- [ ] **Visit Vercel URL**
  - Open: `https://your-project.vercel.app`
  - Expected: Frontend loads correctly

- [ ] **Check console (F12)**
  - Should see: `🚀 Production mode: Using configured backend URL`
  - Should see: `✅ Backend is available and ready`
  - No errors related to backend URL

- [ ] **Run verification tool**
  - Visit: `https://your-project.vercel.app/verify.html`
  - All checks should be ✅ green

### Functionality Testing

- [ ] **Homepage loads**
  - Visit: `https://your-project.vercel.app`
  - Expected: Welcome screen displays

- [ ] **Start test works**
  - Click "Start Detection Test"
  - Expected: Test mode selection appears

- [ ] **Voice recording works**
  - Grant microphone permission
  - Select "Voice Only" or "Complete Analysis"
  - Click "Start Recording"
  - Speak for a few seconds
  - Expected: Recording completes, real-time metrics show

- [ ] **Tremor detection works**
  - Select "Tremor Only" or "Complete Analysis"
  - Click "Start Tremor Test"
  - Hold device steady
  - Expected: Motion data collected, metrics display

- [ ] **Results display**
  - After test completes
  - Expected: Results screen with confidence scores

- [ ] **Download works**
  - Click "Download Report"
  - Try both Simple and Detailed reports
  - Expected: Excel file downloads

### Local Development Still Works

- [ ] **Stop any running processes**
  ```powershell
  # Close PowerShell windows from run.ps1
  ```

- [ ] **Run locally again**
  ```powershell
  .\run.ps1
  ```
  Expected: Everything works as before

- [ ] **Visit local URL**
  - Open: `http://localhost:8000`
  - Expected: App works normally
  - Console should show: `🏠 Local development mode detected`

- [ ] **Test functionality**
  - Voice recording works ✓
  - Tremor detection works ✓
  - Results display ✓
  - Everything unchanged ✓

---

## 🐛 Troubleshooting Checklist

### If Deployment Fails

- [ ] **Check build logs in Vercel**
  - Look for errors in build output
  - Common issues:
    - `BACKEND_URL` not set
    - Missing files
    - Syntax errors

- [ ] **Verify environment variable**
  - Go to: Project Settings → Environment Variables
  - `BACKEND_URL` should exist
  - Value should be correct (no typos)
  - All environments should be checked

- [ ] **Redeploy**
  - Go to: Deployments tab
  - Click three dots (•••) on latest deployment
  - Click "Redeploy"

### If "Backend not available"

- [ ] **Backend is running?**
  ```powershell
  .\run.ps1
  ```

- [ ] **ngrok is active?**
  - Check terminal for ngrok URL
  - Visit ngrok URL in browser
  - Should see the app or API response

- [ ] **Test backend directly**
  ```bash
  curl https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health
  ```
  Expected: JSON response

- [ ] **Check CORS**
  - Backend should allow your Vercel domain
  - Check Flask CORS configuration

### If Config Not Loading

- [ ] **Files exist?**
  - `frontend/js/config.js` exists
  - `frontend/js/vercel-env.js` generated (on Vercel)

- [ ] **Load order correct?**
  - Check `index.html`
  - `config.js` loads before `app.js`

- [ ] **Console errors?**
  - Open browser console (F12)
  - Look for errors loading scripts

### If Local Development Broken

- [ ] **Files not changed?**
  - `server.py` unchanged
  - Backend code unchanged

- [ ] **Ports free?**
  ```powershell
  netstat -ano | findstr :5000
  netstat -ano | findstr :8000
  ```
  Should be empty or run by your processes

- [ ] **Run.ps1 works?**
  - Try running manually:
    ```powershell
    cd backend
    python app.py
    ```
  - In another terminal:
    ```powershell
    cd frontend
    python server.py 8000
    ```

---

## 📊 Success Criteria

Your deployment is successful when:

- ✅ Backend runs locally via `run.ps1`
- ✅ ngrok tunnel is active and accessible
- ✅ Vercel deployment succeeds
- ✅ Vercel URL loads the frontend
- ✅ Frontend connects to backend via ngrok
- ✅ Voice recording works on Vercel URL
- ✅ Tremor detection works on Vercel URL
- ✅ Results display correctly
- ✅ Local development still works (`run.ps1`)
- ✅ No console errors
- ✅ All sensors work (microphone, accelerometer)

---

## 🎯 Quick Commands Reference

### Local Development
```powershell
# Start everything
.\run.ps1

# Start backend only
cd backend
python app.py

# Start frontend only
cd frontend
python server.py 8000
```

### Git
```bash
# Check status
git status

# Add changes
git add frontend/

# Commit
git commit -m "Add Vercel deployment"

# Push
git push origin main
```

### Testing
```bash
# Test backend
curl http://localhost:5000/api/health

# Test ngrok
curl https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health
```

### Vercel CLI (Optional)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from terminal
cd frontend
vercel --prod
```

---

## 📞 Help Resources

| Issue | Document | Location |
|-------|----------|----------|
| Quick start | QUICKSTART_VERCEL.md | `frontend/` |
| Detailed guide | VERCEL_DEPLOYMENT.md | `frontend/` |
| File reference | FILES_REFERENCE.md | `frontend/` |
| This checklist | DEPLOYMENT_CHECKLIST.md | `frontend/` |

---

## ✨ Final Verification

Before you consider deployment complete:

- [ ] I can access my app on Vercel URL
- [ ] Voice and tremor tests work on production
- [ ] Local development still works with `run.ps1`
- [ ] I've tested on mobile device (via Vercel URL)
- [ ] I've documented my Vercel URL
- [ ] I know how to update BACKEND_URL if ngrok changes

**Vercel URL**: ________________________________

**ngrok Domain**: `https://freezingly-nonsignificative-edison.ngrok-free.dev`

**Deployment Date**: ____________________

---

🎉 **Congratulations! Your Parkinson Detection app is deployed!** 🎉

Share your Vercel URL with others to test the app from anywhere in the world! 🌍
