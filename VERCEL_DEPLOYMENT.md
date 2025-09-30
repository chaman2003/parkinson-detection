# 🚀 Vercel Deployment Guide

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **GitHub Repository**: Code pushed to GitHub (✅ Complete)

## 🎯 Frontend Deployment

### Option 1: Vercel Dashboard (Recommended)

1. **Connect GitHub Repository**
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Select `chaman2003/parkinson-detection`
   - Click "Import"

2. **Configure Project**
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave as root)
   - **Build Command**: Leave empty
   - **Output Directory**: `./`
   - **Install Command**: Leave empty

3. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Get your frontend URL: `https://parkinson-detection-xxx.vercel.app`

### Option 2: Vercel CLI

```bash
# In project root directory
vercel --prod

# Follow prompts:
# - Setup and deploy? Y
# - Which scope? (your account)
# - Link to existing project? N
# - Project name: parkinson-detection-frontend
# - Directory: ./
# - Settings correct? Y
```

## 🔧 Backend Deployment

### Option 1: Vercel Dashboard

1. **Create New Project**
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Select `chaman2003/parkinson-detection` again
   - Click "Import"

2. **Configure Backend Project**
   - **Framework Preset**: Other
   - **Root Directory**: `./backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Output Directory**: Leave empty
   - **Install Command**: Leave empty

3. **Environment Variables** (if needed)
   - Add any environment variables
   - `FLASK_ENV=production`

### Option 2: Vercel CLI

```bash
# Navigate to backend directory
cd backend

# Deploy
vercel --prod

# Follow prompts:
# - Setup and deploy? Y
# - Which scope? (your account)
# - Link to existing project? N
# - Project name: parkinson-detection-backend
# - Directory: ./
# - Settings correct? Y
```

## 🔗 Connect Frontend to Backend

After both deployments, update the frontend API URL:

1. **Get Backend URL**: `https://parkinson-detection-backend-xxx.vercel.app`

2. **Update Frontend Code**:
   ```javascript
   // In app.js, update API_BASE_URL
   const API_BASE_URL = 'https://your-backend-url.vercel.app';
   ```

3. **Redeploy Frontend** with updated API URL

## 📋 Deployment Checklist

### Frontend Deployment ✅
- [ ] GitHub repository connected
- [ ] Vercel project created
- [ ] Domain assigned
- [ ] PWA manifest accessible
- [ ] Service worker functional

### Backend Deployment ✅
- [ ] Python environment configured
- [ ] Dependencies installed
- [ ] Flask app running
- [ ] API endpoints accessible
- [ ] CORS configured

### Integration ✅
- [ ] Frontend points to backend URL
- [ ] API calls successful
- [ ] Cross-origin requests working
- [ ] File uploads functional

## 🌐 Custom Domains (Optional)

### Frontend Domain
```bash
vercel --prod --alias your-domain.com
```

### Backend Domain
```bash
cd backend
vercel --prod --alias api.your-domain.com
```

## 🔧 Troubleshooting

### Common Issues

1. **Backend 404 Errors**
   - Check `vercel.json` configuration
   - Ensure `app.py` is in root of backend directory
   - Verify Flask routes are properly defined

2. **CORS Errors**
   - Update Flask-CORS configuration
   - Add frontend domain to allowed origins

3. **Build Failures**
   - Check `requirements.txt` for correct dependencies
   - Verify Python version in `runtime.txt`

4. **Large File Uploads**
   - Vercel has 4.5MB limit for serverless functions
   - Consider using Vercel Edge Functions for larger files

## 📊 Performance Optimization

### Frontend
- Enable Vercel's automatic compression
- Use Vercel Analytics for monitoring
- Implement proper caching headers

### Backend
- Optimize ML model loading
- Use efficient data processing
- Implement proper error handling

## 🔒 Security Considerations

- Set proper CORS origins in production
- Use environment variables for sensitive data
- Implement rate limiting if needed
- Validate all input data

## 📈 Monitoring

- **Frontend**: Vercel Analytics Dashboard
- **Backend**: Vercel Functions Logs
- **Performance**: Web Vitals monitoring
- **Errors**: Function error logs

Your Parkinson's Detection PWA is now ready for global deployment! 🎉