# Quick Start: Vercel Deployment

## 🚀 Deploy in 5 Minutes

### Step 1: Start Your Backend
```powershell
.\run.ps1
```
This starts your Flask backend and creates the ngrok tunnel at:
`https://freezingly-nonsignificative-edison.ngrok-free.dev`

### Step 2: Deploy to Vercel

1. Go to https://vercel.com/new
2. Import your Git repository
3. **Configure:**
   - Root Directory: `frontend`
   - Framework Preset: `Other`
   - Build Command: `npm run build`
   - Output Directory: `.`

4. **Add Environment Variable:**
   - Name: `BACKEND_URL`
   - Value: `https://freezingly-nonsignificative-edison.ngrok-free.dev`
   - ⚠️ No trailing slash!

5. Click **Deploy**

### Step 3: Test
Visit your Vercel URL: `https://your-project.vercel.app`

## ✅ Verification Checklist

- [ ] Backend is running (`run.ps1`)
- [ ] ngrok tunnel is active (check terminal output)
- [ ] `BACKEND_URL` environment variable is set in Vercel
- [ ] Vercel deployment succeeded
- [ ] Frontend loads in browser
- [ ] Browser console shows: `🚀 Production mode: Using configured backend URL`
- [ ] Voice and tremor tests work

## 🔧 Local Development (Unchanged)

The local workflow remains the same:
```powershell
.\run.ps1
```

Then visit: `http://localhost:8000`

## 📝 Notes

- **Frontend**: Hosted on Vercel (globally distributed CDN)
- **Backend**: Running on your local machine via ngrok
- **Connection**: Frontend → ngrok → Your local Flask backend
- **No changes** needed to run locally - the app auto-detects the environment

## 🆘 Troubleshooting

**Problem**: "Backend not available"
- **Solution**: Ensure `run.ps1` is running and ngrok tunnel is active

**Problem**: CORS errors
- **Solution**: Update Flask CORS to allow your Vercel domain

**Problem**: Environment variable not working
- **Solution**: 
  1. Check spelling: `BACKEND_URL` (case-sensitive)
  2. Check value has no trailing slash
  3. Redeploy from Vercel dashboard

## 📚 Full Documentation

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for complete details.
