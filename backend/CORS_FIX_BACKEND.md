# Backend CORS Fix for Vercel Deployment

## Current CORS Configuration (lines 107-115 in backend/app.py)

```python
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type", "Cache-Control", "X-Accel-Buffering"]
    }
})
```

## Recommended Fix

Replace the CORS configuration with this enhanced version:

```python
# Enhanced CORS for Vercel deployment
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins (includes Vercel domain)
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Cache-Control", "X-Accel-Buffering"],
        "supports_credentials": False,
        "max_age": 3600  # Cache preflight requests for 1 hour
    }
})
```

## Alternative: Specific Origin (More Secure)

If you want to restrict to only your Vercel domain:

```python
# Secure CORS for production
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://parkinson-detection.vercel.app",  # Your Vercel domain
            "http://localhost:8000",                    # Local development
            "http://127.0.0.1:8000",                    # Local alternative
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Cache-Control", "X-Accel-Buffering"],
        "supports_credentials": False,
        "max_age": 3600
    }
})
```

## How to Apply

### Option 1: Edit backend/app.py Directly

1. Open `backend/app.py`
2. Find the CORS configuration (around line 107)
3. Replace with the enhanced version above
4. Save the file
5. Restart backend: Stop and run `.\run.ps1` again

### Option 2: Use a Patch File

Create this code in `backend/cors_config.py`:

```python
def configure_cors(app):
    """Enhanced CORS configuration for Vercel deployment"""
    from flask_cors import CORS
    
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "expose_headers": ["Content-Type", "Cache-Control", "X-Accel-Buffering"],
            "supports_credentials": False,
            "max_age": 3600
        }
    })
```

Then in `backend/app.py`, replace the CORS setup with:

```python
from cors_config import configure_cors

app = Flask(__name__)
configure_cors(app)  # Apply enhanced CORS
```

## After Applying Fix

1. **Restart backend**:
   ```powershell
   # Stop current backend (Ctrl+C)
   .\run.ps1
   ```

2. **Test from Vercel**:
   - Visit your Vercel URL
   - Open browser console (F12)
   - Look for CORS errors
   - Should now work! ✅

3. **Verify health endpoint**:
   ```
   Open: https://your-vercel-app.vercel.app
   Console should show: "✅ Backend is available and ready"
   ```

## Troubleshooting

If CORS errors persist:

1. **Clear browser cache** or test in incognito
2. **Check backend logs** for errors
3. **Verify ngrok is running** with correct domain
4. **Test backend directly** in browser:
   ```
   https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health
   ```

## The Issue Explained

The CORS error occurs because:
1. Frontend is on `https://parkinson-detection.vercel.app`
2. Backend is on `https://freezingly-nonsignificative-edison.ngrok-free.dev`
3. Browser blocks cross-origin requests without proper CORS headers
4. Flask needs to explicitly allow the Vercel domain

The fix ensures Flask sends proper `Access-Control-Allow-Origin` headers that allow the browser to make requests from your Vercel domain to the ngrok backend.
