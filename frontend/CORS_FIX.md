# 🔧 CORS Configuration for ngrok Backend

## Issue: CORS Error on Vercel

When deploying to Vercel, you may see this error:
```
Access to fetch at 'https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health' 
from origin 'https://parkinson-detection.vercel.app' has been blocked by CORS policy: 
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

This happens because your Flask backend needs to allow requests from your Vercel domain.

## Solution: Update Backend CORS Configuration

### Step 1: Find Your Vercel Domain

After deploying to Vercel, note your URL, for example:
- `https://parkinson-detection.vercel.app`

### Step 2: Update Flask CORS Settings

Open `backend/app.py` and find the CORS configuration (around line 105):

```python
# Current CORS setup
CORS(app)
```

Update it to specifically allow your Vercel domain:

```python
# Updated CORS setup for Vercel
CORS(app, origins=[
    "https://parkinson-detection.vercel.app",  # Your Vercel domain
    "http://localhost:8000",                    # Local development
    "http://127.0.0.1:8000",                    # Local development alternative
    "*"                                          # Allow all (less secure, but works for testing)
])
```

**Or**, for maximum compatibility during testing, use:

```python
# Allow all origins (development/testing only)
CORS(app, origins="*", supports_credentials=False)
```

### Step 3: Restart Backend

After making changes:

```powershell
# Stop the current backend (Ctrl+C in the terminal)
# Restart using run.ps1
.\run.ps1
```

### Step 4: Verify

1. Visit your Vercel URL
2. Open browser console (F12)
3. Check for CORS errors
4. If gone, backend is now accessible! ✅

## Alternative: ngrok CORS Headers

ngrok might strip CORS headers. To fix this:

### Option 1: Use ngrok Configuration

Create `ngrok.yml` in your project root:

```yaml
version: "2"
authtoken: YOUR_NGROK_AUTH_TOKEN
tunnels:
  backend:
    proto: http
    addr: 8000
    response_headers:
      add:
        - "Access-Control-Allow-Origin: *"
        - "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS"
        - "Access-Control-Allow-Headers: Content-Type, Authorization"
```

Then run ngrok with config:
```powershell
ngrok start --config=ngrok.yml backend
```

### Option 2: Update Flask Response Headers

In `backend/app.py`, add after creating the Flask app:

```python
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
```

## Quick Fix (Temporary)

For immediate testing, use this in `backend/app.py`:

```python
from flask_cors import CORS

app = Flask(__name__)

# Allow all origins - TEMPORARY for testing only
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

## Verification

After applying the fix, test:

```bash
# Test from command line
curl -H "Origin: https://parkinson-detection.vercel.app" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     https://freezingly-nonsignificative-edison.ngrok-free.dev/api/health
```

Should return headers including:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
```

## Security Note

**For Production:**
- Use specific origins, not `"*"`
- Restrict to your Vercel domain only
- Enable credentials only if needed

**For Development/Testing:**
- `"*"` is acceptable
- Makes testing easier
- Update before final deployment

## Example: Complete CORS Setup

Here's a complete example for `backend/app.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# CORS Configuration
if os.environ.get('FLASK_ENV') == 'production':
    # Production: Restrict to specific domain
    CORS(app, origins=[
        "https://parkinson-detection.vercel.app"
    ])
else:
    # Development: Allow all origins
    CORS(app, origins="*")

# Rest of your Flask app code...
```

## Still Having Issues?

1. **Check Flask logs** - Look for CORS-related errors
2. **Test backend directly** - Visit ngrok URL in browser
3. **Check ngrok dashboard** - Visit http://127.0.0.1:4040
4. **Verify backend URL** - Ensure it's correct in Vercel environment variable
5. **Clear browser cache** - Or test in incognito mode

## Contact

If issues persist after following this guide:
1. Check Flask backend logs for errors
2. Verify ngrok tunnel is active
3. Test backend health endpoint directly
4. Review browser console for specific error messages
