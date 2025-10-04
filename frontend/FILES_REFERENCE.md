# 📦 Vercel Deployment - Files Reference

## 📁 All Created/Modified Files

### ✨ New Configuration Files

| File | Purpose | Important? |
|------|---------|-----------|
| `vercel.json` | Vercel deployment configuration | ⭐⭐⭐ Required |
| `package.json` | npm build configuration | ⭐⭐⭐ Required |
| `vercel-build.js` | Build script to inject environment variables | ⭐⭐⭐ Required |
| `js/config.js` | Environment-aware backend URL handler | ⭐⭐⭐ Required |
| `.env.example` | Environment variable template | ⭐⭐ Documentation |
| `.gitignore` | Excludes generated files from git | ⭐⭐ Recommended |

### 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `DEPLOYMENT_SUMMARY.md` | Overview of all changes | Developers |
| `QUICKSTART_VERCEL.md` | 5-minute deployment guide | Everyone |
| `VERCEL_DEPLOYMENT.md` | Complete deployment documentation | Detailed reference |
| `verify.html` | Deployment verification tool | Testing |
| `FILES_REFERENCE.md` | This file - complete file listing | Reference |

### 🔧 Modified Files

| File | Changes | Why |
|------|---------|-----|
| `index.html` | Added config.js and vercel-env.js loading | Load environment configuration |
| `js/app.js` | Uses AppConfig for backend URL | Support both local and Vercel |
| `../README.md` | Added Vercel deployment section | Documentation |

---

## 📋 File Descriptions

### Core Configuration

#### `vercel.json`
```json
{
  "buildCommand": "npm run build",
  "headers": { ... },
  "rewrites": [ ... ]
}
```
**Purpose**: Tells Vercel how to build and configure your app
**Required**: Yes
**Modify**: Only if changing build settings

#### `package.json`
```json
{
  "scripts": {
    "build": "node vercel-build.js"
  }
}
```
**Purpose**: npm configuration for Vercel build process
**Required**: Yes
**Modify**: Only if changing build process

#### `vercel-build.js`
```javascript
// Injects BACKEND_URL into vercel-env.js
const backendUrl = process.env.BACKEND_URL;
fs.writeFileSync('js/vercel-env.js', ...);
```
**Purpose**: Generates `js/vercel-env.js` with environment variable during Vercel build
**Required**: Yes
**Modify**: Only if changing environment variable logic

#### `js/config.js`
```javascript
const AppConfig = {
    getBackendUrl() {
        // Auto-detects local vs production
        // Returns '/api' for local, BACKEND_URL for Vercel
    }
};
```
**Purpose**: Provides environment-aware backend URL
**Required**: Yes
**Modify**: Only if changing URL logic

---

## 🔄 How Files Work Together

### Local Development (`run.ps1`)

```
1. User runs: .\run.ps1
2. Backend starts: localhost:5000
3. Frontend starts: localhost:8000 (server.py)
4. Browser loads: index.html
5. index.html loads: js/config.js
6. config.js detects: hostname === 'localhost'
7. Returns backend URL: '/api'
8. server.py proxies: /api/* → localhost:5000
9. ✅ Everything works!
```

**Files involved**:
- `index.html` - Main HTML
- `js/config.js` - Detects local mode
- `js/app.js` - Uses config for API calls
- `server.py` - Proxies /api/* requests

**Generated files**: None

---

### Vercel Deployment

```
1. Push code to Git
2. Vercel detects: package.json
3. Vercel runs: npm run build
4. Build script runs: vercel-build.js
5. vercel-build.js reads: process.env.BACKEND_URL
6. vercel-build.js creates: js/vercel-env.js
7. Vercel deploys: All files to CDN
8. User visits: https://your-project.vercel.app
9. Browser loads: index.html
10. index.html loads: js/vercel-env.js (sets window.BACKEND_URL)
11. index.html loads: js/config.js
12. config.js detects: hostname !== 'localhost'
13. Returns backend URL: window.BACKEND_URL
14. js/app.js makes API calls: To ngrok domain
15. ✅ Everything works!
```

**Files involved**:
- `vercel.json` - Build configuration
- `package.json` - npm configuration
- `vercel-build.js` - Generate vercel-env.js
- `js/vercel-env.js` - **Generated during build** (not in repo)
- `index.html` - Main HTML
- `js/config.js` - Detects Vercel mode
- `js/app.js` - Uses config for API calls

**Generated files**:
- `js/vercel-env.js` - Created by `vercel-build.js`

---

## 🔍 File Dependencies

### Load Order (Important!)

```html
<!-- In index.html -->
<script src="js/vercel-env.js"></script>    <!-- 1. Loads first (sets window.BACKEND_URL) -->
<script src="js/config.js"></script>         <!-- 2. Then config (reads window.BACKEND_URL) -->
<script src="js/app.js"></script>            <!-- 3. Finally app (uses config) -->
```

**Why this order matters**:
1. `vercel-env.js` must load first to set `window.BACKEND_URL`
2. `config.js` reads `window.BACKEND_URL` to determine backend URL
3. `app.js` uses `AppConfig` to get the backend URL

### Dependency Graph

```
vercel.json
    ↓ (build config)
package.json
    ↓ (build script)
vercel-build.js
    ↓ (generates)
js/vercel-env.js (GENERATED)
    ↓ (sets window.BACKEND_URL)
js/config.js
    ↓ (reads window.BACKEND_URL)
js/app.js
    ↓ (uses AppConfig)
Backend API calls
```

---

## 🚫 Files NOT in Repository

These files are **generated** during deployment:

| File | Generated By | When | Git? |
|------|-------------|------|------|
| `js/vercel-env.js` | `vercel-build.js` | Vercel build | ❌ No (in .gitignore) |
| `.vercel/` | Vercel CLI | Deployment | ❌ No (in .gitignore) |
| `node_modules/` | npm | If you run npm install | ❌ No (in .gitignore) |

**Why not in Git?**
- `js/vercel-env.js` contains environment-specific values
- `.vercel/` contains deployment metadata
- `node_modules/` are installed dependencies

---

## ✅ Pre-Deployment Checklist

Before deploying to Vercel, ensure these files exist:

### Required Files (Must Exist)
- [ ] `vercel.json` - Deployment config
- [ ] `package.json` - Build config
- [ ] `vercel-build.js` - Build script
- [ ] `js/config.js` - Environment handler
- [ ] `index.html` - Main HTML (with config.js loaded)
- [ ] `js/app.js` - Uses AppConfig

### Documentation Files (Recommended)
- [ ] `.env.example` - Environment variable template
- [ ] `QUICKSTART_VERCEL.md` - Quick guide
- [ ] `VERCEL_DEPLOYMENT.md` - Complete guide
- [ ] `.gitignore` - Exclude generated files

### Should NOT Exist in Repo
- [ ] `js/vercel-env.js` - Generated during build
- [ ] `.vercel/` - Generated by Vercel
- [ ] `node_modules/` - Installed dependencies

---

## 🔧 Customization Guide

### Change Backend URL (Vercel)

**In Vercel Dashboard**:
1. Go to project settings
2. Environment Variables
3. Edit `BACKEND_URL`
4. Redeploy

**No code changes needed!**

### Change Build Process

**Edit `vercel-build.js`**:
```javascript
// Add more environment variables
const config = {
    BACKEND_URL: process.env.BACKEND_URL,
    API_KEY: process.env.API_KEY,  // Add new variable
};
```

**Edit `vercel.json`**:
```json
{
  "buildCommand": "npm run build && npm run validate"
}
```

### Add New Environment Variables

1. **Add to Vercel Dashboard**:
   - Name: `NEW_VARIABLE`
   - Value: `your_value`

2. **Update `vercel-build.js`**:
   ```javascript
   window.NEW_VARIABLE = '${process.env.NEW_VARIABLE}';
   ```

3. **Use in `js/config.js`**:
   ```javascript
   getNewVariable() {
       return window.NEW_VARIABLE || 'default_value';
   }
   ```

---

## 📊 File Size Reference

| File | Size | Type |
|------|------|------|
| `vercel.json` | ~700 bytes | JSON Config |
| `package.json` | ~300 bytes | JSON Config |
| `vercel-build.js` | ~1.2 KB | Node.js Script |
| `js/config.js` | ~1.5 KB | JavaScript |
| `js/vercel-env.js` | ~100 bytes | Generated JS |
| `.env.example` | ~300 bytes | Text |
| `QUICKSTART_VERCEL.md` | ~2 KB | Markdown |
| `VERCEL_DEPLOYMENT.md` | ~8 KB | Markdown |

**Total**: ~14 KB of configuration and documentation

---

## 🎯 Quick Reference

### Local Development
```powershell
.\run.ps1
# Visit: http://localhost:8000
```

**Files used**: `js/config.js` (detects local), `server.py` (proxy)

### Vercel Deployment
```bash
# Push to Git
git add .
git commit -m "Vercel deployment ready"
git push

# Deploy in Vercel dashboard
# Set BACKEND_URL environment variable
```

**Files used**: All files, `vercel-build.js` generates `js/vercel-env.js`

### Verification
```
Visit: https://your-domain.vercel.app/verify.html
```

**Files used**: `verify.html`, `js/config.js`, `js/vercel-env.js`

---

## 🆘 Troubleshooting by File

### Issue: "Backend not available"

**Check**: `js/config.js`
```javascript
// Should return correct URL
console.log(window.AppConfig.getBackendUrl());
```

**Check**: `js/vercel-env.js` (on Vercel)
```javascript
// Should exist and contain BACKEND_URL
console.log(window.BACKEND_URL);
```

**Fix**: 
1. Verify `BACKEND_URL` in Vercel settings
2. Redeploy to regenerate `vercel-env.js`

### Issue: "Config not loaded"

**Check**: `index.html`
```html
<!-- Verify load order -->
<script src="js/vercel-env.js"></script>
<script src="js/config.js"></script>
<script src="js/app.js"></script>
```

**Fix**: Ensure `js/config.js` exists and is loaded before `js/app.js`

### Issue: "Build fails on Vercel"

**Check**: `vercel-build.js`
```javascript
// Should have BACKEND_URL
if (!backendUrl) {
    console.error('BACKEND_URL not set!');
    process.exit(1);
}
```

**Fix**: Set `BACKEND_URL` environment variable in Vercel

---

## 📞 Support Files

| Issue | Check File | Solution |
|-------|-----------|----------|
| Deployment config | `vercel.json` | Verify build command |
| Build fails | `vercel-build.js` | Check env vars |
| Backend URL wrong | `js/config.js` | Check detection logic |
| Environment vars | `js/vercel-env.js` | Redeploy to regenerate |
| Local not working | `server.py` | Check proxy config |

---

This reference guide covers all files related to the Vercel deployment. Keep it handy for troubleshooting and modifications! 🚀
