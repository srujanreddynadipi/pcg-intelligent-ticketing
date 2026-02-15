# 🎯 Quick Start: Deploy to HuggingFace Spaces

## TL;DR - 3 Commands to Deploy

```bash
# 1. Run the deployment helper (Windows)
cd itsm-ai-api
.\deploy-to-hf.ps1

# OR (Linux/Mac)
chmod +x deploy-to-hf.sh
./deploy-to-hf.sh

# 2. Create Space at: https://huggingface.co/new-space
#    - Name: itsm-ai-api
#    - SDK: Docker
#    - Click "Create Space"

# 3. Push to HuggingFace
git push hf main
```

## Your API Will Be Live At:
```
https://YOUR-USERNAME-itsm-ai-api.hf.space
https://YOUR-USERNAME-itsm-ai-api.hf.space/docs  (API Documentation)
```

## Test Your Deployed API:
```bash
# Health check
curl https://YOUR-USERNAME-itsm-ai-api.hf.space/health

# Predict ticket
curl -X POST https://YOUR-USERNAME-itsm-ai-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"user":"test@company.com","title":"Email not working","description":"Cannot access my email"}'
```

## What Was Created:
- ✅ `README_HF_SPACES.md` - HuggingFace Space configuration
- ✅ `.dockerignore` - Optimized Docker build
- ✅ `HUGGINGFACE_DEPLOYMENT.md` - Complete deployment guide
- ✅ `deploy-to-hf.ps1` / `deploy-to-hf.sh` - Automated setup scripts

## Manual Deployment (Alternative):

### 1. Prepare Files
```bash
cd itsm-ai-api

# Backup original README
mv README.md README_GITHUB.md

# Use HF Spaces README
mv README_HF_SPACES.md README.md
```

### 2. Initialize Git & Add Remote
```bash
# Initialize (if not already)
git init

# Add HuggingFace remote (replace YOUR-USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/itsm-ai-api
```

### 3. Create Space on HuggingFace
Go to: https://huggingface.co/new-space
- **Name**: itsm-ai-api
- **SDK**: Docker  ⚠️ Must select Docker!
- **Visibility**: Public (or Private)

### 4. Push Code
```bash
git add .
git commit -m "Deploy to HuggingFace Spaces"
git push hf main
```

### 5. Wait for Build
- Go to your Space: https://huggingface.co/spaces/YOUR-USERNAME/itsm-ai-api
- Click **"Logs"** tab
- Build takes 5-15 minutes first time
- Status will change from "Building" → "Running"

## Why HuggingFace Spaces?

| Feature | HuggingFace Spaces | Render | Railway | Fly.io |
|---------|-------------------|---------|----------|--------|
| **RAM** | 16GB FREE | 512MB | 512MB | 256MB |
| **Always On** | ✅ Yes | ❌ Sleeps | ⚠️ Limited | ⚠️ Limited |
| **Build Time** | Fast | Medium | Fast | Fast |
| **ML Optimized** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Cost** | FREE | $7/mo | $5 credit | FREE |

**Winner**: HuggingFace Spaces for ML apps! 🏆

## Common Issues:

### "Repository not found" Error
**Fix**: Make sure you created the Space first on HuggingFace website

### "Permission denied" Error  
**Fix**: Login to HuggingFace CLI
```bash
pip install huggingface-hub
huggingface-cli login
```

### Build Timeout
**Fix**: Your models are cached - subsequent builds will be faster

### API Returns 404
**Fix**: Wait for build to complete (check Logs tab)

## Need More Help?
Read the complete guide: [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md)

## 🎉 That's It!
Your ITSM AI API is now globally accessible, running on HuggingFace infrastructure with 16GB RAM for free!
