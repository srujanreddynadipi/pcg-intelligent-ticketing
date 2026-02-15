# 🚀 HuggingFace Spaces Deployment Guide

Complete guide to deploy the ITSM AI API to HuggingFace Spaces.

## 🎯 Why HuggingFace Spaces?

✅ **2 vCPU, 16GB RAM** - Perfect for ML models with sentence-transformers  
✅ **Free Tier** - Generous resources for demos and hackathons  
✅ **Always-On** - No cold starts like other free platforms  
✅ **HuggingFace Integration** - Direct access to your model hub  
✅ **Built for ML** - Optimized for AI/ML applications  

## 📋 Prerequisites

1. **HuggingFace Account** - Sign up at https://huggingface.co
2. **Git** - Installed on your machine
3. **HuggingFace CLI** (optional but recommended)
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

## 🔧 Step-by-Step Deployment

### Method 1: Using HuggingFace Web Interface (Easiest)

#### 1. Create a New Space
1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Space name**: `itsm-ai-api` (or your choice)
   - **License**: MIT
   - **Space SDK**: Select **Docker**
   - **Visibility**: Public or Private (your choice)
4. Click **"Create Space"**

#### 2. Prepare Your Repository
Navigate to your itsm-ai-api folder:
```bash
cd C:\Users\sruja\Classroom\SysntheticDataPCG\itsm-ai-api
```

#### 3. Replace README.md
The README.md needs HuggingFace-specific YAML frontmatter:
```bash
# Backup original README
mv README.md README_GITHUB.md

# Use the HF Spaces README
mv README_HF_SPACES.md README.md
```

#### 4. Verify Dockerfile
Your existing Dockerfile should work as-is. It:
- Uses Python 3.11-slim (lightweight)
- Installs dependencies from requirements.txt
- Exposes port 8000
- Includes health check

#### 5. Push to HuggingFace Space

**Option A: Using Git (Recommended)**
```bash
# Add HuggingFace remote (replace YOUR-USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/itsm-ai-api

# Add files
git add .
git commit -m "Deploy to HuggingFace Spaces"

# Push to HuggingFace
git push hf main
```

**Option B: Using HuggingFace CLI**
```bash
huggingface-cli repo create itsm-ai-api --type space --space-sdk docker

# Clone the space repo
git clone https://huggingface.co/spaces/YOUR-USERNAME/itsm-ai-api hf-space

# Copy files
cp -r app config utils Dockerfile requirements.txt runtime.txt README.md .dockerignore hf-space/
cd hf-space

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option C: Upload via Web Interface**
1. Go to your Space page
2. Click **"Files"** tab
3. Click **"Add file"** > **"Upload files"**
4. Upload these files:
   - `Dockerfile`
   - `requirements.txt`
   - `runtime.txt`
   - `README.md` (the HF_SPACES version)
   - `.dockerignore`
   - Entire `app/` folder
   - Entire `config/` folder
   - Entire `utils/` folder

#### 6. Wait for Build
- HuggingFace will automatically build your Docker image
- Check the **"Logs"** tab to monitor progress
- First build takes 5-15 minutes (downloads PyTorch, models, etc.)
- Subsequent builds are cached and faster

#### 7. Access Your API
Once deployed, your API will be available at:
```
https://YOUR-USERNAME-itsm-ai-api.hf.space
```

**Test it:**
```bash
# Health check
curl https://YOUR-USERNAME-itsm-ai-api.hf.space/health

# API documentation
# Visit in browser:
https://YOUR-USERNAME-itsm-ai-api.hf.space/docs
```

---

### Method 2: Using Git from Existing Repository

If your code is already on GitHub:

#### 1. Create Space on HuggingFace (as above)

#### 2. Add HuggingFace as Remote
```bash
cd itsm-ai-api
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/itsm-ai-api
```

#### 3. Prepare for HF Spaces
```bash
# Backup and replace README
git mv README.md README_GITHUB.md
git mv README_HF_SPACES.md README.md
git add .dockerignore
git commit -m "Prepare for HuggingFace Spaces deployment"
```

#### 4. Push to Both Remotes
```bash
# Push to GitHub (keep your source)
git push origin main

# Push to HuggingFace (deploy)
git push hf main
```

---

## 🎨 Customization

### Update Space Settings
Edit the YAML frontmatter in README.md:
```yaml
---
title: ITSM AI Ticketing API          # Space display name
emoji: 🎫                              # Icon for your space
colorFrom: blue                        # Gradient start color
colorTo: purple                        # Gradient end color
sdk: docker                            # Must be 'docker'
python_version: 3.11                   # Python version
app_port: 8000                         # FastAPI port
pinned: false                          # Pin to your profile
license: mit                           # License type
tags:                                  # Searchable tags
  - fastapi
  - machine-learning
  - itsm
---
```

### Environment Variables (if needed)
1. Go to Space **Settings**
2. Add secrets under **"Repository secrets"**
3. Access in code via `os.environ.get('SECRET_NAME')`

### Enable Always-On (Optional)
1. Go to Space **Settings**
2. Under **"Sleep time"**, toggle **"Always running"**
3. Prevents space from sleeping after inactivity

---

## 📊 Monitor Your Deployment

### Check Logs
1. Go to your Space page
2. Click **"Logs"** tab
3. View real-time build and runtime logs

### View Metrics
- HuggingFace shows CPU/RAM usage
- Monitor request counts
- Track uptime

---

## 🐛 Troubleshooting

### Build Fails - Out of Memory
**Problem**: Docker build runs out of RAM  
**Solution**: Optimize requirements.txt
```bash
# Use CPU-only PyTorch (smaller)
# Add to requirements.txt:
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.0+cpu
```

### Build Fails - Timeout
**Problem**: Build takes too long (>60 min)  
**Solution**: 
1. Remove unused dependencies
2. Use smaller base image
3. Pre-download cached layers

### Models Not Loading
**Problem**: Can't download from HuggingFace Hub  
**Solution**: Check network access and model permissions
```python
# In model_loader.py, add retry logic
from huggingface_hub import hf_hub_download
import time

for attempt in range(3):
    try:
        # Download model
        break
    except:
        time.sleep(5)
```

### API Returns 502/504
**Problem**: App crashes or times out  
**Solution**: Check Logs tab for Python errors

### Port Issues
**Problem**: "Address already in use"  
**Solution**: Verify app_port in README.md matches main.py
```python
# main.py should use port from environment or 8000
import os
port = int(os.environ.get("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

---

## 🔄 Update Your Deployment

### Quick Updates
```bash
# Make changes
git add .
git commit -m "Update feature X"

# Push to HuggingFace (auto-redeploys)
git push hf main
```

### Rollback
```bash
# View commits
git log

# Rollback to previous commit
git reset --hard COMMIT_HASH
git push hf main --force
```

---

## 💰 Cost & Limits

### Free Tier (Public Spaces)
- ✅ 2 vCPU
- ✅ 16GB RAM
- ✅ 50GB storage
- ✅ Unlimited requests
- ⚠️ May sleep after inactivity (enable Always-On)

### Upgraded Tier (Optional)
- More CPU/GPU options
- Persistent storage
- Private spaces
- Priority support

**Pricing**: Check https://huggingface.co/pricing

---

## 🎯 Production Checklist

Before going live:

- [ ] Test all API endpoints locally
- [ ] Verify health check works
- [ ] Test with real ticket data
- [ ] Enable Always-On (if needed)
- [ ] Add custom domain (optional)
- [ ] Set up monitoring/alerts
- [ ] Document API for users
- [ ] Add rate limiting (if needed)
- [ ] Enable HTTPS (automatic on HF)
- [ ] Add authentication (if needed)

---

## 🚀 Post-Deployment

### Share Your Space
Your API is now live! Share the URL:
```
https://YOUR-USERNAME-itsm-ai-api.hf.space
```

### Embed in Applications
```python
import requests

API_URL = "https://YOUR-USERNAME-itsm-ai-api.hf.space/predict"

response = requests.post(
    API_URL,
    json={
        "user": "user@company.com",
        "title": "Printer not working",
        "description": "Office printer shows error"
    }
)

print(response.json())
```

### Showcase
- Add to your HuggingFace profile
- Share on social media
- Include in your hackathon submission
- Link from GitHub README

---

## 📚 Resources

- [HuggingFace Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

---

## 🤝 Support

**Issues?** 
1. Check Logs tab in your Space
2. Review Troubleshooting section above
3. HuggingFace Community forum: https://discuss.huggingface.co

**Need Help?**
- HuggingFace Discord: https://discord.gg/hugging-face
- GitHub Issues: Create an issue in your repo

---

## 🎉 Congratulations!

Your ITSM AI API is now live on HuggingFace Spaces! 🚀

**Next Steps:**
1. Test the API thoroughly
2. Share with your team
3. Monitor performance
4. Collect feedback
5. Iterate and improve

Good luck with your hackathon! 🏆
