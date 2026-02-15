#!/bin/bash
# HuggingFace Spaces Deployment Helper Script (Linux/Mac)
# Run this from the itsm-ai-api directory

echo "🚀 HuggingFace Spaces Deployment Helper"
echo "======================================="
echo ""

# Step 1: Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found!"
    echo "Please run this script from the itsm-ai-api directory"
    exit 1
fi

echo "✅ Found itsm-ai-api project files"
echo ""

# Step 2: Get HuggingFace username
read -p "Enter your HuggingFace username: " username
if [ -z "$username" ]; then
    echo "❌ Username cannot be empty!"
    exit 1
fi

# Step 3: Get Space name
read -p "Enter your Space name (default: itsm-ai-api): " spaceName
if [ -z "$spaceName" ]; then
    spaceName="itsm-ai-api"
fi

spaceUrl="https://huggingface.co/spaces/$username/$spaceName"

echo ""
echo "📝 Configuration:"
echo "   Username: $username"
echo "   Space Name: $spaceName"
echo "   Space URL: $spaceUrl"
echo ""

# Step 4: Confirm
read -p "Continue with deployment preparation? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

echo ""
echo "🔧 Preparing files for deployment..."

# Step 5: Backup and replace README
if [ -f "README.md" ] && [ ! -f "README_GITHUB.md" ]; then
    echo "   📄 Backing up README.md to README_GITHUB.md..."
    cp README.md README_GITHUB.md
    echo "   ✅ Backup created"
fi

if [ -f "README_HF_SPACES.md" ]; then
    echo "   📄 Using HuggingFace Spaces README..."
    cp README_HF_SPACES.md README.md
    
    # Update the README with actual username
    sed -i.bak "s/YOUR-USERNAME/$username/g" README.md
    sed -i.bak "s/YOUR-SPACE-NAME/$spaceName/g" README.md
    rm README.md.bak 2>/dev/null
    
    echo "   ✅ README.md prepared with your Space details"
else
    echo "   ⚠️  Warning: README_HF_SPACES.md not found"
fi

echo ""
echo "✅ Files prepared successfully!"
echo ""

# Step 6: Git setup
echo "🔧 Git Configuration"
echo ""

read -p "Do you want to set up Git remote for HuggingFace? (y/n): " gitSetup
if [ "$gitSetup" = "y" ] || [ "$gitSetup" = "Y" ]; then
    
    # Check if git is initialized
    if [ ! -d ".git" ]; then
        echo "   📦 Initializing Git repository..."
        git init
    fi
    
    # Check if HF remote exists
    if git remote | grep -q "^hf$"; then
        echo "   ⚠️  Remote 'hf' already exists. Updating URL..."
        git remote set-url hf "https://huggingface.co/spaces/$username/$spaceName"
    else
        echo "   📦 Adding HuggingFace remote..."
        git remote add hf "https://huggingface.co/spaces/$username/$spaceName"
    fi
    
    echo "   ✅ Git remote configured"
    echo ""
    
    # Stage files
    read -p "Stage files for commit? (y/n): " stageFiles
    if [ "$stageFiles" = "y" ] || [ "$stageFiles" = "Y" ]; then
        echo "   📦 Staging files..."
        git add .
        echo "   ✅ Files staged"
        echo ""
        
        # Commit
        read -p "Enter commit message (default: Deploy to HuggingFace Spaces): " commitMsg
        if [ -z "$commitMsg" ]; then
            commitMsg="Deploy to HuggingFace Spaces"
        fi
        
        git commit -m "$commitMsg"
        echo "   ✅ Changes committed"
    fi
fi

echo ""
echo "🎉 Preparation Complete!"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Create a Space on HuggingFace:"
echo "   - Go to: https://huggingface.co/new-space"
echo "   - Name: $spaceName"
echo "   - SDK: Docker"
echo "   - Visibility: Public or Private"
echo ""
echo "2. Push to HuggingFace:"
echo "   git push hf main"
echo ""
echo "3. Monitor deployment:"
echo "   $spaceUrl"
echo ""
echo "4. Once deployed, your API will be at:"
echo "   https://$username-$spaceName.hf.space"
echo "   https://$username-$spaceName.hf.space/docs (API Docs)"
echo ""
echo "📚 For detailed instructions, see: HUGGINGFACE_DEPLOYMENT.md"
echo ""
echo "Good luck! 🚀"
