# HuggingFace Spaces Deployment Helper Script
# Run this from the itsm-ai-api directory

Write-Host "HuggingFace Spaces Deployment Helper" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if we're in the right directory
if (-not (Test-Path "Dockerfile"))
{
    Write-Host "Error: Dockerfile not found!" -ForegroundColor Red
    Write-Host "Please run this script from the itsm-ai-api directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found itsm-ai-api project files" -ForegroundColor Green
Write-Host ""

# Step 2: Get HuggingFace username
$username = Read-Host "Enter your HuggingFace username"
if ([string]::IsNullOrWhiteSpace($username))
{
    Write-Host "Username cannot be empty!" -ForegroundColor Red
    exit 1
}

# Step 3: Get Space name
$spaceName = Read-Host "Enter your Space name (default: itsm-ai-api)"
if ([string]::IsNullOrWhiteSpace($spaceName))
{
    $spaceName = "itsm-ai-api"
}

$spaceUrl = "https://huggingface.co/spaces/$username/$spaceName"

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "   Username: $username"
Write-Host "   Space Name: $spaceName"
Write-Host "   Space URL: $spaceUrl"
Write-Host ""

# Step 4: Confirm
$confirm = Read-Host "Continue with deployment preparation? (y/n)"
if (($confirm -ne "y") -and ($confirm -ne "Y"))
{
    Write-Host "Deployment cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Preparing files for deployment..." -ForegroundColor Cyan

# Step 5: Backup and replace README
if ((Test-Path "README.md") -and (-not (Test-Path "README_GITHUB.md")))
{
    Write-Host "   Backing up README.md to README_GITHUB.md..."
    Copy-Item "README.md" "README_GITHUB.md"
    Write-Host "   Backup created" -ForegroundColor Green
}

if (Test-Path "README_HF_SPACES.md")
{
    Write-Host "   Using HuggingFace Spaces README..."
    Copy-Item "README_HF_SPACES.md" "README.md" -Force
    
    # Update the README with actual username
    $readmeContent = Get-Content "README.md" -Raw
    $readmeContent = $readmeContent -replace "YOUR-USERNAME", $username
    $readmeContent = $readmeContent -replace "YOUR-SPACE-NAME", $spaceName
    Set-Content "README.md" $readmeContent
    
    Write-Host "   README.md prepared with your Space details" -ForegroundColor Green
}
else
{
    Write-Host "   Warning: README_HF_SPACES.md not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Files prepared successfully!" -ForegroundColor Green
Write-Host ""

# Step 6: Git setup
Write-Host "Git Configuration" -ForegroundColor Cyan
Write-Host ""

$gitSetup = Read-Host "Do you want to set up Git remote for HuggingFace? (y/n)"
if (($gitSetup -eq "y") -or ($gitSetup -eq "Y"))
{
    # Check if git is initialized
    if (-not (Test-Path ".git"))
    {
        Write-Host "   Initializing Git repository..."
        git init
    }
    
    # Check if HF remote exists
    $remotes = git remote 2>&1 | Out-String
    
    if ($remotes -match "hf")
    {
        Write-Host "   Remote 'hf' already exists. Updating URL..." -ForegroundColor Yellow
        git remote set-url hf "https://huggingface.co/spaces/$username/$spaceName"
    }
    else
    {
        Write-Host "   Adding HuggingFace remote..."
        git remote add hf "https://huggingface.co/spaces/$username/$spaceName"
    }
    
    Write-Host "   Git remote configured" -ForegroundColor Green
    Write-Host ""
    
    # Stage files
    $stageFiles = Read-Host "Stage files for commit? (y/n)"
    if (($stageFiles -eq "y") -or ($stageFiles -eq "Y"))
    {
        Write-Host "   Staging files..."
        git add .
        Write-Host "   Files staged" -ForegroundColor Green
        Write-Host ""
        
        # Commit
        $commitMsg = Read-Host "Enter commit message (default: Deploy to HuggingFace Spaces)"
        if ([string]::IsNullOrWhiteSpace($commitMsg))
        {
            $commitMsg = "Deploy to HuggingFace Spaces"
        }
        
        git commit -m "$commitMsg"
        Write-Host "   Changes committed" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Preparation Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a Space on HuggingFace:"
Write-Host "   - Go to: https://huggingface.co/new-space" -ForegroundColor Yellow
Write-Host "   - Name: $spaceName" -ForegroundColor Yellow
Write-Host "   - SDK: Docker" -ForegroundColor Yellow
Write-Host "   - Visibility: Public or Private" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Push to HuggingFace:"
Write-Host "   git push hf main" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Monitor deployment:"
Write-Host "   $spaceUrl" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Once deployed, your API will be at:"
Write-Host "   https://$username-$spaceName.hf.space" -ForegroundColor Yellow
Write-Host "   https://$username-$spaceName.hf.space/docs (API Docs)" -ForegroundColor Yellow
Write-Host ""
Write-Host "For detailed instructions, see: HUGGINGFACE_DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Good luck!" -ForegroundColor Green
