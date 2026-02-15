# Vercel Deployment Script for Windows PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Deploying Frontend to Vercel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to frontend directory
$frontendPath = "C:\Users\sruja\Classroom\SysntheticDataPCG\pcg_frontend"
Set-Location $frontendPath

Write-Host "Current directory: " -NoNewline
Write-Host $frontendPath -ForegroundColor Yellow
Write-Host ""

# Check if Vercel CLI is installed
Write-Host "[1/5] Checking Vercel CLI..." -ForegroundColor Green
try {
    $vercelVersion = vercel --version 2>&1 | Select-Object -First 1
    Write-Host "✓ Vercel CLI found: $vercelVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Vercel CLI not found. Installing..." -ForegroundColor Red
    npm install -g vercel
}
Write-Host ""

# Check login status
Write-Host "[2/5] Checking Vercel login status..." -ForegroundColor Green
$whoami = vercel whoami 2>&1 | Select-String -Pattern "^>" | ForEach-Object { $_.Line -replace '^>\s*', '' }
if ($whoami) {
    Write-Host "✓ Logged in as: $whoami" -ForegroundColor Green
} else {
    Write-Host "✗ Not logged in. Please run 'vercel login' first." -ForegroundColor Red
    Write-Host "Run this command in a terminal: vercel login" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Show project info
Write-Host "[3/5] Project Information" -ForegroundColor Green
Write-Host "  Name: PCG Frontend (ITSM Ticket Management)"
Write-Host "  Framework: Next.js 16"
Write-Host "  Build Command: npm run build"
Write-Host "  Output Directory: .next"
Write-Host ""

# Deploy to preview
Write-Host "[4/5] Deploying to Vercel (Preview)..." -ForegroundColor Green
Write-Host "This may take 2-5 minutes..." -ForegroundColor Yellow
Write-Host ""

# Run vercel deploy
$deployOutput = vercel --yes 2>&1

# Display output
$deployOutput | ForEach-Object { Write-Host $_ }

# Extract deployment URL
$previewUrl = $deployOutput | Select-String -Pattern "https://.*\.vercel\.app" | Select-Object -First 1 | ForEach-Object { $_.Matches.Value }

Write-Host ""
Write-Host "[5/5] Deployment Complete!" -ForegroundColor Green
Write-Host ""

if ($previewUrl) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Preview URL:" -ForegroundColor Cyan
    Write-Host "  $previewUrl" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Next Steps:" -ForegroundColor Green
    Write-Host "1. Test your preview deployment" -ForegroundColor White
    Write-Host "2. If everything works, deploy to production:" -ForegroundColor White
    Write-Host "   vercel --prod" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "3. Set environment variables in Vercel Dashboard:" -ForegroundColor White
    Write-Host "   • NEXT_PUBLIC_API_URL" -ForegroundColor Gray
    Write-Host "   • NEXT_PUBLIC_ML_API_URL" -ForegroundColor Gray
    Write-Host "   • NEXT_PUBLIC_GOOGLE_CLIENT_ID" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Update Google OAuth and Backend CORS with your Vercel URL" -ForegroundColor White
} else {
    Write-Host "⚠ Could not extract deployment URL from output." -ForegroundColor Yellow
    Write-Host "Please check the output above for the deployment URL." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "For more information, see: VERCEL_DEPLOYMENT.md" -ForegroundColor Cyan
Write-Host ""
