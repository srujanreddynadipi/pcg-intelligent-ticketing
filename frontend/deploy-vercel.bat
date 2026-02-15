@echo off
REM Vercel Deployment Script for Windows CMD

echo ========================================
echo   Deploying Frontend to Vercel
echo ========================================
echo.

cd /d "C:\Users\sruja\Classroom\SysntheticDataPCG\pcg_frontend"

echo [1/4] Checking Vercel CLI...
vercel --version >nul 2>&1
if errorlevel 1 (
    echo Installing Vercel CLI...
    npm install -g vercel
)

echo.
echo [2/4] Checking login status...
vercel whoami

echo.
echo [3/4] Deploying to Vercel (Preview)...
echo This may take 2-5 minutes...
echo.

vercel --yes

echo.
echo [4/4] Deployment Complete!
echo.
echo To deploy to production, run:
echo   vercel --prod
echo.
echo Don't forget to set environment variables in Vercel Dashboard!
echo.

pause
