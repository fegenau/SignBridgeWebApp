# Quick Start Script for SignBridge Web Application
# This script helps you get started quickly

Write-Host "🤟 SignBridge Web Application - Quick Start" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
$expectedDir = "D:\Proyecto-capstone\Version-web\SignBridgeKeras\SignBridgeKeras\WebApp"

if ($currentDir.Path -ne $expectedDir) {
    Write-Host "⚠️  Changing to WebApp directory..." -ForegroundColor Yellow
    Set-Location $expectedDir
}

Write-Host "📍 Current directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".\venv") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  No virtual environment found" -ForegroundColor Yellow
    Write-Host "💡 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
    
    Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "🚀 Starting Streamlit application..." -ForegroundColor Cyan
Write-Host "📹 The app will open in your browser automatically" -ForegroundColor Yellow
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run Streamlit
streamlit run app.py
