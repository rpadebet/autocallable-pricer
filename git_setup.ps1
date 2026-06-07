# git_setup.ps1
# Run from Windows PowerShell in the project folder:
#   cd "C:\Users\rohit\OneDrive\Documents\Claude Apps\Autocallables"
#   .\git_setup.ps1
#
# This removes any broken .git directory, initializes fresh, and commits everything.

Set-Location "C:\Users\rohit\OneDrive\Documents\Claude Apps\Autocallables"

# Step 1: Remove broken .git if it exists
if (Test-Path ".git") {
    Write-Host "Removing existing (broken) .git directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".git"
}

# Step 2: Initialize fresh repo
git init
git config user.name "Rohit Padebettu"
git config user.email "rohit.pittu@gmail.com"
git branch -M main

# Step 3: Create .gitignore
$gitignore = @"
__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.egg-info/
.pytest_cache/
.streamlit/
*.log
pytest-cache-files-*
"@
$gitignore | Out-File -FilePath ".gitignore" -Encoding utf8 -NoNewline

# Step 4: Stage and commit - v0.1.0 (project structure + core modules)
git add .gitignore requirements.txt
git add scripts/ sample_data/
git add app/__init__.py app/autocallable.py app/data_loader.py app/heston.py
git add app/mc_standard.py app/mc_survival.py app/pde_pricer.py app/vol_surface.py
git add app/components/
git commit -m "feat: core engine modules — FD, MC, Survival MC, Heston, vol surface, autocallable"

# Step 5: Commit Streamlit pages - v0.2.0
git add app/Home.py
git add "app/pages/"
git add "app/components/sidebar.py" 2>$null
git commit -m "feat: Streamlit pages — Home, Pricer, Vol Surface, FDM Visualization"

# Step 6: Commit test suite and fixes - v0.3.0
git add tests/
git add test_results/
git add CHANGELOG.md NEXT_SESSION.md CLAUDE.md README.md 2>$null
git commit -m "fix: 66/66 tests passing — fix file truncations, validation logic, FD resolution"

# Step 7: Commit Greeks page and spot_override — v0.3.1
git add "app/pages/04_Greeks.py"
git add app/mc_standard.py app/mc_survival.py
git add CHANGELOG.md NEXT_SESSION.md CLAUDE.md
git commit -m "feat: Greeks page — Delta/Vega stability (Paper 3); spot_override for correct bump pricing"

Write-Host ""
Write-Host "=== Git setup complete ===" -ForegroundColor Green
Write-Host ""
git log --oneline
Write-Host ""
Write-Host "Next step — create GitHub repo and push:" -ForegroundColor Cyan
Write-Host "  1. Go to https://github.com/new"
Write-Host "  2. Create repo named: autocallable-pricer (public, no README)"
Write-Host "  3. Run these commands:"
Write-Host "     git remote add origin https://github.com/rohitpittu/autocallable-pricer.git"
Write-Host "     git push -u origin main"
Write-Host ""
Write-Host "Then deploy to Streamlit Cloud: https://share.streamlit.io"
