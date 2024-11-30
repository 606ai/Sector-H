# PowerShell script for setting up Windsurf IDE and browser extensions

$ErrorActionPreference = "Stop"

# Configuration
$windsurfConfigDir = "$env:APPDATA\Windsurf"
$sectorHPath = "$PSScriptRoot\.."
$browserExtensions = @{
    "Chrome" = @(
        @{
            name = "Windsurf IDE"
            url = "https://chrome.google.com/webstore/detail/windsurf-ide/your-extension-id"
        }
        @{
            name = "GitHub Copilot"
            url = "https://chrome.google.com/webstore/detail/github-copilot/codebuild-your-extension-id"
        }
        @{
            name = "React Developer Tools"
            url = "https://chrome.google.com/webstore/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi"
        }
        @{
            name = "JSON Viewer"
            url = "https://chrome.google.com/webstore/detail/json-viewer/gbmdgpbipfallnflgajpaliibnhdgobh"
        }
        @{
            name = "Wappalyzer"
            url = "https://chrome.google.com/webstore/detail/wappalyzer-technology-pro/gppongmhjkpfnbhagpmjfkannfbllamg"
        }
    )
    "Edge" = @(
        @{
            name = "Windsurf IDE"
            url = "https://microsoftedge.microsoft.com/addons/detail/windsurf-ide/your-extension-id"
        }
        @{
            name = "GitHub Copilot"
            url = "https://microsoftedge.microsoft.com/addons/detail/github-copilot/your-extension-id"
        }
    )
}

Write-Host "🚀 Setting up Windsurf IDE and development environment..." -ForegroundColor Cyan

# Create Windsurf configuration directory if it doesn't exist
if (-not (Test-Path $windsurfConfigDir)) {
    New-Item -ItemType Directory -Path $windsurfConfigDir | Out-Null
}

# Configure Windsurf IDE settings
$windsurfSettings = @{
    "workspaces" = @(
        @{
            "name" = "Sector-H"
            "path" = $sectorHPath
            "conda_env" = "sector-h"
            "git_enabled" = $true
            "docker_enabled" = $true
        }
    )
    "editor" = @{
        "theme" = "dark"
        "font_family" = "Fira Code"
        "font_size" = 14
        "tab_size" = 4
        "auto_save" = $true
        "format_on_save" = $true
    }
    "ai_features" = @{
        "copilot_enabled" = $true
        "code_completion" = $true
        "code_analysis" = $true
    }
    "docker" = @{
        "compose_file" = "./docker-compose.yml"
        "auto_rebuild" = $true
    }
    "jupyter" = @{
        "default_kernel" = "sector-h"
        "auto_start" = $true
    }
} | ConvertTo-Json -Depth 10

$windsurfSettings | Out-File -FilePath "$windsurfConfigDir\settings.json"

# Install recommended VS Code extensions (as Windsurf IDE is based on VS Code)
$extensions = @(
    "ms-python.python"
    "ms-toolsai.jupyter"
    "ms-azuretools.vscode-docker"
    "dbaeumer.vscode-eslint"
    "esbenp.prettier-vscode"
    "GitHub.copilot"
    "eamodio.gitlens"
    "ms-vscode.powershell"
    "redhat.vscode-yaml"
    "ms-python.vscode-pylance"
    "ms-python.black-formatter"
    "ms-vscode.cmake-tools"
    "ms-vscode.cpptools"
    "twxs.cmake"
)

foreach ($extension in $extensions) {
    Write-Host "Installing VS Code extension: $extension" -ForegroundColor Green
    code --install-extension $extension
}

# Display browser extension links
Write-Host "`n📦 Recommended browser extensions:" -ForegroundColor Cyan
foreach ($browser in $browserExtensions.Keys) {
    Write-Host "`n$browser Extensions:" -ForegroundColor Yellow
    foreach ($extension in $browserExtensions[$browser]) {
        Write-Host "- $($extension.name): $($extension.url)"
    }
}

# Create .vscode settings for the project
$vscodePath = Join-Path $sectorHPath ".vscode"
if (-not (Test-Path $vscodePath)) {
    New-Item -ItemType Directory -Path $vscodePath | Out-Null
}

$vscodeSettings = @{
    "python.defaultInterpreterPath" = "conda:sector-h"
    "python.formatting.provider" = "black"
    "editor.formatOnSave" = $true
    "editor.codeActionsOnSave" = @{
        "source.organizeImports" = $true
    }
    "files.exclude" = @{
        "**/__pycache__" = $true
        "**/.pytest_cache" = $true
        "**/*.pyc" = $true
    }
    "jupyter.notebookFileRoot" = "${workspaceFolder}"
} | ConvertTo-Json -Depth 10

$vscodeSettings | Out-File -FilePath "$vscodePath\settings.json"

Write-Host @"

✅ Setup Complete!

🔧 Configured:
- Windsurf IDE workspace settings
- VS Code extensions
- Project-specific settings
- Jupyter integration
- Docker integration

📝 Next steps:
1. Install the recommended browser extensions using the links above
2. Restart Windsurf IDE to apply all settings
3. Open the Sector-H workspace in Windsurf IDE

💡 Additional recommendations:
- Install Windows Terminal for better terminal experience
- Consider installing PowerToys for enhanced productivity
- Set up WSL2 for better Docker performance

For any issues, please check the documentation or raise an issue on the project repository.
"@ -ForegroundColor Cyan
