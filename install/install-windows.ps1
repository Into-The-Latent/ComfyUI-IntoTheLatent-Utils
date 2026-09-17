<#
.SYNOPSIS
    Installs or updates the ComfyUI-IntoTheLatent-Utils node pack in an existing ComfyUI.

.DESCRIPTION
    Works on Windows PowerShell 5.1 and PowerShell 7. Needs no admin rights and no git:
    without git the pack is downloaded as a zip from GitHub.

    1. Finds your ComfyUI folder (or asks for it) and checks that it really is ComfyUI.
    2. Finds the Python that ComfyUI uses (portable python_embeded, venv or .venv).
       It never installs into the system Python.
    3. Clones the pack into custom_nodes, or updates the copy that is already there.
    4. Installs the pack's Python requirements and checks that your torch was not replaced.

    Easiest use: put this file (and install-windows.bat) into your ComfyUI folder and
    double click install-windows.bat.

.PARAMETER ComfyPath
    ComfyUI folder: the one that contains main.py and custom_nodes. For the portable build
    that is ComfyUI_windows_portable\ComfyUI (the portable root folder is accepted too).

.PARAMETER PythonPath
    Full path to the python.exe ComfyUI runs with. Only needed if it is not found.

.PARAMETER Branch
    Git branch to install. Default: main.

.PARAMETER NoPause
    Do not wait for Enter at the end (for unattended use).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install-windows.ps1 -ComfyPath "D:\AI\ComfyUI"
#>
[CmdletBinding()]
param(
    [string]$ComfyPath,
    [string]$PythonPath,
    [string]$Branch = 'main',
    [switch]$NoPause
)

# 'Stop' would turn the progress text git and pip write to stderr into fatal errors on
# PowerShell 5.1. Native commands are checked through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is many times faster without the bar
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

$PackName = 'ComfyUI-IntoTheLatent-Utils'
$RegistryName = 'comfyui-intothelatent-utils'     # folder name ComfyUI Manager uses
$RepoUrl = 'https://github.com/Into-The-Latent/ComfyUI-IntoTheLatent-Utils'

function Write-Step([string]$Text) { Write-Host ''; Write-Host "== $Text" -ForegroundColor Cyan }
function Write-Info([string]$Text) { Write-Host "   $Text" }
function Write-Good([string]$Text) { Write-Host "   $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "   WARNING: $Text" -ForegroundColor Yellow }

function Stop-Install([string]$Text) {
    Write-Host ''
    Write-Host "FAILED: $Text" -ForegroundColor Red
    Write-Host 'Fix the problem above and run the script again. Running it more than once is safe.'
    if (-not $NoPause) { [void](Read-Host 'Press Enter to close') }
    exit 1
}

function Get-ComfyKind([string]$Path) {
    # 'source'  = git clone or portable build: main.py + comfy\ + custom_nodes\
    # 'desktop' = base folder of the ComfyUI Desktop app: custom_nodes\ + models\ + .venv\, no main.py
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) { return $null }
    $hasNodes = Test-Path -LiteralPath (Join-Path $Path 'custom_nodes') -PathType Container
    if (-not $hasNodes) { return $null }
    $hasMain = Test-Path -LiteralPath (Join-Path $Path 'main.py') -PathType Leaf
    $hasComfy = Test-Path -LiteralPath (Join-Path $Path 'comfy') -PathType Container
    if ($hasMain -and $hasComfy) { return 'source' }
    $hasModels = Test-Path -LiteralPath (Join-Path $Path 'models') -PathType Container
    $hasVenv = Test-Path -LiteralPath (Join-Path $Path '.venv') -PathType Container
    if ($hasModels -and $hasVenv) { return 'desktop' }
    return $null
}

function Resolve-ComfyCandidate([string]$Path) {
    # Accepts the ComfyUI folder itself, or a folder that contains it (portable root).
    if (-not $Path) { return $null }
    $Path = $Path.Trim().Trim('"').Trim("'")
    if (-not $Path) { return $null }
    if (Get-ComfyKind $Path) { return (Resolve-Path -LiteralPath $Path).Path }
    $child = Join-Path $Path 'ComfyUI'
    if (Get-ComfyKind $child) { return (Resolve-Path -LiteralPath $child).Path }
    return $null
}

function Find-ComfyRoot {
    if ($ComfyPath) {
        $found = Resolve-ComfyCandidate $ComfyPath
        if ($found) { return $found }
        Stop-Install "'$ComfyPath' is not a ComfyUI folder. It must contain main.py and a custom_nodes folder."
    }
    # Look at the script's folder and the current folder, and up to four levels above each, so
    # the script works from ComfyUI\, ComfyUI\custom_nodes\ or the portable root.
    $starts = @()
    if ($PSScriptRoot) { $starts += $PSScriptRoot }
    $starts += (Get-Location).Path
    foreach ($start in $starts) {
        $dir = $start
        for ($i = 0; $i -lt 5 -and $dir; $i++) {
            $found = Resolve-ComfyCandidate $dir
            if ($found) { return $found }
            $dir = Split-Path -Parent $dir
        }
    }
    Write-Info 'ComfyUI was not found next to this script.'
    Write-Info 'Type or paste the path of your ComfyUI folder (the one with main.py and custom_nodes).'
    Write-Info 'Portable build: ...\ComfyUI_windows_portable\ComfyUI'
    for ($try = 0; $try -lt 3; $try++) {
        $answer = Read-Host '   ComfyUI folder'
        $found = Resolve-ComfyCandidate $answer
        if ($found) { return $found }
        Write-Warn 'That folder does not look like ComfyUI (no main.py + custom_nodes).'
    }
    Stop-Install 'No ComfyUI folder given.'
}

function Test-Python([string]$Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe -PathType Leaf)) { return $false }
    $null = & $Exe -c 'import sys' 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Find-Python([string]$Root) {
    if ($PythonPath) {
        $p = $PythonPath.Trim().Trim('"')
        if (Test-Python $p) { return (Resolve-Path -LiteralPath $p).Path }
        Stop-Install "'$PythonPath' is not a working python.exe."
    }
    $parent = Split-Path -Parent $Root
    $candidates = @(
        (Join-Path $parent 'python_embeded\python.exe'),   # portable build (yes, it is spelled "embeded")
        (Join-Path $Root 'venv\Scripts\python.exe'),
        (Join-Path $Root '.venv\Scripts\python.exe'),
        (Join-Path $parent 'venv\Scripts\python.exe'),
        (Join-Path $parent '.venv\Scripts\python.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Python $c) { return (Resolve-Path -LiteralPath $c).Path }
    }
    Write-Warn 'Could not find the Python that ComfyUI uses (no python_embeded, venv or .venv).'
    Write-Info 'This script never installs into your system Python, because ComfyUI would not see the packages.'
    Write-Info 'Paste the full path to the python.exe that starts your ComfyUI (conda users: the env''s python.exe).'
    for ($try = 0; $try -lt 3; $try++) {
        $answer = (Read-Host '   python.exe').Trim().Trim('"')
        if (Test-Python $answer) { return (Resolve-Path -LiteralPath $answer).Path }
        Write-Warn 'That is not a working python.exe.'
    }
    Stop-Install 'No Python given.'
}

function Get-TorchInfo([string]$Exe) {
    # No quotes inside the Python code: Windows PowerShell 5.1 mangles them for native commands.
    $out = & $Exe -s -c 'import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    return ("$out").Trim()
}

function Find-ExistingPack([string]$NodesDir) {
    foreach ($dir in (Get-ChildItem -LiteralPath $NodesDir -Directory -ErrorAction SilentlyContinue)) {
        if ($dir.Name -ieq $PackName -or $dir.Name -ieq $RegistryName) { return $dir.FullName }
    }
    return $null
}

function Install-FromZip([string]$Target) {
    # Copies over the existing files and deletes nothing, so files the pack wrote at runtime
    # (its hash cache) survive an update.
    $zipUrl = "$RepoUrl/archive/refs/heads/$Branch.zip"
    $work = Join-Path ([IO.Path]::GetTempPath()) ("itl-utils-" + [Guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $work -Force
    try {
        $zip = Join-Path $work 'pack.zip'
        Write-Info "Downloading $zipUrl"
        try { Invoke-WebRequest -Uri $zipUrl -OutFile $zip -UseBasicParsing -ErrorAction Stop }
        catch { Stop-Install "Download failed: $($_.Exception.Message). Check your internet connection." }
        try { Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $work 'x') -Force -ErrorAction Stop }
        catch { Stop-Install "Could not unpack the download: $($_.Exception.Message)" }
        $inner = Get-ChildItem -LiteralPath (Join-Path $work 'x') -Directory | Select-Object -First 1
        if (-not $inner -or -not (Test-Path -LiteralPath (Join-Path $inner.FullName 'requirements.txt'))) {
            Stop-Install 'The download does not look like the node pack (no requirements.txt).'
        }
        $null = New-Item -ItemType Directory -Path $Target -Force
        try { Copy-Item -Path (Join-Path $inner.FullName '*') -Destination $Target -Recurse -Force -ErrorAction Stop }
        catch { Stop-Install "Could not copy the files to '$Target': $($_.Exception.Message). Is ComfyUI still running?" }
    }
    finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ------------------------------------------------------------------------------------------------

Write-Host ''
Write-Host "Into The Latent - $PackName installer" -ForegroundColor Cyan
Write-Host 'Close ComfyUI before you continue: Windows cannot replace files that are in use.'

Write-Step 'Looking for ComfyUI'
$root = Find-ComfyRoot
$kind = Get-ComfyKind $root
Write-Good "ComfyUI: $root"
if ($kind -eq 'desktop') { Write-Info 'This is the base folder of the ComfyUI Desktop app.' }
$nodesDir = Join-Path $root 'custom_nodes'

Write-Step 'Looking for the Python that ComfyUI uses'
$python = Find-Python $root
$pyVersion = (& $python -c 'import sys;print(sys.version.split()[0])' 2>$null)
Write-Good "Python:  $python ($pyVersion)"
$torchBefore = Get-TorchInfo $python
if ($torchBefore) {
    Write-Info "torch:   $torchBefore   (version, CUDA, CUDA available)"
}
else {
    Write-Warn 'torch cannot be imported with this Python. Is this really the Python ComfyUI runs with?'
    Write-Info 'Continuing. The Breeze TTS and Whisper nodes need torch 2.7 or newer.'
}

Write-Step 'Installing the node pack'
$git = Get-Command git -ErrorAction SilentlyContinue
$existing = Find-ExistingPack $nodesDir
$disabled = Join-Path $nodesDir '.disabled'
if (Test-Path -LiteralPath $disabled) {
    if (Find-ExistingPack $disabled) { Write-Warn 'A disabled copy of the pack exists in custom_nodes\.disabled (ComfyUI Manager). It is left alone.' }
}

if ($existing) {
    $target = $existing
    $isGitCopy = Test-Path -LiteralPath (Join-Path $target '.git')
    if ($isGitCopy -and $git) {
        Write-Info "Found a git copy, updating: $target"
        & git -C $target pull --ff-only
        if ($LASTEXITCODE -ne 0) {
            Write-Warn 'git pull failed (local changes, or a different branch is checked out). The files were left as they are.'
        }
    }
    else {
        if ($isGitCopy) { Write-Warn 'This copy was cloned with git, but git is not installed. Updating it from a zip download instead.' }
        else { Write-Info "Found a copy without git (ComfyUI Manager or zip), updating it from a zip download: $target" }
        Install-FromZip $target
    }
}
else {
    $target = Join-Path $nodesDir $PackName
    $cloned = $false
    if ($git) {
        Write-Info "git clone -> $target"
        # core.longpaths: without it the clone dies with "Filename too long" when ComfyUI sits in a
        # deep folder (Windows 260 character limit).
        & git -c core.longpaths=true clone --depth 1 --branch $Branch "$RepoUrl.git" $target
        if ($LASTEXITCODE -eq 0) { $cloned = $true }
        else {
            Write-Warn 'git clone failed, falling back to a zip download.'
            if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }
    else {
        Write-Info 'git is not installed. That is fine: downloading a zip instead.'
        Write-Info 'Tip: with git (winget install Git.Git) later updates are faster.'
    }
    if (-not $cloned) { Install-FromZip $target }
}

$requirements = Join-Path $target 'requirements.txt'
if (-not (Test-Path -LiteralPath $requirements)) { Stop-Install "No requirements.txt in '$target': the pack was not installed completely." }
Write-Good "Pack:    $target"

Write-Step 'Installing Python requirements'
# Python puts the current folder on its module path and pip lists it. Run from a neutral folder so
# that a folder that cannot be listed, or stray .py files next to the script, cannot break pip.
Set-Location -LiteralPath ([IO.Path]::GetTempPath())
$null = & $python -s -m pip --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Info 'pip is missing in this Python, trying to add it (ensurepip).'
    & $python -s -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { Stop-Install 'This Python has no pip and ensurepip failed.' }
}
# -s: ignore packages in the per-user site folder, the same way the portable build starts ComfyUI.
& $python -s -m pip install -r $requirements --no-warn-script-location
if ($LASTEXITCODE -ne 0) { Stop-Install 'pip install failed. Read the pip messages above.' }

Write-Step 'Checking the result'
$null = & $python -s -c 'import PIL, numpy, soundfile, librosa, transformers, huggingface_hub' 2>$null
if ($LASTEXITCODE -eq 0) { Write-Good 'All required Python packages can be imported.' }
else { Write-Warn 'Some required packages cannot be imported. Scroll up for pip errors.' }

$torchAfter = Get-TorchInfo $python
if ($torchBefore -and $torchAfter -ne $torchBefore) {
    Write-Host ''
    Write-Host '   !! Your torch installation changed during the install !!' -ForegroundColor Red
    Write-Host "      before: $torchBefore" -ForegroundColor Red
    Write-Host "      after:  $torchAfter" -ForegroundColor Red
    Write-Host '      This should never happen with this pack. If ComfyUI now reports' -ForegroundColor Red
    Write-Host '      "Torch not compiled with CUDA enabled", reinstall your CUDA torch build.' -ForegroundColor Red
}
elseif ($torchBefore) { Write-Good "torch is unchanged: $torchAfter" }

Write-Host ''
Write-Host 'Done. Restart ComfyUI to load the nodes.' -ForegroundColor Green
Write-Host 'The Breeze TTS and Whisper models are downloaded the first time you use those nodes.'
if (-not $NoPause) { [void](Read-Host 'Press Enter to close') }
exit 0
