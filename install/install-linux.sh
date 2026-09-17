#!/usr/bin/env bash
# Installs or updates the ComfyUI-IntoTheLatent-Utils node pack in an existing ComfyUI.
#
# Needs no root and no git: without git the pack is downloaded as a tarball from GitHub
# (that needs curl or wget, and tar).
#
#   1. Finds your ComfyUI folder (or asks for it) and checks that it really is ComfyUI.
#   2. Finds the Python that ComfyUI uses (venv, .venv, or the active venv / conda env).
#      It never installs into the system Python.
#   3. Clones the pack into custom_nodes, or updates the copy that is already there.
#   4. Installs the pack's Python requirements and checks that your torch was not replaced.
#
# Usage:
#   bash install-linux.sh                         # run it from inside your ComfyUI folder
#   bash install-linux.sh --comfy ~/ComfyUI
#   bash install-linux.sh --comfy ~/ComfyUI --python ~/miniconda3/envs/comfy/bin/python
#
# Options:
#   --comfy PATH    ComfyUI folder: the one that contains main.py and custom_nodes
#   --python PATH   the python ComfyUI runs with (only needed if it is not found)
#   --branch NAME   git branch to install (default: main)
#   -h, --help      show this text

set -u

PACK_NAME="ComfyUI-IntoTheLatent-Utils"
REGISTRY_NAME="comfyui-intothelatent-utils"   # folder name ComfyUI Manager uses
REPO_URL="https://github.com/Into-The-Latent/ComfyUI-IntoTheLatent-Utils"

COMFY_PATH=""
PYTHON_PATH=""
BRANCH="main"

if [ -t 1 ]; then C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_OFF=$'\033[0m'
else C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_OFF=""; fi

step() { printf '\n%s== %s%s\n' "$C_CYAN" "$1" "$C_OFF"; }
info() { printf '   %s\n' "$1"; }
good() { printf '   %s%s%s\n' "$C_GREEN" "$1" "$C_OFF"; }
warn() { printf '   %sWARNING: %s%s\n' "$C_YELLOW" "$1" "$C_OFF"; }
die()  { printf '\n%sFAILED: %s%s\n' "$C_RED" "$1" "$C_OFF" >&2
         echo "Fix the problem above and run the script again. Running it more than once is safe." >&2; exit 1; }
usage() { sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --comfy)  [ $# -ge 2 ] || die "--comfy needs a path";  COMFY_PATH="$2"; shift 2 ;;
        --python) [ $# -ge 2 ] || die "--python needs a path"; PYTHON_PATH="$2"; shift 2 ;;
        --branch) [ $# -ge 2 ] || die "--branch needs a name"; BRANCH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

# 'source'  = git clone: main.py + comfy/ + custom_nodes/
# 'desktop' = base folder of the ComfyUI Desktop app: custom_nodes/ + models/ + .venv/, no main.py
comfy_kind() {
    local p="$1"
    [ -n "$p" ] && [ -d "$p/custom_nodes" ] || return 1
    if [ -f "$p/main.py" ] && [ -d "$p/comfy" ]; then echo source; return 0; fi
    if [ -d "$p/models" ] && [ -d "$p/.venv" ]; then echo desktop; return 0; fi
    return 1
}

# Accepts the ComfyUI folder itself, or a folder that contains it.
resolve_comfy() {
    local p="${1%/}"
    [ -n "$p" ] || return 1
    p="${p/#\~/$HOME}"
    if comfy_kind "$p" >/dev/null; then (cd "$p" && pwd); return 0; fi
    if comfy_kind "$p/ComfyUI" >/dev/null; then (cd "$p/ComfyUI" && pwd); return 0; fi
    return 1
}

find_comfy() {
    local found start dir i answer
    if [ -n "$COMFY_PATH" ]; then
        resolve_comfy "$COMFY_PATH" && return 0
        die "'$COMFY_PATH' is not a ComfyUI folder. It must contain main.py and a custom_nodes folder."
    fi
    # The script's folder and the current folder, and up to four levels above each.
    for start in "$(cd "$(dirname "$0")" && pwd)" "$PWD"; do
        dir="$start"
        for i in 1 2 3 4 5; do
            if found="$(resolve_comfy "$dir")"; then echo "$found"; return 0; fi
            [ "$dir" = "/" ] && break
            dir="$(dirname "$dir")"
        done
    done
    [ -t 0 ] || die "ComfyUI was not found. Run the script from inside your ComfyUI folder or pass --comfy PATH."
    {
        info "ComfyUI was not found next to this script."
        info "Type the path of your ComfyUI folder (the one with main.py and custom_nodes)."
    } >&2
    for i in 1 2 3; do
        read -r -p "   ComfyUI folder: " answer || break
        if found="$(resolve_comfy "$answer")"; then echo "$found"; return 0; fi
        warn "That folder does not look like ComfyUI (no main.py + custom_nodes)." >&2
    done
    die "No ComfyUI folder given."
}

python_works() { [ -n "$1" ] && [ -x "$1" ] && "$1" -c 'import sys' >/dev/null 2>&1; }
has_torch()    { "$1" -s -c 'import torch' >/dev/null 2>&1; }

find_python() {
    local root="$1" parent c i answer
    parent="$(dirname "$root")"
    if [ -n "$PYTHON_PATH" ]; then
        PYTHON_PATH="${PYTHON_PATH/#\~/$HOME}"
        python_works "$PYTHON_PATH" && { echo "$PYTHON_PATH"; return 0; }
        die "'$PYTHON_PATH' is not a working python."
    fi
    for c in "$root/venv/bin/python" "$root/.venv/bin/python" "$parent/venv/bin/python" "$parent/.venv/bin/python"; do
        python_works "$c" && { echo "$c"; return 0; }
    done
    # An activated venv or conda env counts only if it has torch: that is the sign it is ComfyUI's.
    for c in "${VIRTUAL_ENV:-}/bin/python" "${CONDA_PREFIX:-}/bin/python"; do
        if [ "$c" != "/bin/python" ] && python_works "$c" && has_torch "$c"; then echo "$c"; return 0; fi
    done
    [ -t 0 ] || die "Could not find the Python that ComfyUI uses. Pass it with --python PATH."
    {
        warn "Could not find the Python that ComfyUI uses (no venv or .venv, no active env with torch)."
        info "This script never installs into your system Python: ComfyUI would not see the packages,"
        info "and most distributions forbid it. Type the path of the python that starts your ComfyUI."
    } >&2
    for i in 1 2 3; do
        read -r -p "   python: " answer || break
        answer="${answer/#\~/$HOME}"
        python_works "$answer" && { echo "$answer"; return 0; }
        warn "That is not a working python." >&2
    done
    die "No Python given."
}

torch_info() { "$1" -s -c 'import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())' 2>/dev/null; }

find_existing_pack() {
    local nodes="$1" d name
    for d in "$nodes"/*/; do
        [ -d "$d" ] || continue
        name="$(basename "$d" | tr '[:upper:]' '[:lower:]')"
        if [ "$name" = "$REGISTRY_NAME" ]; then echo "${d%/}"; return 0; fi
    done
    return 1
}

# Copies over the existing files and deletes nothing, so files the pack wrote at runtime
# (its hash cache) survive an update.
install_from_tarball() {
    local target="$1" url work
    url="$REPO_URL/archive/refs/heads/$BRANCH.tar.gz"
    command -v tar >/dev/null 2>&1 || die "tar is not installed (needed to unpack the download)."
    work="$(mktemp -d)" || die "Could not create a temporary folder."
    info "Downloading $url"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$work/pack.tar.gz" || { rm -rf "$work"; die "Download failed. Check your internet connection."; }
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$work/pack.tar.gz" || { rm -rf "$work"; die "Download failed. Check your internet connection."; }
    else
        rm -rf "$work"
        die "Neither git, curl nor wget is installed. Install one of them, for example: sudo apt install git"
    fi
    mkdir -p "$work/x" && tar -xzf "$work/pack.tar.gz" -C "$work/x" --strip-components=1 \
        || { rm -rf "$work"; die "Could not unpack the download."; }
    [ -f "$work/x/requirements.txt" ] || { rm -rf "$work"; die "The download does not look like the node pack (no requirements.txt)."; }
    mkdir -p "$target" && cp -a "$work/x/." "$target/" \
        || { rm -rf "$work"; die "Could not copy the files to '$target'."; }
    rm -rf "$work"
}

# --------------------------------------------------------------------------------------------------

printf '\n%sInto The Latent - %s installer%s\n' "$C_CYAN" "$PACK_NAME" "$C_OFF"
[ "$(id -u)" -eq 0 ] && warn "You are running as root. Files in custom_nodes will belong to root; run it as your normal user unless ComfyUI itself runs as root."

step "Looking for ComfyUI"
ROOT="$(find_comfy)" || exit 1
KIND="$(comfy_kind "$ROOT")"
good "ComfyUI: $ROOT"
[ "$KIND" = "desktop" ] && info "This is the base folder of the ComfyUI Desktop app."
NODES="$ROOT/custom_nodes"
[ -w "$NODES" ] || die "No write permission for $NODES. Run the script as the user that owns ComfyUI."

step "Looking for the Python that ComfyUI uses"
PY="$(find_python "$ROOT")" || exit 1
# --python may be relative, and pip is run from another folder later: make it absolute, without
# resolving the symlink (a venv's python is a symlink, and following it would leave the venv).
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
good "Python:  $PY ($("$PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null))"
TORCH_BEFORE="$(torch_info "$PY")"
if [ -n "$TORCH_BEFORE" ]; then
    info "torch:   $TORCH_BEFORE   (version, CUDA, CUDA available)"
else
    warn "torch cannot be imported with this Python. Is this really the Python ComfyUI runs with?"
    info "Continuing. The Breeze TTS and Whisper nodes need torch 2.7 or newer."
fi

step "Installing the node pack"
HAVE_GIT=0; command -v git >/dev/null 2>&1 && HAVE_GIT=1
if [ -d "$NODES/.disabled" ] && find_existing_pack "$NODES/.disabled" >/dev/null; then
    warn "A disabled copy of the pack exists in custom_nodes/.disabled (ComfyUI Manager). It is left alone."
fi

if TARGET="$(find_existing_pack "$NODES")"; then
    if [ -d "$TARGET/.git" ] && [ "$HAVE_GIT" -eq 1 ]; then
        info "Found a git copy, updating: $TARGET"
        git -C "$TARGET" pull --ff-only \
            || warn "git pull failed (local changes, or a different branch is checked out). The files were left as they are."
    else
        if [ -d "$TARGET/.git" ]; then warn "This copy was cloned with git, but git is not installed. Updating it from a download instead."
        else info "Found a copy without git (ComfyUI Manager or archive), updating it from a download: $TARGET"; fi
        install_from_tarball "$TARGET"
    fi
else
    TARGET="$NODES/$PACK_NAME"
    CLONED=0
    if [ "$HAVE_GIT" -eq 1 ]; then
        info "git clone -> $TARGET"
        if git clone --depth 1 --branch "$BRANCH" "$REPO_URL.git" "$TARGET"; then CLONED=1
        else warn "git clone failed, falling back to a download."; rm -rf "$TARGET"; fi
    else
        info "git is not installed. That is fine: downloading an archive instead."
    fi
    [ "$CLONED" -eq 1 ] || install_from_tarball "$TARGET"
fi

[ -f "$TARGET/requirements.txt" ] || die "No requirements.txt in '$TARGET': the pack was not installed completely."
good "Pack:    $TARGET"

step "Installing Python requirements"
# Python puts the current folder on its module path and pip lists it. Run from a neutral folder so
# that a folder that cannot be listed, or stray .py files next to the script, cannot break pip.
cd "${TMPDIR:-/tmp}" || die "Cannot enter the temporary folder."
if ! "$PY" -s -m pip --version >/dev/null 2>&1; then
    info "pip is missing in this Python, trying to add it (ensurepip)."
    "$PY" -s -m ensurepip --upgrade || die "This Python has no pip and ensurepip failed. Debian/Ubuntu: sudo apt install python3-venv python3-pip"
fi
"$PY" -s -m pip install -r "$TARGET/requirements.txt" || die "pip install failed. Read the pip messages above."

step "Checking the result"
if "$PY" -s -c 'import PIL, numpy, soundfile, librosa, transformers, huggingface_hub' >/dev/null 2>&1; then
    good "All required Python packages can be imported."
else
    warn "Some required packages cannot be imported. Scroll up for pip errors."
    info "If soundfile fails: it needs the system library libsndfile (Debian/Ubuntu: sudo apt install libsndfile1)."
fi

TORCH_AFTER="$(torch_info "$PY")"
if [ -n "$TORCH_BEFORE" ] && [ "$TORCH_AFTER" != "$TORCH_BEFORE" ]; then
    printf '\n%s   !! Your torch installation changed during the install !!\n' "$C_RED"
    printf '      before: %s\n      after:  %s\n' "$TORCH_BEFORE" "$TORCH_AFTER"
    printf '      This should never happen with this pack. Reinstall your CUDA / ROCm torch build if ComfyUI no longer finds your GPU.%s\n' "$C_OFF"
elif [ -n "$TORCH_BEFORE" ]; then
    good "torch is unchanged: $TORCH_AFTER"
fi

printf '\n%sDone. Restart ComfyUI to load the nodes.%s\n' "$C_GREEN" "$C_OFF"
echo "The Breeze TTS and Whisper models are downloaded the first time you use those nodes."
