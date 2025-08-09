# --- 1/3: Updating system packages and installing all dependencies ---
echo "--- 1/3: Updating system packages and installing dependencies (FFmpeg, eSpeak, Build Tools) ---"
# We are root, so we use apt-get directly. DEBIAN_FRONTEND=noninteractive prevents popups.
DEBIAN_FRONTEND=noninteractive apt-get update -y
# THE FIX: Added libespeak-dev to provide the legacy library files needed to compile aeneas.
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg espeak-ng libespeak-ng-dev libespeak-dev build-essential git python3-dev autoconf automake libtool pkg-config

# --- 2/3: Installing Python dependencies ---
echo "--- 2/3: Installing Python dependencies ---"

# Upgrade pip
pip install --upgrade pip

# CRITICAL FIX for Aeneas: Aeneas requires an older version of setuptools (<60)
# to compile correctly on newer Python versions.
echo "Downgrading setuptools to fix aeneas compatibility issue..."
pip install "setuptools==59.5.0"

# Now install the rest of the requirements. Aeneas should now succeed.
echo "Installing Aeneas, Whisper, Gradio, and other libraries..."
pip install aeneas numpy scipy opencv-python-headless openai-whisper gradio torch

# --- 3/3: Compiling and installing mp4fpsmod ---
echo "--- 3/3: Compiling and installing mp4fpsmod ---"
# Clean up previous failed attempt if it exists
rm -rf mp4fpsmod
git clone https://github.com/nu774/mp4fpsmod.git
cd mp4fpsmod

# Compile the tool
./bootstrap.sh
./configure
make

# Install the compiled binary (no sudo needed)
make install