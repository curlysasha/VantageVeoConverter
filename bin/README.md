# Binary Dependencies

Place compiled binaries here for portable deployment.

## FFmpeg (required)

### Linux/WSL - Static builds:
```bash
# Download and extract static FFmpeg
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*-amd64-static/ffmpeg ./
cp ffmpeg-*-amd64-static/ffprobe ./
chmod +x ffmpeg ffprobe
```

### Windows:
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract `ffmpeg.exe` and `ffprobe.exe` to this directory

## mp4fpsmod (required)

### Pre-compiled binaries:
- **Linux/WSL**: `mp4fpsmod` (no extension)
- **Windows**: `mp4fpsmod.exe`

### Building from source:
```bash
git clone https://github.com/nu774/mp4fpsmod.git
cd mp4fpsmod
./bootstrap.sh
./configure
make
sudo make install
# Copy binary to this directory:
cp /usr/local/bin/mp4fpsmod ./
chmod +x mp4fpsmod
```

### Windows build:
1. Open `vcproj/mp4fpsmod.sln` in Visual Studio
2. Build -> Build Solution
3. Copy `Release/mp4fpsmod.exe` here

## Testing:
```bash
# Test FFmpeg
./ffmpeg -version
./ffprobe -version

# Test mp4fpsmod
./mp4fpsmod
# Should show: mp4fpsmod 0.28 (libmp4v2 2.1.2)
```

## Final structure:
```
bin/
├── ffmpeg      (or ffmpeg.exe)
├── ffprobe     (or ffprobe.exe)  
├── mp4fpsmod   (or mp4fpsmod.exe)
└── README.md
```

## Deployment Options:

### 1. Local bin/ directory (recommended):
Place binaries in `bin/` directory relative to project root

### 2. Environment variables:
```bash
export FFMPEG_PATH=/path/to/ffmpeg
export FFPROBE_PATH=/path/to/ffprobe  
export MP4FPSMOD_PATH=/path/to/mp4fpsmod
```

### 3. System PATH:
Install binaries system-wide so they're available in PATH

### 4. Docker deployment:
```dockerfile
FROM ubuntu:22.04
RUN apt update && apt install -y ffmpeg
COPY bin/mp4fpsmod /usr/local/bin/
COPY . /app
WORKDIR /app
CMD ["python", "app_rife_compact.py"]
```

The system will automatically search in this order:
1. Project bin/ directory (multiple possible locations)
2. Environment variables (FFMPEG_PATH, etc.)
3. System PATH
4. Shows detailed error if not found