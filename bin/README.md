# Binary Dependencies

Place compiled mp4fpsmod binary here.

## mp4fpsmod

The project requires mp4fpsmod binary for VFR (Variable Frame Rate) video processing.

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
```

### Windows build:
1. Open `vcproj/mp4fpsmod.sln` in Visual Studio
2. Build -> Build Solution
3. Copy `Release/mp4fpsmod.exe` or `Debug/mp4fpsmod.exe` here

### Make executable (Linux/WSL only):
```bash
chmod +x mp4fpsmod
```

### Test:
```bash
./mp4fpsmod 
# Should show: mp4fpsmod 0.28 (libmp4v2 2.1.2)
```