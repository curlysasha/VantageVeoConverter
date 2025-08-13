#!/bin/bash

# VantageVeoConverter RunPod Deployment Script
# Автоматическая сборка и деплой на RunPod Serverless

set -e  # Exit on any error

echo "🚀 VantageVeoConverter RunPod Deployment Script"
echo "=================================================="

# Configuration
DOCKER_USERNAME=${DOCKER_USERNAME:-""}
IMAGE_NAME="vantage-veo"
VERSION=${VERSION:-"v1.0.0"}
PLATFORM="linux/amd64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check requirements
check_requirements() {
    info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    success "Docker found"
    
    # Check Docker login
    if ! docker info &> /dev/null; then
        error "Docker is not running or not accessible"
    fi
    success "Docker is running"
    
    # Check if we're in the right directory
    if [[ ! -f "runpod_handler.py" ]]; then
        error "runpod_handler.py not found. Make sure you're in the VantageVeoConverter directory"
    fi
    
    if [[ ! -f "Dockerfile.runpod" ]]; then
        error "Dockerfile.runpod not found. Make sure you're in the VantageVeoConverter directory"
    fi
    
    success "All required files found"
}

# Get Docker username
get_docker_username() {
    if [[ -z "$DOCKER_USERNAME" ]]; then
        read -p "Enter your Docker Hub username: " DOCKER_USERNAME
    fi
    
    if [[ -z "$DOCKER_USERNAME" ]]; then
        error "Docker username is required"
    fi
    
    info "Using Docker username: $DOCKER_USERNAME"
}

# Build Docker image
build_image() {
    local full_image_name="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    
    info "Building Docker image: $full_image_name"
    info "Platform: $PLATFORM"
    
    # Build the image
    if docker build \
        -f Dockerfile.runpod \
        -t "$full_image_name" \
        --platform "$PLATFORM" \
        --progress=plain \
        .; then
        success "Docker image built successfully"
    else
        error "Docker build failed"
    fi
    
    # Tag as latest
    docker tag "$full_image_name" "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    success "Tagged as latest"
}

# Push to Docker Hub
push_image() {
    local full_image_name="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    
    info "Pushing to Docker Hub..."
    
    # Login check
    if ! docker info | grep -q "Username:"; then
        warn "Not logged in to Docker Hub"
        if ! docker login; then
            error "Docker login failed"
        fi
    fi
    
    # Push versioned image
    if docker push "$full_image_name"; then
        success "Pushed $full_image_name"
    else
        error "Failed to push $full_image_name"
    fi
    
    # Push latest
    if docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"; then
        success "Pushed ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    else
        warn "Failed to push latest tag"
    fi
}

# Generate deployment instructions
generate_instructions() {
    local full_image_name="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    
    cat << EOF

🎉 Deployment Complete!
======================

Your Docker image has been built and pushed successfully:
📦 Image: ${full_image_name}

Next steps:
1. Go to RunPod Console: https://www.runpod.io/console/serverless
2. Create New Endpoint:
   - Name: vantage-veo-converter
   - Docker Image: ${full_image_name}
   - GPU: RTX 4090 PRO (recommended) or RTX 3090 (cheapest)
   - Container Disk: 15 GB
   - Worker Timeout: 300 seconds
   - Max Workers: 3
   - Min Workers: 0

3. Test your endpoint:

curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "input": {
      "video_url": "https://example.com/video.mp4",
      "audio_url": "https://example.com/audio.wav",
      "use_rife": true,
      "diagnostic_mode": false
    }
  }'

📚 Full documentation: RUNPOD_DEPLOYMENT.md
💰 Estimated cost: $0.68-$1.12 per hour (depending on GPU)
⚡ Cold start time: 10-20 seconds
🔥 Warm requests: 2-5 seconds per minute of video

Happy deploying! 🚀

EOF
}

# Local test function
test_locally() {
    local full_image_name="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    
    info "Testing Docker image locally..."
    
    # Test basic container startup
    if docker run --rm "$full_image_name" python -c "import runpod; import torch; print('✅ Container test passed')"; then
        success "Container test passed"
    else
        error "Container test failed"
    fi
    
    # Test handler import
    if docker run --rm "$full_image_name" python -c "from runpod_handler import handler; print('✅ Handler import test passed')"; then
        success "Handler import test passed"
    else
        error "Handler import test failed"
    fi
}

# Main execution
main() {
    echo
    info "Starting deployment process..."
    
    check_requirements
    get_docker_username
    
    # Build
    echo
    info "Step 1/4: Building Docker image..."
    build_image
    
    # Test locally
    echo
    info "Step 2/4: Testing locally..."
    test_locally
    
    # Push
    echo
    info "Step 3/4: Pushing to Docker Hub..."
    push_image
    
    # Instructions
    echo
    info "Step 4/4: Generating deployment instructions..."
    generate_instructions
    
    success "Deployment script completed successfully!"
}

# Help function
show_help() {
    cat << EOF
VantageVeoConverter RunPod Deployment Script

Usage:
  $0 [OPTIONS]

Options:
  -h, --help          Show this help message
  -u, --user USER     Docker Hub username
  -v, --version VER   Version tag (default: v1.0.0)
  --test-only         Only test the image locally, don't push
  --build-only        Only build the image, don't push

Environment Variables:
  DOCKER_USERNAME     Docker Hub username
  VERSION            Version tag

Examples:
  $0 -u myuser -v v1.2.0
  DOCKER_USERNAME=myuser VERSION=latest $0
  $0 --test-only

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--user)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Execute based on flags
if [[ "$TEST_ONLY" == "true" ]]; then
    check_requirements
    get_docker_username
    build_image
    test_locally
    success "Test completed successfully!"
elif [[ "$BUILD_ONLY" == "true" ]]; then
    check_requirements
    get_docker_username
    build_image
    success "Build completed successfully!"
else
    main
fi