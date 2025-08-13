#!/bin/bash
# Simple cURL test for RunPod endpoint

# Configuration - REPLACE THESE VALUES
RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY_HERE"
ENDPOINT_ID="YOUR_ENDPOINT_ID_HERE"

# Test URLs
VIDEO_URL="https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
AUDIO_URL="https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"

if [ "$RUNPOD_API_KEY" = "YOUR_RUNPOD_API_KEY_HERE" ] || [ "$ENDPOINT_ID" = "YOUR_ENDPOINT_ID_HERE" ]; then
    echo "❌ Please set RUNPOD_API_KEY and ENDPOINT_ID in this script"
    echo ""
    echo "Get them from:"
    echo "- API Key: RunPod Settings → API Keys"  
    echo "- Endpoint ID: RunPod Endpoints → Your endpoint"
    exit 1
fi

echo "🧪 Testing RunPod VantageVeoConverter endpoint..."
echo "📋 Endpoint: $ENDPOINT_ID"

# Submit job
echo ""
echo "📤 Submitting test job..."

RESPONSE=$(curl -s -X POST \
  "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{
    \"input\": {
      \"video_url\": \"$VIDEO_URL\",
      \"audio_url\": \"$AUDIO_URL\",
      \"use_rife\": false,
      \"diagnostic_mode\": false
    }
  }")

echo "Response: $RESPONSE"

# Extract job ID
JOB_ID=$(echo $RESPONSE | grep -o '"id":"[^"]*' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job. Check your API key and endpoint ID."
    echo "Response: $RESPONSE"
    exit 1
fi

echo "✅ Job submitted with ID: $JOB_ID"

# Poll for results
echo ""
echo "⏳ Polling for results..."

for i in {1..60}; do  # Poll for up to 5 minutes (60 * 5s = 300s)
    STATUS_RESPONSE=$(curl -s \
      "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY")
    
    STATUS=$(echo $STATUS_RESPONSE | grep -o '"status":"[^"]*' | cut -d'"' -f4)
    
    echo "[$i/60] Status: $STATUS"
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "✅ Job completed successfully!"
        echo "📊 Full response:"
        echo $STATUS_RESPONSE | jq '.' 2>/dev/null || echo $STATUS_RESPONSE
        exit 0
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "❌ Job failed!"
        echo "📊 Error response:"
        echo $STATUS_RESPONSE | jq '.' 2>/dev/null || echo $STATUS_RESPONSE
        exit 1
    fi
    
    sleep 5
done

echo ""
echo "⏰ Timeout after 5 minutes"
echo "📊 Last response:"
echo $STATUS_RESPONSE | jq '.' 2>/dev/null || echo $STATUS_RESPONSE
exit 1