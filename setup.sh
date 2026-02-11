#!/bin/bash

echo "====================================="
echo "Transcriptor Setup Script"
echo "====================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies (this may take several minutes)..."
pip install -r requirements.txt

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY (if using Whisper API)"
echo "2. Edit .env file and add your YOUTUBE_CHANNEL_URL"
echo "3. Get a Hugging Face token at: https://huggingface.co/settings/tokens"
echo "4. Accept the pyannote user agreement at: https://huggingface.co/pyannote/speaker-diarization"
echo "5. Add HF_TOKEN to your .env file"
echo ""
echo "Then run:"
echo "  source venv/bin/activate"
echo "  python create_voice_profile.py"
echo "  python transcribe_channel.py"
echo ""
