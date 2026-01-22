#!/bin/bash

# OCR Document Processing Pipeline Setup Script

echo "🚀 Setting up OCR Document Processing Pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Install Marker library
echo "🔍 Installing Marker library..."
if [ ! -d "marker" ]; then
    git clone https://github.com/VikParuchuri/marker.git
    cd marker
    pip install -e .
    cd ..
else
    echo "✅ Marker already exists"
fi

# Install Surya library
echo "🌅 Installing Surya library..."
if [ ! -d "surya" ]; then
    git clone https://github.com/VikParuchuri/surya.git
    cd surya
    pip install -e .
    cd ..
else
    echo "✅ Surya already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p .streamlit
mkdir -p /tmp/marker_debug

# Create Streamlit secrets template if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "🔑 Creating Streamlit secrets template..."
    cat > .streamlit/secrets.toml << EOF
GROQ_API_KEY = "your_groq_api_key_here"
EOF
    echo "⚠️ Please update .streamlit/secrets.toml with your actual Groq API key"
fi

# Check MongoDB installation
echo "🍃 Checking MongoDB..."
if ! command -v mongod &> /dev/null; then
    echo "⚠️ MongoDB not found. Please install MongoDB:"
    echo "   Ubuntu/Debian: sudo apt install mongodb"
    echo "   macOS: brew install mongodb-community"
    echo "   Or use Docker: docker run -d -p 27017:27017 mongo"
else
    echo "✅ MongoDB found"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .streamlit/secrets.toml with your Groq API key"
echo "2. Start MongoDB service"
echo "3. Run: ./start_app.sh"
