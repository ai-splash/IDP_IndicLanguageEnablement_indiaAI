#!/bin/bash

# OCR Document Processing Pipeline Start Script

echo "🚀 Starting OCR Document Processing Pipeline..."

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Check if MongoDB is running
echo "🍃 Checking MongoDB connection..."
if ! mongosh --eval "db.adminCommand('ping')" --quiet > /dev/null 2>&1; then
    echo "⚠️ MongoDB not running. Starting MongoDB..."
    
    # Try different MongoDB start methods
    if command -v systemctl &> /dev/null; then
        sudo systemctl start mongod
    elif command -v brew &> /dev/null; then
        brew services start mongodb-community
    else
        echo "❌ Please start MongoDB manually or use Docker:"
        echo "   Docker: docker run -d -p 27017:27017 mongo"
        exit 1
    fi
    
    # Wait for MongoDB to start
    sleep 3
fi

# Verify MongoDB is accessible
if mongosh --eval "db.adminCommand('ping')" --quiet > /dev/null 2>&1; then
    echo "✅ MongoDB is running"
else
    echo "❌ Cannot connect to MongoDB. Please check your MongoDB installation."
    exit 1
fi

# Check if Groq API key is set
if [ ! -f ".streamlit/secrets.toml" ] || ! grep -q "GROQ_API_KEY" .streamlit/secrets.toml; then
    echo "⚠️ Groq API key not configured. Please update .streamlit/secrets.toml"
fi

# Start the Streamlit application
echo "🌟 Starting Streamlit application..."
echo "📱 Access the app at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
