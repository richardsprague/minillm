#!/bin/bash
# Development setup script for Docker environment

set -e

echo "🐳 Setting up Mini LLM Chat development environment with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p logs

# Build the development image
echo "🔨 Building development Docker image..."
docker-compose -f docker-compose.dev.yml build

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 To start developing:"
echo "  docker-compose -f docker-compose.dev.yml up"
echo ""
echo "📝 To run commands in the container:"
echo "  docker-compose -f docker-compose.dev.yml exec minillm-dev bash"
echo ""
echo "🌐 Web interface will be available at:"
echo "  http://localhost:8000"
echo ""
echo "💡 Your code changes will be reflected immediately without rebuilding!"