#!/bin/bash
# Quick development commands for Docker

case "$1" in
    "start")
        echo "🚀 Starting Mini LLM Chat development server..."
        docker-compose -f docker-compose.dev.yml up
        ;;
    "stop")
        echo "🛑 Stopping development server..."
        docker-compose -f docker-compose.dev.yml down
        ;;
    "restart")
        echo "🔄 Restarting development server..."
        docker-compose -f docker-compose.dev.yml restart
        ;;
    "shell")
        echo "🐚 Opening shell in development container..."
        docker-compose -f docker-compose.dev.yml exec minillm-dev bash
        ;;
    "logs")
        echo "📋 Showing development server logs..."
        docker-compose -f docker-compose.dev.yml logs -f
        ;;
    "build")
        echo "🔨 Rebuilding development image..."
        docker-compose -f docker-compose.dev.yml build --no-cache
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose -f docker-compose.dev.yml down -v
        docker system prune -f
        ;;
    "test")
        echo "🧪 Running tests in container..."
        docker-compose -f docker-compose.dev.yml exec minillm-dev pytest
        ;;
    *)
        echo "🐳 Mini LLM Chat Docker Development Commands"
        echo ""
        echo "Usage: $0 {start|stop|restart|shell|logs|build|clean|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the development server"
        echo "  stop    - Stop the development server"
        echo "  restart - Restart the development server"
        echo "  shell   - Open a shell in the development container"
        echo "  logs    - Show and follow server logs"
        echo "  build   - Rebuild the development image"
        echo "  clean   - Clean up Docker resources"
        echo "  test    - Run tests in the container"
        echo ""
        echo "🌐 Web interface available at: http://localhost:8000"
        ;;
esac