#!/usr/bin/env python3

# Simple test server to check endpoints
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == '/models/available':
            response = [
                {"name": "MinillM-505M", "type": "local", "description": "Local model: MinillM-505M"},
                {"name": "microsoft/DialoGPT-small", "type": "huggingface", "description": "Huggingface model: microsoft/DialoGPT-small"},
                {"name": "gpt-3.5-turbo", "type": "openai", "description": "Openai model: gpt-3.5-turbo"}
            ]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/models/current':
            response = {
                "name": "MinillM-505M",
                "type": "local", 
                "loaded": True,
                "description": "Local model: MinillM-505M"
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/health':
            response = {
                "status": "healthy",
                "model_loaded": True,
                "config_loaded": True
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        elif path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            try:
                with open('web/static/index.html', 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"<h1>Mini LLM Chat Test Server</h1><p>Web interface files not found</p>")
        
        elif path.startswith('/static/'):
            file_path = path[1:]  # Remove leading /
            try:
                import os
                import mimetypes
                if os.path.exists(file_path):
                    mime_type, _ = mimetypes.guess_type(file_path)
                    self.send_response(200)
                    self.send_header('Content-Type', mime_type or 'application/octet-stream')
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            except Exception:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/models/upload':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            # Simulate successful upload
            response = {"success": True, "message": "Model uploaded successfully (demo mode)"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/models/switch':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            response = {"success": True, "message": "Model switched successfully"}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/chat':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            # Read the request content
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                import json
                request_data = json.loads(post_data.decode('utf-8'))
                user_message = request_data.get('messages', [])[-1].get('content', '') if request_data.get('messages') else ''
                
                # Generate contextual responses based on the question
                if "fastest animal" in user_message.lower():
                    test_response = "The fastest animal is the peregrine falcon, which can reach speeds of over 240 mph (386 km/h) when diving. On land, the cheetah is the fastest, reaching speeds up to 70 mph (112 km/h)."
                elif "strawberry" in user_message.lower() and "r" in user_message.lower():
                    test_response = "Let me count the Rs in 'strawberry': s-t-r-a-w-b-e-r-r-y. There are 3 Rs in the word 'strawberry' - one in the middle and two at the end."
                elif "han dynasty" in user_message.lower() and "roman" in user_message.lower():
                    test_response = "That's a fascinating hypothetical! Both were military powerhouses of their time. The Romans had superior siege warfare, engineering, and disciplined legions. The Han had crossbows, cavalry, and larger numbers. In a direct confrontation, it would likely depend on terrain, logistics, and leadership. Historically, they never fought as they were separated by thousands of miles and the Parthian Empire."
                elif "capital of france" in user_message.lower():
                    test_response = "The capital of France is Paris. It's located in the north-central part of the country and has been the capital since 508 AD."
                elif "quantum computing" in user_message.lower():
                    test_response = "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can be in multiple states simultaneously, potentially solving certain problems exponentially faster."
                elif "joke" in user_message.lower():
                    test_response = "Why don't scientists trust atoms? Because they make up everything! 😄"
                elif "meaning of life" in user_message.lower():
                    test_response = "According to Douglas Adams in 'The Hitchhiker's Guide to the Galaxy,' the answer is 42. But philosophically, many believe the meaning of life is to find purpose, create connections, learn, grow, and leave the world a little better than you found it."
                else:
                    test_response = f"This is a test response to: '{user_message}'. I'm a demo server, so I can't provide real AI responses, but your question has been received!"
                    
            except:
                test_response = "This is a test response from the demo server."
            
            response = {
                "response": test_response,
                "usage": {"input_tokens": 10, "output_tokens": len(test_response.split()), "total_tokens": 10 + len(test_response.split())},
                "model_info": {"model": "Test", "version": "0.1.0"}
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), TestHandler)
    print("Test server running on http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")