# api/analyze.py
from http.server import BaseHTTPRequestHandler
import json
import requests
import base64
from PIL import Image
from io import BytesIO
import tempfile
import os

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
    
    def do_GET(self):
        """Health check"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        message = {
            "status": "healthy",
            "service": "yearbook-bridge-analyze",
            "method": "direct-http"
        }
        
        self.wfile.write(json.dumps(message).encode())
        return
    
    def do_POST(self):
        """Handle image analysis request"""
        # Get content length
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_error_response(400, "No data provided")
            return
        
        # Read the request body
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            image_base64 = data.get('image', '')
            enhance = data.get('enhance', True)
            
            if not image_base64:
                self.send_error_response(400, "No image provided")
                return
            
            # Decode base64 image
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            
            # Call HuggingFace Space API directly
            result = self.call_huggingface_space(image_bytes, enhance)
            
            if result:
                self.send_success_response(result)
            else:
                self.send_error_response(503, "HuggingFace Space unavailable")
                    
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def call_huggingface_space(self, image_bytes, enhance):
        """Call HuggingFace Space directly via HTTP API"""
        try:
            # Convert image bytes to base64 for API
            image_base64 = base64.b64encode(image_bytes).decode()
            
            # HuggingFace Spaces Gradio API endpoint
            api_url = "https://pororoororoq-photo-analyzer.hf.space/run/predict"
            
            # Prepare the payload
            payload = {
                "data": [
                    f"data:image/png;base64,{image_base64}",
                    enhance
                ]
            }
            
            # Make the request
            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats from Gradio
                if isinstance(result, dict) and 'data' in result:
                    # Gradio typically returns {"data": [...]}
                    data = result.get('data', [])
                    if data and isinstance(data[0], (dict, str)):
                        # If it's a string, try to parse it
                        if isinstance(data[0], str):
                            try:
                                return json.loads(data[0])
                            except:
                                return {"status": "success", "raw_result": data[0]}
                        return data[0]
                
                # Return as-is if it's already in the expected format
                return result
            
            return None
            
        except requests.exceptions.Timeout:
            print("HuggingFace API timeout")
            return None
        except Exception as e:
            print(f"Error calling HuggingFace: {e}")
            return None
    
    def send_success_response(self, data):
        """Send successful JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_error_response(self, code, message):
        """Send error JSON response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        error_data = {
            "status": "error",
            "error": message
        }
        self.wfile.write(json.dumps(error_data).encode())
