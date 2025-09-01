"""
Vercel serverless function to bridge between Render and HuggingFace
Place this in api/analyze.py in your GitHub repo
"""

from http.server import BaseHTTPRequestHandler
import json
from gradio_client import Client, handle_file
import base64
from PIL import Image
from io import BytesIO
import tempfile
import os

# Initialize client globally (reused across invocations)
client = None

def get_client():
    """Get or create Gradio client"""
    global client
    if client is None:
        try:
            client = Client("pororoororoq/photo-analyzer")
            print("Connected to HuggingFace Space")
        except Exception as e:
            print(f"Failed to connect: {e}")
            client = None
    return client

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_POST(self):
        """Handle POST request"""
        # Set CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Get image data
            image_base64 = data.get('image', '')
            enhance = data.get('enhance', True)
            
            if not image_base64:
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": "No image provided"
                }).encode())
                return
            
            # Get or create client
            client = get_client()
            if not client:
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": "Cannot connect to HuggingFace"
                }).encode())
                return
            
            # Decode base64 image
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name, 'JPEG')
                tmp_path = tmp.name
            
            try:
                # Call HuggingFace
                result = client.predict(
                    image=handle_file(tmp_path),
                    enhance_option=enhance,
                    api_name="/predict"
                )
                
                # Send response
                self.end_headers()
                
                # Ensure result is JSON serializable
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        pass
                
                self.wfile.write(json.dumps(result).encode())
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "error": str(e)
            }).encode())
    
    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        client = get_client()
        
        self.wfile.write(json.dumps({
            "status": "healthy",
            "huggingface_connected": client is not None
        }).encode())
