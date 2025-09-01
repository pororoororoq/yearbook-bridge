# api/analyze.py
from http.server import BaseHTTPRequestHandler
import json
from gradio_client import Client, handle_file
import base64
from PIL import Image
from io import BytesIO
import tempfile
import os

# Global client
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
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
    
    def do_GET(self):
        """Health check with HuggingFace status"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        client = get_client()
        
        message = {
            "status": "healthy",
            "service": "yearbook-bridge-analyze",
            "huggingface_connected": client is not None
        }
        
        self.wfile.write(json.dumps(message).encode())
        return
    
    def do_POST(self):
        """Handle image analysis request"""
        # Get content length
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No data provided"}).encode())
            return
        
        # Read the request body
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            image_base64 = data.get('image', '')
            enhance = data.get('enhance', True)
            
            if not image_base64:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": "No image provided"
                }).encode())
                return
            
            # Get or create client
            client = get_client()
            if not client:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error": "Cannot connect to HuggingFace Space"
                }).encode())
                return
            
            # Decode base64 image
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Save temporarily (gradio_client needs a file)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name, 'JPEG')
                tmp_path = tmp.name
            
            try:
                # Call HuggingFace via gradio_client
                result = client.predict(
                    image=handle_file(tmp_path),
                    enhance_option=enhance,
                    api_name="/predict"
                )
                
                # Send successful response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # If result is string, try to parse it
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        pass
                
                # Send the result
                self.wfile.write(json.dumps(result).encode())
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            # Send error response
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_message = {
                "status": "error",
                "error": str(e)
            }
            
            self.wfile.write(json.dumps(error_message).encode())
            
            import traceback
            traceback.print_exc()
        
        return
