"""
Vercel serverless function to bridge between Render and HuggingFace
Place this in api/analyze.py in your GitHub repo
"""

from gradio_client import Client, handle_file
import base64
from PIL import Image
from io import BytesIO
import tempfile
import os
import json

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

def handler(request):
    """Vercel serverless function handler"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        }
    
    # Handle GET request (health check)
    if request.method == 'GET':
        client = get_client()
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": {
                "status": "healthy",
                "huggingface_connected": client is not None
            }
        }
    
    # Handle POST request (analyze image)
    if request.method == 'POST':
        try:
            # Parse request body
            data = json.loads(request.body)
            image_base64 = data.get('image', '')
            enhance = data.get('enhance', True)
            
            if not image_base64:
                return {
                    "statusCode": 400,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": {
                        "status": "error",
                        "error": "No image provided"
                    }
                }
            
            # Get or create client
            client = get_client()
            if not client:
                return {
                    "statusCode": 503,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": {
                        "status": "error",
                        "error": "Cannot connect to HuggingFace"
                    }
                }
            
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
                
                # Ensure result is JSON serializable
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        pass
                
                return {
                    "statusCode": 200,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": result
                }
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": {
                    "status": "error",
                    "error": str(e)
                }
            }
    
    # Method not allowed
    return {
        "statusCode": 405,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": {
            "status": "error",
            "error": "Method not allowed"
        }
    }
